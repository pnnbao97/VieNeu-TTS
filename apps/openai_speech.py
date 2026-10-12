"""
VieNeu-TTS — OpenAI-compatible speech API (streaming), CPU or GPU.
==================================================================
Drop-in for ``POST /v1/audio/speech``: any OpenAI SDK / Pipecat / LiveKit /
Vercel AI client works by changing ``base_url``. Audio streams as it is
generated — the first bytes leave ~115 ms after the request on an RTX 3060,
~600 ms on a 6-core CPU (see docs/streaming.vi.md for every number).

    uv run python -m apps.openai_speech                 # http://127.0.0.1:8000
    VIENEU_BACKEND=onnx uv run python -m apps.openai_speech   # force CPU

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
    with client.audio.speech.with_streaming_response.create(
        model="vieneu-v3-turbo", voice="Mai Anh", input="Xin chào! Đây là chế độ streaming của VieNeu, phát tới đâu nghe tới đó.",
        response_format="pcm",              # 48 kHz s16le mono (see `sample_rate` below)
    ) as r:
        for chunk in r.iter_bytes(4096):
            play(chunk)

Request body (OpenAI fields + a few extras, all optional but ``input``):
    model            any string; reported back as-is
    input            text, chunked internally at ``max_chars`` (256)
    voice            preset name (GET /v1/voices) or one added with POST /v1/voices
    response_format  "pcm" (s16le, no header) | "wav" (header with unknown length,
                     plays as it streams). mp3/opus/aac/flac → 400.
    stream_format    "audio" (default: raw chunked body) | "sse" (Server-Sent
                     Events: {"type":"speech.audio.delta","audio":"<base64>"} ...
                     {"type":"speech.audio.done","usage":{...}})
    speed            accepted for compatibility, ignored (header X-VieNeu-Ignored)
    sample_rate      48000 (native) | 24000 (OpenAI's pcm rate) | 16000 | 8000 —
                     resampled on the fly (soxr), per chunk, no extra latency
    temperature, top_k, top_p, repetition_penalty, max_chars   sampling extras,
                     range-checked (out of range → 400): 0-2 / 1-1024 / (0, 1] / 1-2 / 64-512

POST /v1/voices refuses (409) the name of a built-in voice or alias, so a client
cannot swap out a voice other clients rely on (e.g. the default one).

Concurrency: GPU serves ``VIENEU_MAX_STREAMS`` streams at once (continuous
batching, default 16); CPU serves one (fp32) or two (int8) — the ONNX engine
interleaves them frame by frame but they share the cores.
Extra requests wait up to ``VIENEU_QUEUE_TIMEOUT`` s in a queue of at most
``VIENEU_QUEUE`` entries, then get 429 with Retry-After.

Environment:
    VIENEU_BACKEND=auto|onnx|pytorch   VIENEU_DEVICE=auto|cuda|cpu
    VIENEU_PRECISION=fp32|int8 (CPU)   VIENEU_ONNX_DIR=... (local ONNX export)
    VIENEU_MAX_STREAMS=16 (GPU)        VIENEU_QUEUE=16   VIENEU_QUEUE_TIMEOUT=10
    VIENEU_API_KEY=...  (Bearer auth; unset = open — logged as a warning when
                        HOST is not a loopback address)
    VIENEU_WATERMARK=1                 HOST=127.0.0.1  PORT=8000
                                       (HOST=0.0.0.0 to serve other machines; the
                                       Docker profiles set it)
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import re
import struct
import threading
import time
import uuid
import wave
from typing import Any, Iterator, Optional

import anyio

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from vieneu import Vieneu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vieneu.api")

SAMPLE_RATE = 48_000
MODEL_ID = "vieneu-v3-turbo"
RATES = (48_000, 24_000, 16_000, 8_000)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ── engine + admission ───────────────────────────────────────────────────────

class Engine:
    """The model plus the admission gate (how many streams run at once)."""

    def __init__(self):
        backend = os.environ.get("VIENEU_BACKEND", "auto")
        device = os.environ.get("VIENEU_DEVICE", "auto")
        kw: dict = dict(backend=backend, device=device,
                        precision=os.environ.get("VIENEU_PRECISION", "fp32"),
                        onnx_dir=os.environ.get("VIENEU_ONNX_DIR") or None,
                        max_streams=_env_int("VIENEU_MAX_STREAMS", 16))
        log.info("⏳ loading VieNeu-TTS v3 Turbo (backend=%s device=%s)", backend, device)
        t = time.perf_counter()
        self.tts = Vieneu(mode="v3turbo", **kw)
        self.backend = self.tts.backend
        # The voices and aliases that exist before any client enrolls one;
        # POST /v1/voices may not replace them.
        self.builtin_voices = frozenset(self.tts._preset_voices) | frozenset(self.tts._voice_aliases)
        self.watermark = os.environ.get("VIENEU_WATERMARK", "1") != "0"
        # GPU: the scheduler batches every stream. CPU: the ONNX engine interleaves
        # calls frame by frame, but they share the cores — measured on a 6-core
        # desktop, fp32 keeps one stream real-time (RTF 0.58; two → 1.19), int8
        # two (RTF 0.35; two → 0.67). VIENEU_MAX_STREAMS overrides either.
        self.sched = self.tts._get_stream_scheduler() if self.backend == "pytorch" else None
        if self.sched is not None:
            self.max_streams = self.sched.B
        else:
            cpu_default = 2 if kw["precision"] == "int8" else 1
            self.max_streams = _env_int("VIENEU_MAX_STREAMS", cpu_default) if "VIENEU_MAX_STREAMS" in os.environ else cpu_default
        self.max_queue = _env_int("VIENEU_QUEUE", self.max_streams)
        self.queue_timeout = float(os.environ.get("VIENEU_QUEUE_TIMEOUT", "10"))
        self._gate = threading.BoundedSemaphore(self.max_streams)
        self._lock = threading.Lock()
        self.active = 0
        self.waiting = 0
        # Warm-up: capture the CUDA graphs / load the ONNX graphs before the
        # first client pays for it.
        for _ in self.tts.infer_stream("Xin chào.", apply_watermark=False):
            pass
        log.info("✅ ready in %.1fs: backend=%s max_streams=%d queue=%d", time.perf_counter() - t,
                 self.backend, self.max_streams, self.max_queue)

    def acquire(self) -> None:
        with self._lock:
            if self.waiting >= self.max_queue:
                raise HTTPException(429, "server busy: queue full", headers={"Retry-After": "1"})
            self.waiting += 1
        try:
            if not self._gate.acquire(timeout=self.queue_timeout):
                raise HTTPException(429, "server busy: timed out waiting for a stream slot",
                                    headers={"Retry-After": "2"})
        finally:
            with self._lock:
                self.waiting -= 1
        with self._lock:
            self.active += 1

    def release(self) -> None:
        with self._lock:
            self.active -= 1
        self._gate.release()

    def stream_error(self) -> Optional[BaseException]:
        """Why the GPU stream scheduler died, or None. A dead scheduler refuses
        every request until the process restarts."""
        return getattr(self.sched, "error", None) if self.sched is not None else None

    def voices(self) -> list:
        out = []
        for name, v in self.tts._preset_voices.items():
            out.append({"id": name, "name": name, "description": v.get("description", ""),
                        "gender": v.get("gender", ""), "featured": v.get("featured"),
                        "aliases": list(v.get("aliases") or [])})
        return out


class _Slot:
    """One ``Engine`` stream slot, released exactly once: by the stream's own
    ``finally``, or by ``_SpeechResponse`` when the response ends without the
    body ever being iterated (client gone while queued)."""

    def __init__(self, eng: Engine):
        self._eng = eng
        self._lock = threading.Lock()
        self._held = True

    def release(self) -> None:
        with self._lock:
            held, self._held = self._held, False
        if held:
            self._eng.release()


class _SpeechResponse(StreamingResponse):
    """A ``StreamingResponse`` that closes its body the moment the response ends
    — done, failed or client gone — instead of leaving that to the garbage
    collector. Starlette drops a sync body without closing it when the client
    disconnects, and the abandoned generators sit in a reference cycle, so
    their ``finally`` (freeing the slot, cancelling the scheduler row) could
    wait minutes on an idle server; with every slot stuck, all requests got 429.
    """

    def __init__(self, body: Iterator[bytes], slot: _Slot, **kw: Any):
        super().__init__(body, **kw)
        self._body = body
        self._slot = slot

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # close() runs the generators' finally blocks, so keep it off the
            # event loop. By now no thread is inside the body: anyio waits for
            # a cancelled to_thread call to return.
            await anyio.to_thread.run_sync(self._close)

    def _close(self) -> None:
        try:
            self._body.close()
        except Exception:   # noqa: BLE001 — the slot must be freed regardless
            log.exception("closing the speech body failed")
        finally:
            self._slot.release()


ENGINE: Optional[Engine] = None


def engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine()
    return ENGINE


# ── auth / errors in OpenAI's shape ───────────────────────────────────────────

def _auth(authorization: Optional[str] = Header(default=None)) -> None:
    key = os.environ.get("VIENEU_API_KEY")
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "invalid api key")


def _error(status: int, message: str, code: Optional[str] = None, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error",
                                   "code": code or status}}, status_code=status, headers=headers)


app = FastAPI(title="VieNeu-TTS speech API", version="1.0")


@app.exception_handler(HTTPException)
async def _http_exc(_req: Request, exc: HTTPException):
    return _error(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _invalid_request(_req: Request, exc: RequestValidationError):
    # Field and reason only: FastAPI's default echoes the rejected value, and a
    # NaN (which json.loads accepts) cannot be encoded back — the 422 became a 500.
    errs = exc.errors()
    first = errs[0] if errs else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p not in ("body", "query", "header"))
    msg = first.get("msg", "invalid request")
    return _error(400, f"{field}: {msg}" if field else msg)


@app.on_event("startup")
def _startup() -> None:
    engine()


# ── audio helpers ────────────────────────────────────────────────────────────

def _pcm16(x: np.ndarray) -> bytes:
    return (np.asarray(x, dtype=np.float32) * 32767).clip(-32768, 32767).astype("<i2").tobytes()


def _wav_header(rate: int) -> bytes:
    """A WAV header claiming an unknown (huge) length, so players start at once."""
    h = io.BytesIO()
    with wave.open(h, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.setnframes(0)
    b = bytearray(h.getvalue())
    struct.pack_into("<I", b, 4, 0xFFFFFFFF)      # RIFF size
    struct.pack_into("<I", b, len(b) - 4, 0xFFFFFFFF)   # data size
    return bytes(b)


class _Resampler:
    """48 kHz → ``rate`` chunk by chunk with carried state (soxr), no added latency."""

    def __init__(self, rate: int):
        self.rate = rate
        self._rs = None
        if rate != SAMPLE_RATE:
            import soxr
            self._rs = soxr.ResampleStream(SAMPLE_RATE, rate, 1, dtype="float32", quality="HQ")

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        if self._rs is None:
            return x
        return self._rs.resample_chunk(np.asarray(x, dtype=np.float32), last=last)


# ── /v1/audio/speech ─────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str = Field(min_length=1, max_length=20_000)
    voice: Optional[str] = None
    response_format: str = "wav"
    stream_format: str = "audio"
    speed: Optional[float] = None
    instructions: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    # Bounded so one request cannot break the GPU scheduler every stream shares:
    # e.g. a huge top_k overflowed its int64 slot and NaN reached multinomial.
    temperature: float = Field(0.8, ge=0.0, le=2.0, allow_inf_nan=False)
    top_k: int = Field(25, ge=1, le=1024)   # 1024 = the codec's vocabulary
    top_p: float = Field(0.95, gt=0.0, le=1.0, allow_inf_nan=False)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, allow_inf_nan=False)
    max_chars: int = Field(256, ge=64, le=512)


def _speech_chunks(eng: Engine, req: SpeechRequest, rid: str, slot: _Slot) -> Iterator[np.ndarray]:
    """float32 chunks at ``req.sample_rate``; logs TTFA / RTF; holds one stream slot."""
    t0 = time.perf_counter()
    first = None
    emitted = 0
    rs = _Resampler(req.sample_rate)
    audio = eng.tts.infer_stream(
        req.input, voice=req.voice or None, apply_watermark=eng.watermark,
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        repetition_penalty=req.repetition_penalty, max_chars=req.max_chars,
    )
    try:
        for chunk in audio:
            if chunk is None or len(chunk) == 0:
                continue
            if first is None:
                first = time.perf_counter() - t0
            emitted += len(chunk)
            out = rs(chunk)
            if len(out):
                yield out
        tail = rs(np.zeros(0, np.float32), last=True)
        if len(tail):
            yield tail
    finally:
        audio.close()   # cancels the scheduler rows now, not when collected
        slot.release()
        total = time.perf_counter() - t0
        audio_s = emitted / SAMPLE_RATE
        log.info("%s done: ttfa=%s total=%.2fs audio=%.2fs rtf=%s active=%d", rid,
                 f"{first*1000:.0f}ms" if first is not None else "-", total, audio_s,
                 f"{total/audio_s:.2f}" if audio_s else "-", eng.active)


def _audio_body(chunks: Iterator[np.ndarray], fmt: str, rate: int) -> Iterator[bytes]:
    try:
        if fmt == "wav":
            yield _wav_header(rate)
        for c in chunks:
            yield _pcm16(c)
    finally:
        chunks.close()


def _sse_body(chunks: Iterator[np.ndarray], fmt: str, rate: int) -> Iterator[bytes]:
    def ev(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
    n = 0
    try:
        if fmt == "wav":
            yield ev({"type": "speech.audio.delta", "audio": base64.b64encode(_wav_header(rate)).decode()})
        for c in chunks:
            n += len(c)
            yield ev({"type": "speech.audio.delta", "audio": base64.b64encode(_pcm16(c)).decode()})
        yield ev({"type": "speech.audio.done",
                  "usage": {"output_samples": n, "sample_rate": rate, "seconds": round(n / rate, 3)}})
    finally:
        chunks.close()


@app.post("/v1/audio/speech", dependencies=[Depends(_auth)])
def speech(req: SpeechRequest):
    eng = engine()
    fmt = req.response_format.lower()
    if fmt not in ("pcm", "wav"):
        raise HTTPException(400, f"response_format '{fmt}' not supported; use 'pcm' or 'wav'")
    if req.stream_format not in ("audio", "sse"):
        raise HTTPException(400, "stream_format must be 'audio' or 'sse'")
    if req.sample_rate not in RATES:
        raise HTTPException(400, f"sample_rate must be one of {RATES}")
    if req.voice and eng.tts.resolve_voice_name(req.voice) is None:
        raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /v1/voices")
    if eng.stream_error() is not None:
        # Say so now: once the 200 headers are out, a failure only truncates the body.
        raise HTTPException(503, "speech engine is down; restart the server (see /health)")
    rid = f"spk-{uuid.uuid4().hex[:8]}"
    eng.acquire()   # 429 if the server is full; released when the stream ends
    slot = _Slot(eng)
    chunks = _speech_chunks(eng, req, rid, slot)
    headers = {"X-Request-Id": rid, "X-Sample-Rate": str(req.sample_rate), "Cache-Control": "no-store"}
    ignored = [k for k in ("speed", "instructions") if getattr(req, k) is not None]
    if ignored:
        headers["X-VieNeu-Ignored"] = ",".join(ignored)
    if req.stream_format == "sse":
        return _SpeechResponse(_sse_body(chunks, fmt, req.sample_rate), slot,
                               media_type="text/event-stream", headers=headers)
    media = "audio/wav" if fmt == "wav" else "audio/pcm"
    return _SpeechResponse(_audio_body(chunks, fmt, req.sample_rate), slot, media_type=media, headers=headers)


# ── discovery ────────────────────────────────────────────────────────────────

@app.get("/v1/models", dependencies=[Depends(_auth)])
def models():
    eng = engine()
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "vieneu",
        "sample_rate": SAMPLE_RATE, "backend": eng.backend,
        "max_streams": eng.max_streams, "response_formats": ["pcm", "wav"],
        "stream_formats": ["audio", "sse"], "sample_rates": list(RATES),
    }]}


@app.get("/v1/voices", dependencies=[Depends(_auth)])
def voices():
    return {"object": "list", "data": engine().voices()}


# A voice name is a label: letters/digits (any script), spaces, dots, dashes,
# underscores; 1-64 chars, no control characters. It ends up in logs and in the
# voices file, so anything else is rejected at the boundary.
_VOICE_NAME = re.compile(r"[^\W_][\w .\-]{0,63}")
_CLIP_EXTS = {".wav": ".wav", ".mp3": ".mp3", ".flac": ".flac", ".ogg": ".ogg", ".m4a": ".m4a"}
_MAX_CLIP_BYTES = 20 * 1024 * 1024


@app.post("/v1/voices", dependencies=[Depends(_auth)])
def add_voice(name: str = Form(...), file: UploadFile = File(...), denoise: bool = Form(True),
              description: str = Form("")):
    """Enroll a 3-8 s reference clip as ``voice=name`` (in memory, for this process).

    Plain ``def``: FastAPI runs it in a thread, so the blocking upload read,
    temp file and enrolment do not stall the event loop.
    """
    import tempfile
    eng = engine()
    if not _VOICE_NAME.fullmatch(name):
        raise HTTPException(400, "voice name: 1-64 letters, digits, spaces, '.', '-' or '_'")
    if not _VOICE_NAME.fullmatch(description or "x"):
        raise HTTPException(400, "description: 1-64 letters, digits, spaces, '.', '-' or '_'")
    if name in eng.builtin_voices:
        # add_voice would replace it for every client, including the default voice.
        raise HTTPException(409, f"'{name}' is a built-in voice; choose another name")
    # The suffix only picks the decoder; it is mapped to a constant, never taken from the client.
    ext = _CLIP_EXTS.get(os.path.splitext(file.filename or "")[1].lower(), ".wav")
    data = file.file.read(_MAX_CLIP_BYTES + 1)
    if len(data) > _MAX_CLIP_BYTES:
        raise HTTPException(413, "reference clip larger than 20 MB")
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        eng.tts.add_voice(name, path, denoise=denoise, description=description)
    except HTTPException:
        raise
    except Exception as e:   # noqa: BLE001
        raise HTTPException(400, f"could not enroll voice: {e}")
    finally:
        os.unlink(path)
    return {"id": name, "name": name, "description": description}


@app.get("/health")
def health():
    eng = engine()
    body = {"status": "ok", "backend": eng.backend, "max_streams": eng.max_streams,
            "active": eng.active, "waiting": eng.waiting, "sample_rate": SAMPLE_RATE}
    err = eng.stream_error()
    if err is not None:
        # 503 so a container healthcheck restarts us: the scheduler does not recover.
        return JSONResponse({**body, "status": "error", "error": f"stream scheduler is down: {err!r}"},
                            status_code=503)
    return body


def main() -> None:
    # Loopback unless told otherwise: the API is open when VIENEU_API_KEY is unset.
    host = os.environ.get("HOST", "127.0.0.1")
    port = _env_int("PORT", 8000)
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("VIENEU_API_KEY"):
        log.warning("⚠️ listening on %s:%d without VIENEU_API_KEY: anyone who can reach this port "
                    "can use the API and enroll voices", host, port)
    # One worker: the model (and on GPU the scheduler) is process-local.
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
