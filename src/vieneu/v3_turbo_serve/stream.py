"""
Streaming with continuous batching: one CUDA graph, rows come and go.
=====================================================================

Why
---
``infer_stream`` used to run the single-sequence engine: no CUDA graph
(~55 ms a frame), a lock for the whole utterance, first audio after
280-360 ms on an RTX 3060 and one listener at a time. The batch engine
(``fused.py``) does a frame in 7-10 ms for up to 32 rows, but its batch is
static: rows are loaded together and the graph runs until the last one is
done, so a request that arrives mid-way would wait for the whole batch.

What
----
* ``StreamFrame`` — the fused frame over ``B`` *slots* instead of a batch.
  A slot is filled by writing one prefilled prompt into the ring KV cache
  right-aligned to the shared write index (``StaticBackbone.load_row``),
  and freed when its row hits EOS or its cap. Sampling settings are per-row
  tensors, so every request keeps its own temperature/top-k/top-p/penalty
  with a single captured graph. Empty slots keep stepping on stale state;
  their output is never read and the sampler tolerates whatever they make.
* ``V3TurboStreamScheduler`` — one worker thread owns the GPU. Each tick it
  admits pending requests (one HF prefill for all of them), replays the
  graph (one frame for every row) and, every ``decode_frames`` ticks, pushes
  the new frames of every row through ONE streaming codec session (the MOSS
  decoder keeps per-row state and lets rows join and leave), handing each
  request its audio through a queue. Idle when no rows are active.

Measured on an RTX 3060 (2026-09-15) through ``infer_stream`` from threads,
``max_streams=16``: first audio 115 ms for one stream, 130-165 ms with 2-8
concurrent, 185 ms median with 16 starting at once, 134 ms for a request
arriving while 15 others play; worst RTF 0.59 at 16 (41 % headroom). The
codec, not the backbone, is the cost: a frame of the graph is 7-10 ms
for any batch, a codec call is ~65 ms + ~2.5 ms per *reserved* slot
whatever the frame count, which is why ``max_streams`` should match the
load (32 slots: one stream already costs RTF 0.70, 32 at once reach RTF
0.93 with 450 ms first audio). The codec's own ``use_cuda_graph`` mode
was 3x cheaper but produced wrong audio (corr 0.2 with the plain decode),
so it is not used.

Caveats
-------
* ``repetition_window`` is the ring size of the on-device history, fixed per
  scheduler; requests asking for another window get the scheduler's.
* No babble/short-row retry: audio is already on its way.
* Each tick holds ``CAPTURE_LOCK`` shared (``_v3_turbo_engine/capture_lock.py``):
  when a batch engine in the same process captures a new graph (~0.5 s, once
  per batch bucket / cache length / sampling setting), the streams pause for it
  instead of breaking the capture.
* A request with bad input (voice, reference codes, sampling) fails alone. If
  the worker itself dies, every waiting request gets the error, ``submit``
  refuses new work and ``error`` says why: restart the process.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vieneu_utils.core_utils import check_sampling
from .._v3_turbo_engine.capture_lock import CAPTURE_LOCK
from .._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
from .fused import GpuRepHistory, StaticBackbone, acoustic_frame_gpu, sample_gpu_rows

logger = logging.getLogger(__name__)

SAMPLES_PER_FRAME = 3840   # 48 kHz / 12.5 frames per second


class StreamFrame:
    """The fused frame over ``B`` slots that rows enter (``admit``) and leave.

    Device state (all static, read by the captured graph):
      ``h`` (B, H)               backbone hidden feeding the acoustic decoder
      ``codes`` (R, B, n_vq)     ring of every frame's codes, at ``frame mod R``
      ``finished`` (B,)          slot is empty or its row is done
      ``length`` (B,)            global frame count at which the row finished
      ``caps`` (B,)              global frame count at which the row is cut
      ``frame`` ()               global frame counter (also the codes ring index)
      sampling (B, 1)            temperature / top_k / top_p / penalty per row
      ``spk`` (B, D)             speaker anchor re-added at every step (or None)
    """

    def __init__(self, model, B: int, max_len: int, *, repetition_window: int = DEFAULT_REP_WINDOW,
                 codes_ring: int = 512, warmup: int = 3):
        self.model = model
        cfg = model.config
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        self.B, self.max_len, self.ring = int(B), int(max_len), int(codes_ring)
        n_vq, H = cfg.n_vq, cfg.hidden_size
        self.n_vq = n_vq
        self.vocab = int(cfg.audio_vocab_size)
        self.bb = StaticBackbone(model, self.B, self.max_len)
        self.rep = GpuRepHistory(self.B, n_vq, cfg.audio_vocab_size, repetition_window, dev)
        self.h = torch.zeros(self.B, H, device=dev, dtype=dt)
        self.codes = torch.zeros(self.ring, self.B, n_vq, device=dev, dtype=torch.long)
        self.finished = torch.ones(self.B, device=dev, dtype=torch.bool)
        self.length = torch.zeros(self.B, device=dev, dtype=torch.long)
        self.caps = torch.zeros(self.B, device=dev, dtype=torch.long)
        self.frame = torch.zeros((), device=dev, dtype=torch.long)
        self.temperature = torch.ones(self.B, 1, device=dev)
        self.top_k = torch.full((self.B, 1), 25, device=dev, dtype=torch.long)
        self.top_p = torch.full((self.B, 1), 0.95, device=dev)
        self.penalty = torch.ones(self.B, 1, device=dev)
        spk_dim = getattr(cfg, "speaker_embedding_dim", None)
        self.spk = (torch.zeros(self.B, spk_dim, device=dev, dtype=torch.float32)
                    if model.xvec_proj is not None and spk_dim else None)
        self._pad = cfg.audio_pad_token_id
        self._sgs = cfg.speech_generation_start_token_id

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self._body()
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        """Empty every slot (after capture, or to start over)."""
        self.h.zero_()
        self.codes.zero_()
        self.finished.fill_(True)
        self.length.zero_()
        self.caps.zero_()
        self.frame.zero_()
        self.rep.reset()
        self.bb.cur.zero_()
        self.bb.bias.fill_(float("-inf"))

    def _sample(self, logits: torch.Tensor, ch: int) -> torch.Tensor:
        logits = self.rep.penalise(logits, ch, self.penalty)
        code = sample_gpu_rows(logits, self.temperature, self.top_k, self.top_p)
        self.rep.add(ch, code)
        return code

    def _body(self) -> None:
        codes, is_eos = acoustic_frame_gpu(
            self.model, self.h, temperature=1.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0, rep=None, sample_fn=self._sample,
        )
        self.rep.advance()
        self.codes.index_copy_(0, torch.remainder(self.frame, self.ring).view(1), codes.unsqueeze(0))
        n = self.frame + 1
        done_now = ~self.finished & (is_eos | (n >= self.caps))
        self.length.copy_(torch.where(done_now, n, self.length))
        self.finished.logical_or_(done_now)
        self.frame += 1
        B, n_vq = codes.shape
        slot = torch.full((B, 1, n_vq + 1), self._pad, dtype=torch.long, device=codes.device)
        slot[:, :, 0] = self._sgs
        slot[:, 0, 1:] = codes
        se = self.model._build_inputs_embeds(slot, speaker_emb=self.spk)
        self.h.copy_(self.bb.step(se))

    @torch.no_grad()
    def admit(self, b: int, keys: List[torch.Tensor], values: List[torch.Tensor], h_last: torch.Tensor,
              spk: Optional[torch.Tensor], cap: int, *, temperature: float, top_k: int, top_p: float,
              repetition_penalty: float) -> int:
        """Fill slot ``b`` with a prefilled prompt; returns the global frame
        index of the row's first frame. ``cap`` is its own frame budget."""
        T = keys[0].shape[1]
        if T + cap + 2 > self.max_len:
            raise ValueError(f"prompt ({T} tokens) + {cap} frames exceeds the {self.max_len}-token cache")
        self.bb.load_row(b, keys, values)
        self.h[b].copy_(h_last.to(self.h.dtype))
        self.rep.reset_rows(torch.tensor([b], device=self.h.device))
        self.temperature[b] = float(temperature)
        # top_k >= vocab keeps every code and <= 0 disables top-k, so clamping
        # changes nothing — except that a huge value can't overflow the int64 slot.
        self.top_k[b] = min(max(int(top_k), 0), self.vocab)
        self.top_p[b] = float(top_p)
        self.penalty[b] = float(repetition_penalty)
        if self.spk is not None:
            if spk is None:
                raise ValueError("this model needs a speaker anchor for every row")
            self.spk[b].copy_(spk.view(-1).to(self.spk.dtype))
        f0 = int(self.frame)
        self.caps[b] = f0 + int(cap)
        self.length[b] = 0
        self.finished[b] = False
        return f0

    def replay(self) -> None:
        self.graph.replay()

    def status(self) -> List[int]:
        """``[finished(B) as 0/1..., length(B)...]`` in one host read."""
        return torch.cat([self.finished.long(), self.length]).tolist()

    def recent_codes(self, k: int) -> torch.Tensor:
        """The last ``k`` global frames of every slot, ``(k, B, n_vq)`` on the host."""
        f = int(self.frame)
        idx = torch.remainder(torch.arange(f - k, f, device=self.codes.device), self.ring)
        return self.codes.index_select(0, idx).cpu()


class StreamHandle:
    """A submitted request: iterate for its audio (float32 @ 48 kHz), in order.

    ``gen_done`` is set the moment the last frame is generated (before its
    audio is decoded) — the caller's cue to submit the next chunk. Closing
    the iterator (client gone) cancels the row.
    """

    def __init__(self, sched: "V3TurboStreamScheduler", req: Dict[str, Any]):
        self._sched = sched
        self.req = req
        self.q: "queue.Queue" = queue.Queue()
        self.gen_done = threading.Event()
        self.cancelled = False
        # Worker-owned bookkeeping.
        self.slot: Optional[int] = None
        self.f0 = 0              # global frame of the row's first frame
        self.decoded_upto = 0    # global frame up to which audio was emitted
        self.finished_at: Optional[int] = None   # global frame count at EOS/cap
        self.frames = 0          # frames kept (EOS frame included)

    def __iter__(self):
        try:
            while True:
                # The worker's next tick may be queued behind a capture that
                # waits for this thread: never wait here holding CAPTURE_LOCK.
                CAPTURE_LOCK.ensure_not_held("infer_stream")
                item = self.q.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.close()

    def close(self) -> None:
        self.cancelled = True
        self.gen_done.set()


class V3TurboStreamScheduler:
    """Serves many ``infer_stream`` calls from one graph; see the module doc."""

    def __init__(self, batch_engine, *, max_streams: int = 16, decode_frames: int = 4,
                 leadin_frames: int = 2, repetition_window: int = DEFAULT_REP_WINDOW,
                 max_admit_per_tick: int = 8):
        self.be = batch_engine
        self.tts = batch_engine.tts
        self.model = batch_engine.model
        self.cfg = batch_engine.config
        self.tok = self.tts.audio_tokenizer
        self.B = max(1, int(max_streams))
        self.K = max(1, int(decode_frames))
        # A row that has not sent audio yet gets an extra codec call once it
        # has this many frames instead of waiting for the next K boundary.
        # A codec call costs ~65 ms + ~3 ms per reserved slot on a 3060
        # regardless of frame count, so this buys ~2 frames (~20 ms) of
        # first-audio latency per newcomer for one extra call.
        self.leadin = max(1, min(int(leadin_frames), self.K))
        # Note: an idle GPU drops to its lowest power state within ~2 s (RTX
        # 3060: P8, 210 MHz) and the first request after that pays +100..300 ms
        # on its first chunk. Periodic small kernels from here do NOT hold the
        # clocks (tried a 2048² matmul and whole-graph replays at 4-10 Hz: the
        # driver still parks the GPU), so this is left to the host — lock the
        # clocks (``nvidia-smi -lgc``) or set "prefer maximum performance".
        self.max_admit = max(1, int(max_admit_per_tick))
        self.repetition_window = int(repetition_window)
        # Built in the caller's thread while batch runs may be going on in
        # others: the capture keeps every other v3 Turbo GPU job out.
        with CAPTURE_LOCK.exclusive():
            self.frame = StreamFrame(self.model, self.B, int(self.cfg.max_position_embeddings),
                                     repetition_window=self.repetition_window)
        self.tick = 0
        self._slots: List[Optional[StreamHandle]] = [None] * self.B
        self._codec_rows: List[StreamHandle] = []   # order the codec session knows
        self._pending: "queue.Queue[StreamHandle]" = queue.Queue()
        self._cv = threading.Condition()
        self._running = True
        self._error: Optional[BaseException] = None
        self._codec_fresh = True
        self._thread = threading.Thread(target=self._run, name="v3turbo-stream", daemon=True)
        self._thread.start()

    # ── public ────────────────────────────────────────────────────────────────
    @property
    def error(self) -> Optional[BaseException]:
        """Why the worker died, or ``None`` while it runs. After a crash
        ``submit`` refuses all work until the process restarts."""
        return self._error

    def submit(self, *, phonemes: str, speaker_emb, ref_codes, use_ref_codes: bool = True,
               temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
               repetition_penalty: float = 1.2, max_new_frames: int = 300,
               repetition_window: Optional[int] = None) -> StreamHandle:
        # Checked in the caller's thread: a bad value must fail this call, not
        # the worker every stream shares (NaN reaching multinomial on CUDA is a
        # device-side assert that poisons the whole process).
        temperature, top_k, top_p, repetition_penalty = check_sampling(
            temperature, top_k, top_p, repetition_penalty)
        if repetition_window is not None and int(repetition_window) != self.repetition_window:
            logger.debug("stream: repetition_window=%s ignored, scheduler uses %d",
                         repetition_window, self.repetition_window)
        h = StreamHandle(self, dict(
            phonemes=phonemes, speaker_emb=speaker_emb, ref_codes=ref_codes,
            use_ref_codes=use_ref_codes, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, max_new_frames=max(0, int(max_new_frames)),
        ))
        with self._cv:   # same lock as _shutdown: nothing is queued after its drain
            if self._error is not None:
                raise RuntimeError("stream scheduler is down") from self._error
            if not self._running:
                raise RuntimeError("stream scheduler is closed")
            self._pending.put(h)
            self._cv.notify()
        return h

    @property
    def n_active(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    def close(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify()
        self._thread.join(timeout=5)

    # ── worker ────────────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            while True:
                with self._cv:
                    while self._running and self._pending.empty() and self.n_active == 0:
                        self._cv.wait()
                    if not self._running:
                        break
                # One tick is GPU work (prefill, replay, codec): a capture in
                # another thread (a batch engine's new graph) waits for it, and
                # the next tick waits for the capture.
                with CAPTURE_LOCK.shared():
                    self._admit()
                    if self.n_active:
                        self._step()
        except BaseException as e:   # noqa: BLE001 — everything waiting must hear about it
            logger.exception("stream scheduler died")
            self._shutdown(e, crashed=True)
        else:
            # close(): no request may be left waiting on a queue nobody fills.
            self._shutdown(RuntimeError("stream scheduler closed"), crashed=False)
        finally:
            self._reset_codec()

    def _shutdown(self, err: BaseException, *, crashed: bool) -> None:
        """Stop taking work and hand ``err`` to every request still waiting —
        the running rows and the queue (``_admit`` tells the ones it holds)."""
        with self._cv:
            if crashed:
                self._error = err
            self._running = False
            waiting = [h for h in self._slots if h is not None] + self._drain_pending()
        for h in waiting:
            h.q.put(err)

    def _drain_pending(self) -> List[StreamHandle]:
        out = []
        while True:
            try:
                out.append(self._pending.get_nowait())
            except queue.Empty:
                return out

    @torch.no_grad()
    def _admit(self) -> None:
        free = [b for b, s in enumerate(self._slots) if s is None]
        todo: List[StreamHandle] = []
        # At most one request per free slot; the rest wait in the queue. (Taking
        # up to ``max_admit`` regardless ran ``free`` dry, and the IndexError
        # killed the worker whenever requests outnumbered free slots.)
        while len(todo) < min(len(free), self.max_admit):
            try:
                h = self._pending.get_nowait()
            except queue.Empty:
                break
            if h.cancelled:
                h.q.put(None)
                continue
            todo.append(h)
        if not todo:
            return
        # Off the queue but not in a slot: no one else knows these requests, so
        # if the worker dies in here they must hear it from here.
        unanswered = list(todo)

        def reject(h: StreamHandle, err: BaseException) -> None:
            # A request's own bad input (voice, reference codes, ...) fails that
            # request only; the worker keeps serving everyone else.
            h.q.put(err)
            unanswered.remove(h)

        try:
            batch: List[StreamHandle] = []
            embeds, spks = [], []
            for h in todo:
                try:
                    e = self.be._prompt_embeds(h.req)
                    s = self.tts._resolve_speaker_emb(h.req.get("speaker_emb"))
                except Exception as err:   # noqa: BLE001
                    reject(h, err)
                    continue
                batch.append(h)
                embeds.append(e)
                spks.append(s)
            if not batch:
                return
            try:
                last_h, cache, mask, _pos = self.be.bb.prefill(embeds)
            except Exception as err:   # noqa: BLE001 — e.g. out of memory on long prompts
                for h in batch:
                    reject(h, err)
                return
            T = mask.shape[1]
            for i, h in enumerate(batch):
                Ti = embeds[i].shape[0]
                keys, values = [], []
                for l in range(len(self.frame.bb.layers)):
                    layer = cache.layers[l] if hasattr(cache, "layers") else None
                    k = layer.keys if layer is not None else cache.key_cache[l]
                    v = layer.values if layer is not None else cache.value_cache[l]
                    keys.append(k[i, :, T - Ti:T])
                    values.append(v[i, :, T - Ti:T])
                b = free.pop(0)
                r = h.req
                try:
                    h.f0 = self.frame.admit(
                        b, keys, values, last_h[i], spks[i], r["max_new_frames"],
                        temperature=r["temperature"], top_k=r["top_k"], top_p=r["top_p"],
                        repetition_penalty=r["repetition_penalty"],
                    )
                except Exception as err:   # noqa: BLE001
                    free.insert(0, b)
                    reject(h, err)
                    continue
                h.slot = b
                h.decoded_upto = h.f0
                self._slots[b] = h
                unanswered.remove(h)
                if r["max_new_frames"] == 0:
                    # Nothing to generate: finished before its first frame.
                    h.finished_at = h.f0
                    h.gen_done.set()
                self._codec_rows.append(h)
        except BaseException as err:
            for h in unanswered:
                h.q.put(err)
            raise

    def _step(self) -> None:
        self.frame.replay()
        self.tick += 1
        st = self.frame.status()
        fin, length = st[:self.B], st[self.B:]
        for b, h in enumerate(self._slots):
            if h is None or h.finished_at is not None:
                continue
            if h.cancelled:
                h.finished_at = self.tick
            elif fin[b]:
                h.finished_at = length[b]
                h.frames = h.finished_at - h.f0
                h.gen_done.set()
        if self.tick % self.K == 0 or any(
            h is not None and h.decoded_upto == h.f0 and self.tick - h.f0 >= self.leadin
            for h in self._slots
        ):
            self._decode()

    @torch.no_grad()
    def _decode(self) -> None:
        if not self._codec_rows:
            return
        blk = self.frame.recent_codes(self.K)            # (K, B, n_vq), global frames [tick-K, tick)
        base = self.tick - self.K
        dev = self.frame.codes.device
        rows, real, finalize = [], [], []
        for pos, h in enumerate(self._codec_rows):
            hi = self.tick if h.finished_at is None else min(h.finished_at, self.tick)
            lo = h.decoded_upto
            k = hi - lo
            if k > 0:
                rows.append(blk[lo - base:hi - base, h.slot].permute(1, 0).to(dev))
            else:
                # A row finalised right on a decode boundary has nothing new;
                # the codec still wants a frame from it (its output is dropped —
                # the decoder has no lookahead, so earlier audio is unaffected).
                rows.append(torch.zeros(self.frame.n_vq, 1, dtype=torch.long, device=dev))
            real.append(k)
            h.decoded_upto = hi
            if h.finished_at is not None and hi >= h.finished_at:
                finalize.append(pos)
        out = self.tok.batch_decode(
            rows, num_quantizers=self.frame.n_vq, streaming=True, max_batch_size=self.B,
            reset_stream=self._codec_fresh, finalize_indices=finalize or None,
        )
        self._codec_fresh = False
        audio = out.audio.float().mean(1).cpu().numpy()   # (n, L)
        lens = (out.audio_lengths.tolist() if getattr(out, "audio_lengths", None) is not None
                else [audio.shape[-1]] * len(rows))
        keep: List[StreamHandle] = []
        for pos, h in enumerate(self._codec_rows):
            n = min(int(lens[pos]), real[pos] * SAMPLES_PER_FRAME)
            if n > 0 and not h.cancelled:
                h.q.put(audio[pos, :n].copy())
            if pos in finalize:
                h.q.put(None)
                self._slots[h.slot] = None
            else:
                keep.append(h)
        self._codec_rows = keep
        if not keep:
            self._reset_codec()

    def _reset_codec(self) -> None:
        reset = getattr(self.tok, "_reset_batch_decode_streaming_state", None)
        if callable(reset):
            try:
                with CAPTURE_LOCK.shared():   # frees the codec's device state
                    reset()
            except Exception:   # noqa: BLE001
                pass
        self._codec_fresh = True
