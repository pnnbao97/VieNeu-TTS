"""
VieNeu-TTS v3 Nano backend (ONNX, CPU-only, torch-free).
=========================================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3nano")                       # v3 Turbo stays the default
    wav = tts.infer("Xin chào", voice="Minh Quân")         # preset voice
    tts.save(wav, "out.wav")                          # 24 kHz

v3 Nano is a 48M-parameter flow-matching model for machines where v3 Turbo is too
slow — old laptops, mini PCs, single-board computers. What you trade for that:

  * 24 kHz output (Turbo: 48 kHz).
  * Voice cloning enrolls a clip torch-free (speaker_encoder + codec_encoder +
    reference_encoder ONNX, fetched on first use): ``infer(ref_audio=...)``,
    ``encode_reference`` and ``add_voice`` work like v3 Turbo.
  * Lower quality on English and code-switched text (trained on ~4,000 h of
    Vietnamese; English exposure is incidental).
  * Non-autoregressive: no frame-level streaming — ``infer_stream`` yields one
    finished chunk at a time.

Quality/speed knobs: ``steps`` (Euler steps, 16 default; 8 is ~2x faster and still
intelligible — pair it with ``sway=-1``) and ``cfg`` (classifier-free guidance, 3.0
default; 0 halves the compute but hurts intelligibility).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    plan_v3_chunks,
)
from vieneu_utils.core_utils import (
    join_audio_chunks, gaps_to_silence, pause_pad_samples, trim_and_fade as _trim_and_fade,
)

logger = logging.getLogger("Vieneu.V3Nano")

_NANO_REPO = "pnnbao-ump/VieNeu-TTS-v3-Nano"
_GRAPH_FILES = ["text_encoder.onnx", "duration_predictor.onnx", "vector_estimator.onnx",
                "codec_decoder.onnx", "config.json", "constants.npz"]
# Voice cloning (fetched lazily on the first clone): x-vector + codec encoder +
# reference/style encoder. The denoiser is optional (falls back to v3 Turbo's copy).
_CLONE_FILES = ["speaker_encoder.onnx", "codec_encoder.onnx", "reference_encoder.onnx"]
_DENOISER_FILE = "denoiser.onnx"
_TURBO_REPO_FOR_DENOISER = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
_REF_SECONDS = 5.0               # style tokens come from the first 5 s of the clip
_MAX_REF_SECONDS = 30.0          # x-vector sees at most this much audio
_MAX_CHUNK_SECONDS = 15.0        # training clips were <= 15 s; longer targets drift
_MIN_FRAMES = 2


class _Dev:
    """So callers' ``engine.device.type == "cuda"`` checks work (always CPU here)."""
    type = "cpu"


class OnnxV3NanoEngine:
    """The four ONNX graphs + the sampling loop. Torch-free.

    Public surface mirrors the v3 Turbo engines closely enough for the shared web
    UI code: ``infer(phonemes=..., speaker_emb=..., ref_codes=...)`` where, for
    Nano, ``ref_codes`` is the ``[50, 256]`` float style-token array of the voice.
    """
    SAMPLE_RATE = 24_000

    def __init__(self, repo: str = _NANO_REPO, local_dir: Optional[str] = None,
                 threads: int = 0, hf_token: Optional[str] = None):
        import onnxruntime as ort

        self._lock = threading.RLock()
        self.device = _Dev()
        self.repo, self.local_dir, self.hf_token = repo, local_dir, hf_token
        d = Path(local_dir) if local_dir else self._fetch(repo, hf_token)
        self.cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        c = np.load(d / "constants.npz")
        self.null_spk = c["null_spk"].astype(np.float32)[None]           # [1,192]
        self.null_style = c["null_style"].astype(np.float32)[None]       # [1,S,C]
        # Latent normalisation for the reference/style path (cloning only).
        self.lat_mean = c["latent_mean"].astype(np.float32)[None, :, None] if "latent_mean" in c.files else None
        self.lat_std = c["latent_std"].astype(np.float32)[None, :, None] if "latent_std" in c.files else None
        self.lat_scale = float(c["latent_scale"]) if "latent_scale" in c.files else float(self.cfg.get("latent_scale", 1.0))
        self.group = int(self.cfg.get("group", 6))
        self.ref_max_frames = int(self.cfg.get("ref_max_frames", 140))
        self.vocab: Dict[str, int] = self.cfg["vocab"]
        self.bos, self.eos, self.pad = int(self.cfg["bos_id"]), int(self.cfg["eos_id"]), int(self.cfg["pad_id"])
        self.fps = float(self.cfg.get("flow_fps", 24000 / 256 / 6))
        # <|emotion_k|> -> single vocab char (①②③...), exactly as in training
        self.emotion_map: Dict[str, str] = dict(self.cfg.get("emotion_tags", {}))
        self._oov_warned: set = set()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        intra = int(threads) if threads and threads > 0 else min(max((os.cpu_count() or 8) // 2, 1), 8)
        so.intra_op_num_threads = intra
        self.ort_intra_op_threads = intra
        so.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        self.s_text = ort.InferenceSession(str(d / "text_encoder.onnx"), so, providers=prov)
        self.s_dur = ort.InferenceSession(str(d / "duration_predictor.onnx"), so, providers=prov)
        self.s_ve = ort.InferenceSession(str(d / "vector_estimator.onnx"), so, providers=prov)
        self.s_dec = ort.InferenceSession(str(d / "codec_decoder.onnx"), so, providers=prov)
        # the unconditional text context is voice-independent: build it once
        null_ids = np.array([[self.bos, self.eos]], dtype=np.int64)
        self._null_ctx = self.s_text.run(None, {"ids": null_ids, "style": self.null_style})[0]
        self._null_mask = null_ids != self.pad
        self._so, self._prov = so, prov
        # cloning graphs: created on first use (see _ensure_clone_sessions)
        self.s_codec_enc = self.s_ref = None
        self.speaker_encoder = None
        self.denoiser = None
        self._denoiser_tried = False

    @staticmethod
    def _fetch(repo: str, token: Optional[str]) -> Path:
        from huggingface_hub import hf_hub_download
        last = None
        for fn in _GRAPH_FILES:
            last = hf_hub_download(repo, fn, repo_type="model", token=token)
        return Path(last).parent

    # ── voice cloning: reference clip -> (x-vector, style tokens) ─────────────
    def _clone_file(self, fn: str, repo: Optional[str] = None) -> str:
        """Resolve a cloning artifact from ``local_dir`` (if given) or the HF repo."""
        if self.local_dir and repo is None:
            p = Path(self.local_dir) / fn
            if p.is_file():
                return str(p)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo or self.repo, fn, repo_type="model", token=self.hf_token)

    def _ensure_clone_sessions(self) -> None:
        if self.s_codec_enc is not None:
            return
        import onnxruntime as ort
        from ._v3_turbo_engine.speaker import OnnxSpeakerEncoder
        if self.lat_mean is None or self.lat_std is None:
            raise RuntimeError("Nano bundle lacks latent_mean/latent_std in constants.npz — cannot enroll references.")
        self.speaker_encoder = OnnxSpeakerEncoder(self._clone_file("speaker_encoder.onnx"),
                                                  max_seconds=_MAX_REF_SECONDS)
        self.s_codec_enc = ort.InferenceSession(self._clone_file("codec_encoder.onnx"), self._so, providers=self._prov)
        self.s_ref = ort.InferenceSession(self._clone_file("reference_encoder.onnx"), self._so, providers=self._prov)

    def _get_denoiser(self):
        """resemble-enhance denoiser (ONNX, torch-free); None if unavailable."""
        if not self._denoiser_tried:
            self._denoiser_tried = True
            try:
                from ._v3_turbo_engine.onnx_denoiser import OnnxDenoiser
                try:
                    path = self._clone_file(_DENOISER_FILE)
                except Exception:
                    path = self._clone_file(_DENOISER_FILE, repo=_TURBO_REPO_FOR_DENOISER)
                self.denoiser = OnnxDenoiser(path)
            except Exception as e:                       # cloning still works, just un-denoised
                logger.warning("Nano denoiser unavailable (%s) — enrolling without denoise.", e)
                self.denoiser = None
        return self.denoiser

    @staticmethod
    def _load_mono(ref_audio, sr: Optional[int]) -> Tuple[np.ndarray, int]:
        if isinstance(ref_audio, (str, bytes)) or hasattr(ref_audio, "__fspath__"):
            import soundfile as sf
            wav, sr = sf.read(str(ref_audio), dtype="float32", always_2d=True)   # (n, ch)
            wav = wav.mean(axis=1)
        else:
            wav = np.asarray(ref_audio, dtype=np.float32)
            if sr is None:
                raise ValueError("Pass `sr` when giving a waveform array.")
            if wav.ndim == 2:
                wav = wav.mean(axis=0) if wav.shape[0] <= wav.shape[1] else wav.mean(axis=1)
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)

    def _group_latent(self, z: np.ndarray) -> np.ndarray:
        """[1, C, T] -> [1, C*group, ceil(T/group)] (pads T), as in training."""
        g = self.group
        B, C, T = z.shape
        if T % g:
            z = np.pad(z, ((0, 0), (0, 0), (0, g - T % g)))
            T = z.shape[-1]
        return z.reshape(B, C, T // g, g).transpose(0, 1, 3, 2).reshape(B, C * g, T // g)

    def prepare_reference(self, ref_audio, *, sr: Optional[int] = None, denoise: bool = True,
                          ref_seconds: float = _REF_SECONDS, max_seconds: float = _MAX_REF_SECONDS,
                          use_ref_codes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice: ``(speaker_emb [192], style [50, 256])`` — the exact preset layout.

        Recipe (matches the Nano export README / infer.py): x-vector from the 80-mel
        Kaldi fbank of the whole clip (<= ``max_seconds``); style tokens from the codec
        latent of the first ``ref_seconds`` seconds, normalised with the training
        latent mean/std/scale, grouped x``group`` and cropped to ``ref_max_frames``.
        ``use_ref_codes`` is accepted for API parity with v3 Turbo and ignored.
        """
        self._ensure_clone_sessions()
        wav, sr = self._load_mono(ref_audio, sr)
        wav = wav[: int(max_seconds * sr)]
        if denoise:
            den = self._get_denoiser()
            if den is not None:
                wav = np.asarray(den.denoise(wav, sr), dtype=np.float32)
                sr = 44100
        spk = self.speaker_encoder.embed(wav, sr)                             # (192,)
        from ._v3_turbo_engine.speaker.audio_utils import high_quality_resample
        m24 = wav if sr == self.SAMPLE_RATE else high_quality_resample(wav, sr, self.SAMPLE_RATE)
        m24 = np.asarray(m24, dtype=np.float32)[: int(self.SAMPLE_RATE * ref_seconds)]
        inp = self.s_codec_enc.get_inputs()[0].name
        mu = self.s_codec_enc.run(None, {inp: m24[None, None]})[0]           # [1, 24, T]
        z = (mu - self.lat_mean) / self.lat_std * self.lat_scale
        z = self._group_latent(z)[:, :, : min(int(ref_seconds * self.fps), self.ref_max_frames)]
        ref_mask = np.ones((1, z.shape[2]), dtype=bool)
        style = self.s_ref.run(None, {"ref": z.astype(np.float32), "ref_mask": ref_mask})[0][0]   # [S, C]
        return np.asarray(spk, dtype=np.float32), np.asarray(style, dtype=np.float32)


    # ── text ──────────────────────────────────────────────────────────────────
    def encode_phones(self, phones: str) -> np.ndarray:
        """Phone string (may contain ``<|emotion_k|>``) -> ids [1, L]. Unknown characters
        are dropped with a one-time warning instead of failing the whole request."""
        s = phones
        for tag, ch in self.emotion_map.items():
            s = s.replace(tag, ch)
        ids = [self.bos]
        for ch in s:
            i = self.vocab.get(ch)
            if i is None:
                if ch not in self._oov_warned:
                    self._oov_warned.add(ch)
                    logger.warning("v3 Nano: bỏ qua ký tự ngoài vocab %r (U+%04X)", ch, ord(ch))
                continue
            ids.append(i)
        ids.append(self.eos)
        return np.array([ids], dtype=np.int64)

    # ── synthesis ─────────────────────────────────────────────────────────────
    def infer(self, phonemes: str, speaker_emb: np.ndarray, ref_codes: np.ndarray, *,
              steps: int = 16, cfg: float = 3.0, sway: float = 0.0, speed: float = 1.0,
              seed: Optional[int] = None, **_ignored: Any) -> np.ndarray:
        """One chunk of phonemes -> float32 waveform @ 24 kHz.

        ``ref_codes`` is the voice's ``[50, 256]`` style-token array (name kept for
        parity with the Turbo engine so the web UI can call both the same way).
        """
        with self._lock:
            spk = np.asarray(speaker_emb, dtype=np.float32).reshape(1, -1)
            style = np.asarray(ref_codes, dtype=np.float32)
            style = style.reshape(1, *style.shape[-2:])
            ids = self.encode_phones(phonemes)
            mask = ids != self.pad
            ctx = self.s_text.run(None, {"ids": ids, "style": style})[0]
            log_s = float(self.s_dur.run(None, {"ctx": ctx, "ctx_mask": mask, "spk": spk})[0][0])
            secs = min(math.exp(log_s) / max(float(speed), 1e-3), _MAX_CHUNK_SECONDS)
            T = max(_MIN_FRAMES, int(round(secs * self.fps)))
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((1, 144, T)).astype(np.float32)
            n = max(1, int(steps))
            u = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
            tg = u + float(sway) * (np.cos(np.pi / 2 * u) - 1 + u)
            for i in range(n):
                t = np.array([tg[i]], dtype=np.float32)
                v = self.s_ve.run(None, {"x": x, "t": t, "ctx": ctx, "ctx_mask": mask,
                                         "spk": spk, "style": style})[0]
                if cfg > 0:
                    vu = self.s_ve.run(None, {"x": x, "t": t, "ctx": self._null_ctx, "ctx_mask": self._null_mask,
                                              "spk": self.null_spk, "style": self.null_style})[0]
                    v = vu + float(cfg) * (v - vu)
                x = x + np.float32(tg[i + 1] - tg[i]) * v
            wav = self.s_dec.run(None, {"x": x})[0][0, 0]
            return _trim_and_fade(np.clip(wav, -1.0, 1.0).astype(np.float32), self.SAMPLE_RATE)


# ``_trim_and_fade`` giờ là ``vieneu_utils.core_utils.trim_and_fade`` (dùng chung với
# join_audio_chunks của v3 Turbo); tên cũ giữ lại cho test/code ngoài.


class V3NanoVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Nano (ONNX, CPU). Preset voices only."""

    def __init__(
        self,
        backbone_repo: str = _NANO_REPO,
        onnx_dir: Optional[str] = None,
        threads: int = 0,          # ORT intra-op threads; 0 = ~physical cores, cap 8
        steps: int = 16,           # default Euler steps for every call (8 = faster, slightly rougher)
        cfg: float = 3.0,          # default classifier-free guidance
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = OnnxV3NanoEngine.SAMPLE_RATE
        logger.info(f"⏳ Loading VieNeu-TTS v3 Nano (ONNX/CPU) from: {onnx_dir or backbone_repo} ...")
        self.engine = OnnxV3NanoEngine(repo=backbone_repo, local_dir=onnx_dir, threads=threads, hf_token=hf_token)
        self.backend = "onnx"
        self.default_steps = int(steps)
        self.default_cfg = float(cfg)
        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        self._default_voice: Optional[str] = None
        self._load_nano_voices()
        logger.info("✅ VieNeu-TTS v3 Nano ready (24 kHz, CPU, %d preset voices)", len(self._preset_voices))

    # ── voices ────────────────────────────────────────────────────────────────
    def _load_nano_voices(self) -> None:
        """Presets from assets/voices_v3_nano.json: 192-d x-vector + [50,256] style tokens.

        The dict also exposes the style array under ``codes`` so code written for
        the v3 Turbo preset layout (``speaker_emb`` + ``codes``) works unchanged.
        """
        path = Path(__file__).parent / "assets" / "voices_v3_nano.json"
        if not path.exists():
            logger.warning("voices_v3_nano.json not found — no preset voices available.")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, v in data.get("presets", {}).items():
            style = np.asarray(v["style"], dtype=np.float32)
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": style,
                "codes": style,                              # alias (see docstring)
                "speaker_emb": np.asarray(v["speaker_emb"], dtype=np.float32),
                "podcast": True,
            }
        self._default_voice = data.get("default_voice") or next(iter(self._preset_voices), None)
        logger.info(f"📢 Loaded {len(self._preset_voices)} preset voices (default: {self._default_voice})")

    def list_preset_voices(self) -> List[tuple]:
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in self._preset_voices.items()]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice from a clip -> ``(speaker_emb [192], style [50, 256])``.

        Same pre-clean as v3 Turbo (edge-silence trim, optional denoise). The pair can
        be passed back as ``voice={"speaker_emb": ..., "style": ...}`` or registered
        with :meth:`add_voice`.
        """
        import os
        from .v3turbo import V3TurboVieNeuTTS
        clean_ref = V3TurboVieNeuTTS._preclean_reference_audio(ref_audio)
        try:
            return self.engine.prepare_reference(str(clean_ref), denoise=denoise)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass

    def add_voice(self, name: str, ref_audio: Union[str, Path], *, denoise: bool = True,
                  description: str = "", gender: str = "", save: bool = False, **kwargs: Any) -> str:
        """Register a cloned voice under ``name`` for ``infer(..., voice=name)``.

        ``save=True`` persists it to ``assets/voices_v3_nano.json`` (see :meth:`save_voices`).
        """
        if not name or not str(name).strip():
            raise ValueError("Tên giọng không được để trống.")
        spk, style = self.encode_reference(ref_audio, denoise=denoise)
        self._preset_voices[name] = {
            "description": description, "gender": gender,
            "style": style, "codes": style, "speaker_emb": spk, "podcast": True,
        }
        if not self._default_voice:
            self._default_voice = name
        if save:
            self.save_voices()
        logger.info(f"➕ Added voice '{name}'.")
        return name

    def remove_voice(self, name: str, save: bool = False) -> None:
        self._preset_voices.pop(name, None)
        if self._default_voice == name:
            self._default_voice = next(iter(self._preset_voices), None)
        if save:
            self.save_voices()

    def save_voices(self, path: Optional[Union[str, Path]] = None) -> str:
        """Persist the registered voices (x-vector + style tokens) as a Nano preset file."""
        path = Path(path) if path else (Path(__file__).parent / "assets" / "voices_v3_nano.json")
        presets = {}
        for name, v in self._preset_voices.items():
            presets[name] = {
                "description": v.get("description", ""), "gender": v.get("gender", ""),
                "ref_seconds": _REF_SECONDS,
                "speaker_emb": [round(float(x), 6) for x in np.asarray(v["speaker_emb"]).reshape(-1)],
                "style": [[round(float(x), 5) for x in row] for row in np.asarray(v["style"])],
            }
        data = {"meta": {"note": "VieNeu-TTS v3 Nano preset voices: 192-d x-vector + 50x256 style tokens"},
                "default_voice": self._default_voice, "presets": presets}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def _resolve_voice(self, voice, ref_audio, denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if ref_audio is not None:
            return self.encode_reference(ref_audio, denoise=denoise)
        if isinstance(voice, dict):
            preset = voice
        elif isinstance(voice, str):
            preset = self.get_preset_voice(voice)
        else:
            preset = self.get_preset_voice(None)
        style = preset.get("style", preset.get("codes"))
        if style is None or preset.get("speaker_emb") is None:
            raise ValueError("Voice dict must carry 'speaker_emb' and 'style' (v3 Nano preset layout).")
        return preset["speaker_emb"], style

    # ── public API ────────────────────────────────────────────────────────────
    def _chunk_wavs(self, chunks: List[str], spk, style, steps, cfg, sway, speed, seed) -> List[np.ndarray]:
        out = []
        for i, chunk in enumerate(chunks):
            ph = phonemize_text_with_emotions(chunk)
            out.append(self.engine.infer(ph, spk, style, steps=steps, cfg=cfg, sway=sway, speed=speed,
                                         seed=None if seed is None else int(seed) + i))
        return out

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,          # accepted for API parity with v3 Turbo; ignored
        denoise: bool = True,        # clean the reference clip before enrolling (ref_audio only)
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sway: float = 0.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
        max_chars: int = 140,       # Nano was trained on <= 15 s clips: keep chunks short
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Synthesize ``text`` into one 24 kHz waveform with a preset ``voice``."""
        chunks, gaps, explicit_pauses = plan_v3_chunks(text, max_chars=max_chars)
        spk, st = self._resolve_voice(voice, ref_audio, denoise)
        if not chunks:
            return join_audio_chunks([], self.sample_rate, explicit_pauses_ms=explicit_pauses)
        wavs = self._chunk_wavs(chunks, spk, st, steps or self.default_steps,
                                self.default_cfg if cfg is None else cfg, sway, speed, seed)
        wav = join_audio_chunks(wavs, self.sample_rate, silence_ps=gaps_to_silence(gaps),
                                explicit_pauses_ms=explicit_pauses)
        return self._apply_watermark(wav) if apply_watermark else wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,
        denoise: bool = True,        # clean the reference clip before enrolling (ref_audio only)
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sway: float = 0.0,
        speed: float = 1.0,
        max_chars: int = 140,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        """Chunk-level streaming: yields each finished chunk (no frame-level streaming —
        the flow model generates a whole chunk at once)."""
        chunks, gaps, explicit_pauses = plan_v3_chunks(text, max_chars=max_chars)
        spk, st = self._resolve_voice(voice, ref_audio, denoise)
        sr = self.sample_rate
        if not chunks:
            if explicit_pauses:
                yield np.zeros(explicit_pauses[0] * sr // 1000, dtype=np.float32)
            return
        pauses = gaps_to_silence(gaps)
        prev: Optional[np.ndarray] = None
        for ci, chunk in enumerate(chunks):
            if ci in explicit_pauses:
                yield np.zeros(explicit_pauses[ci] * sr // 1000, dtype=np.float32)
                prev = None
            ph = phonemize_text_with_emotions(chunk)
            wav = self.engine.infer(ph, spk, st, steps=steps or self.default_steps,
                                    cfg=self.default_cfg if cfg is None else cfg, sway=sway, speed=speed)
            if not wav.size:
                continue
            if prev is not None:
                # Cùng khoảng nghỉ theo ranh giới như join_audio_chunks (engine đã trim mép).
                pad = pause_pad_samples(prev, wav, sr, pauses[ci - 1])
                if pad > 0:
                    yield np.zeros(pad, dtype=np.float32)
            prev = wav
            yield self._apply_watermark(wav) if apply_watermark else wav
        if len(chunks) in explicit_pauses:
            yield np.zeros(explicit_pauses[len(chunks)] * sr // 1000, dtype=np.float32)

    def infer_batch(
        self,
        texts: List[str],
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Sequential on CPU (one waveform per text, same order). A ``ref_audio`` is
        enrolled once and reused for every text."""
        if ref_audio is not None:
            spk, st = self.encode_reference(ref_audio, denoise=kwargs.pop("denoise", True))
            voice, ref_audio = {"speaker_emb": spk, "style": st}, None
        return [self.infer(t, ref_audio=ref_audio, voice=voice, apply_watermark=apply_watermark, **kwargs)
                for t in texts]

    def close(self) -> None:
        self.engine = None
