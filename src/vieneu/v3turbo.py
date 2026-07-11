"""
VieNeu-TTS v3 Turbo backend (PyTorch).
======================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    wav = tts.infer("Xin chào", ref_audio="ref.wav")   # clone a voice
    wav = tts.infer("Xin chào", voice="Xuân Vĩnh", style="doc_truyen")  # preset voice
    tts.save(wav, "out.wav")
"""
import logging
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence

logger = logging.getLogger("Vieneu.V3Turbo")


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo (PyTorch)."""

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        model_subfolder: str = "update",
        moss_tokenizer: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device: str = "auto",
        dtype: str = "auto",
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch" to force
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        precision: str = "int8",   # ONNX/CPU backbone: "int8" (mặc định, nhanh ~3x/frame, nhỏ 4x) | "fp32" (chất-lượng-tối-đa)
        onnx_subfolder: Optional[str] = None,   # override thủ công subfolder; None → suy từ `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = mặc định engine (~nhân vật lý, cap 8). Đặt số cụ thể để tinh chỉnh.
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = 48_000

        # `precision` chỉ áp cho đường ONNX/CPU (chọn subfolder graph int8 vs fp32).
        # Đường PyTorch/GPU dùng torch fp32/bf16, không liên quan.
        if onnx_subfolder is None:
            onnx_subfolder = {"int8": "onnx_int8", "fp32": "onnx_update"}.get(str(precision).lower(), "onnx_int8")

        if device in (None, "auto"):
            try:
                import torch
                dev_type = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev_type = "cpu"
        else:
            dev_type = "cuda" if "cuda" in str(device).lower() else str(device).lower()
        use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")

        if use_onnx:
            # Torch-free CPU engine. Reads its ONNX graphs from `onnx_subfolder` in the
            # model repo (uploaded separately).
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                onnx_subfolder=onnx_subfolder,
                threads=threads,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
            )
            self.backend = "pytorch"
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")

        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        self._default_voice: Optional[str] = None
        self._load_v3_voices()

    # ── Preset voices (speaker embedding + reference codes) ─────────────────────
    def _load_v3_voices(self) -> None:
        """Load the built-in voices from assets/voices_v3_turbo.json.

        Each preset carries a 192-d ``speaker_emb`` and pre-encoded ``codes``.
        """
        import json
        path = Path(__file__).parent / "assets" / "voices_v3_turbo.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, v in data.get("presets", {}).items():
            emb = v.get("speaker_emb")
            codes = v.get("codes")
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "speaker_emb": np.asarray(emb, dtype=np.float32) if emb is not None else None,
                "codes": np.asarray(codes, dtype=np.int64) if codes is not None else None,
            }
        self._default_voice = data.get("default_voice")
        logger.info(f"📢 Loaded {len(self._preset_voices)} preset voices (default: {self._default_voice})")

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for the built-in voices."""
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in self._preset_voices.items()]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice from a wav → ``(speaker_emb, ref_codes)``."""
        return self.engine.prepare_reference(str(ref_audio), denoise=denoise, use_ref_codes=True)

    def denoise(self, ref_audio: Union[str, Path], out_path: Optional[Union[str, Path]] = None,
                max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Clean up a reference clip and return ``(wav, sample_rate)`` at 44.1 kHz.

        Pass ``out_path`` to also save the cleaned wav. ``max_seconds`` optionally
        trims the clip first. Use the result as a nicer reference for cloning.
        """
        import soundfile as sf
        den = getattr(self.engine, "denoiser", None)
        if den is None:
            raise RuntimeError("Denoiser không khả dụng trên backend này.")
        wav, sr = sf.read(str(ref_audio))
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if max_seconds and len(wav) > int(max_seconds * sr):
            wav = wav[: int(max_seconds * sr)]
        clean = den.denoise(wav, sr)          # -> float32 mono @ 44100
        if out_path is not None:
            sf.write(str(out_path), clean, 44100)
        return clean, 44100

    def add_voice(self, name: str, ref_audio: Union[str, Path], *, denoise: bool = True,
                  use_ref_codes: bool = True, description: str = "", gender: str = "",
                  style: str = "tu_nhien", save: bool = False) -> str:
        """Register a custom voice under ``name`` for use as ``infer(..., voice=name)``.

        ``ref_audio`` is enrolled once (denoised + trimmed, then speaker embedding +
        reference codes are extracted). Pass ``denoise=False`` if the clip is already
        clean. Set ``save=True`` to persist it to the voices file for later sessions.
        """
        if not name or not str(name).strip():
            raise ValueError("Tên giọng không được để trống.")
        speaker_emb, ref_codes = self.engine.prepare_reference(
            str(ref_audio), denoise=denoise, use_ref_codes=use_ref_codes)
        self._preset_voices[name] = {
            "description": description,
            "gender": gender,
            "style": style,
            "speaker_emb": np.asarray(speaker_emb, dtype=np.float32),
            "codes": None if ref_codes is None else np.asarray(ref_codes, dtype=np.int64),
        }
        if not self._default_voice:
            self._default_voice = name
        if save:
            self.save_voices()
        logger.info(f"➕ Added voice '{name}'.")
        return name

    def remove_voice(self, name: str, save: bool = False) -> None:
        """Remove a registered voice by name."""
        self._preset_voices.pop(name, None)
        if self._default_voice == name:
            self._default_voice = next(iter(self._preset_voices), None)
        if save:
            self.save_voices()

    def save_voices(self, path: Optional[Union[str, Path]] = None) -> str:
        """Persist the current voices (speaker embedding + codes) to a JSON file."""
        import json
        path = Path(path) if path else (Path(__file__).parent / "assets" / "voices_v3_turbo.json")
        presets = {}
        for n, v in self._preset_voices.items():
            emb = v.get("speaker_emb")
            codes = v.get("codes")
            presets[n] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
                "codes": np.asarray(codes, dtype=int).tolist() if codes is not None else None,
            }
        data = {"meta": {"note": "v3 turbo voices: speaker embedding + reference codes"},
                "default_voice": self._default_voice, "presets": presets}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Saved {len(presets)} voices → {path}")
        return str(path)

    def _resolve_ref(self, voice, ref_audio, denoise, use_ref_codes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve the requested voice to ``(speaker_emb, ref_codes)``.

        Precedence: cloned ``ref_audio`` → preset ``voice`` (name or dict) → default preset.
        """
        if ref_audio is not None:
            return self.engine.prepare_reference(str(ref_audio), denoise=denoise, use_ref_codes=use_ref_codes)
        preset = None
        if isinstance(voice, str):
            preset = self._preset_voices.get(voice)
            if preset is None:
                raise ValueError(f"Voice '{voice}' not found. Available: {list(self._preset_voices)}")
        elif isinstance(voice, dict):
            preset = voice
        elif self._default_voice:
            preset = self._preset_voices[self._default_voice]
        if preset is None:
            raise ValueError("Provide a preset `voice` name or a `ref_audio` to clone.")
        codes = preset.get("codes") if use_ref_codes else None
        return preset["speaker_emb"], codes

    # ── Public API ───────────────────────────────────────────────────────────
    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: str = "tu_nhien",
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        max_chars: int = 256,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)

        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        all_wavs: List[np.ndarray] = []
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            wav = self.engine.infer(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                style=style, use_ref_codes=use_ref_codes,
                temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            )
            all_wavs.append(wav)

        # Im lặng theo loại ranh giới: ngắt đoạn > hết câu > ngắt trong câu.
        final_wav = join_audio_chunks(
            all_wavs, self.sample_rate, silence_ps=gaps_to_silence(gaps)
        )
        return self._apply_watermark(final_wav) if apply_watermark else final_wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: str = "tu_nhien",
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        chunks = normalize_to_chunks_v3(text, max_chars=max_chars)
        # Prefer the engine's native frame-level streaming (low first-audio latency);
        # both the PyTorch and ONNX engines expose infer_stream. Fall back to one
        # full infer per chunk if not available.
        stream_fn = getattr(self.engine, "infer_stream", None)
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            if stream_fn is not None:
                for sub in stream_fn(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    style=style, use_ref_codes=use_ref_codes,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
                ):
                    if sub is None or len(sub) == 0:
                        continue
                    yield self._apply_watermark(sub) if apply_watermark else sub
            else:
                wav = self.engine.infer(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    style=style, use_ref_codes=use_ref_codes,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
                )
                yield self._apply_watermark(wav) if apply_watermark else wav

    def infer_batch(self, texts: List[str], apply_watermark: bool = True, **kwargs: Any) -> List[np.ndarray]:
        return [self.infer(t, apply_watermark=apply_watermark, **kwargs) for t in texts]

    def close(self) -> None:
        self.engine = None
