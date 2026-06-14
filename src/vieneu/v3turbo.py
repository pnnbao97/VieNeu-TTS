"""
VieNeu-TTS v3 Turbo backend (PyTorch).
======================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    wav = tts.infer("Xin chào", ref_audio="ref.wav")
    tts.save(wav, "out.wav")
"""
import logging
from pathlib import Path
from typing import Any, Generator, List, Optional, Union

import numpy as np

from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import phonemize_text_with_emotions
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks
try:
    from .vietnamese_processor.pipeline import process_vietnamese_text
except ImportError:
    process_vietnamese_text = None

logger = logging.getLogger("Vieneu.V3Turbo")


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo (PyTorch)"""

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        moss_tokenizer: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device: str = "auto",
        dtype: str = "auto",
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch" to force
        onnx_repo: Optional[str] = None,  # HF repo holding onnx/ (default = backbone_repo)
        onnx_dir: Optional[str] = None,   # local dir with the 4 onnx files (skips HF download)
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ):
        # KHÔNG truyền codec_repo → bỏ qua NeuCodec của base; v3 có codec MOSS riêng.
        super().__init__()
        self.sample_rate = 48_000  # v3 = 48 kHz (ghi đè 24 kHz của base)

        # Pick engine by device: CPU → torch-free ONNX engine (fast + light), GPU →
        # PyTorch. Device is resolved WITHOUT hard-requiring torch, so a torch-free
        # CPU install (onnxruntime only) still works.
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
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU, torch-free) from: {backbone_repo} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                hf_token=hf_token,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
                hf_token=hf_token,
            )
            self.backend = "pytorch"
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")

        # Built-in default voices. The emotion checkpoint identifies each default
        # speaker by a reserved token (``reserved_id``, ids 13..42) plus that
        # speaker's fixed MOSS ref codes — this is the "emotion" path. Users may
        # instead clone any voice by passing `ref_audio` (or `ref_codes`), which
        # falls back to the original emotion-tag path (no reserved token).
        self._preset_voices = {}
        self._default_voice = None
        self._load_v3_voices()

    def _load_v3_voices(self) -> None:
        """Load the built-in default voices from assets/voices_v3_turbo.json.

        Each preset carries a ``reserved_id`` (the speaker token) and pre-encoded
        ``codes`` (the speaker's fixed reference frames).
        """
        import json
        path = Path(__file__).parent / "assets" / "voices_v3_turbo.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, v in data.get("presets", {}).items():
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "reserved_id": int(v["reserved_id"]) if v.get("reserved_id") is not None else None,
                "codes": np.asarray(v["codes"], dtype=np.int64),  # (T, n_vq)
            }
        self._default_voice = data.get("default_voice")
        logger.info(f"📢 Loaded {len(self._preset_voices)} preset voices "
                    f"(default: {self._default_voice})")

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for the built-in default voices.

        Format matches the base API: label = description (falls back to the name),
        value = the voice name used by :meth:`get_preset_voice`.
        """
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in self._preset_voices.items()]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        """Return ``{"codes", "reserved_id", "text"}`` for a built-in default voice.

        ``reserved_id`` is the speaker token used by the emotion path; ``codes`` are
        that speaker's fixed reference frames. ``text`` is always empty (v3 does not
        use a reference transcript).
        """
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        v = self._preset_voices[name]
        return {"codes": v["codes"], "reserved_id": v.get("reserved_id"), "text": ""}

    # ── Reference voice (clone from ref audio / codes, or a preset by name) ──
    def encode_reference(self, ref_audio: Union[str, Path]) -> np.ndarray:
        """Encode a reference wav into MOSS ref codes ``(T, n_vq)``."""
        return self.engine._encode_ref(str(ref_audio))

    def _preset_codes(self, name: str) -> np.ndarray:
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]["codes"]

    def _resolve_v3_ref(self, voice, ref_audio, ref_codes):
        """Resolve the voice to ``(ref_codes, voice_token_id)``.

        ``voice_token_id`` is the speaker reserved token for a built-in default
        voice (the "emotion" path); it is ``None`` when cloning, so the engine
        falls back to the emotion-tag path with free reference frames.

        Precedence: explicit ``ref_codes`` -> cloned ``ref_audio`` (both = clone,
        token None) -> preset ``voice`` (name string or dict, token = reserved_id)
        -> default preset.
        """
        if ref_codes is not None:
            return np.asarray(ref_codes), None
        if ref_audio is not None:
            return self.encode_reference(ref_audio), None
        if isinstance(voice, str):
            v = self._preset_voices.get(voice)
            if v is None:
                raise ValueError(f"Voice '{voice}' not found. Available: {list(self._preset_voices)}")
            return v["codes"], v.get("reserved_id")
        if isinstance(voice, dict) and voice.get("codes") is not None:
            tok = voice.get("reserved_id")
            return np.asarray(voice["codes"]), (int(tok) if tok is not None else None)
        if self._default_voice:
            v = self._preset_voices[self._default_voice]
            return v["codes"], v.get("reserved_id")
        raise ValueError("Provide a preset `voice` name, `ref_audio`, or `ref_codes`.")

    # ── Public API ───────────────────────────────────────────────────────────
    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_codes: Optional[np.ndarray] = None,
        ref_text: Optional[str] = None,  # noqa: ARG002 (v3 không dùng ref transcript)
        voice: Optional[Union[str, dict]] = None,
        emotion: str = "natural",
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
        ref_codes, voice_token_id = self._resolve_v3_ref(voice, ref_audio, ref_codes)

        # Normalize text via vietnamese_processor
        if process_vietnamese_text is not None:
            try:
                text = process_vietnamese_text(text)
            except Exception as e:
                logger.warning(f"Failed to normalize text via vietnamese_processor: {e}")

        # Chunking kiểu v1/v2 (standard): cắt TEXT theo max_chars, ghép có silence/crossfade.
        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        all_wavs: List[np.ndarray] = []
        for chunk in chunks:
            phonemes = phonemize_text_with_emotions(chunk)  # SEAPipeline + inline cues
            wav = self.engine.infer(
                text="", phonemes=phonemes, ref_codes=ref_codes,
                emotion=emotion, voice_token_id=voice_token_id,
                temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            )
            all_wavs.append(wav)

        final_wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
        return self._apply_watermark(final_wav) if apply_watermark else final_wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_codes: Optional[np.ndarray] = None,
        voice: Optional[Union[str, dict]] = None,
        emotion: str = "natural",
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        ref_codes, voice_token_id = self._resolve_v3_ref(voice, ref_audio, ref_codes)

        # Normalize text via vietnamese_processor
        if process_vietnamese_text is not None:
            try:
                text = process_vietnamese_text(text)
            except Exception as e:
                logger.warning(f"Failed to normalize text via vietnamese_processor: {e}")

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            phonemes = phonemize_text_with_emotions(chunk)
            wav = self.engine.infer(
                text="", phonemes=phonemes, ref_codes=ref_codes,
                emotion=emotion, voice_token_id=voice_token_id,
                temperature=temperature, top_k=top_k, top_p=top_p,
                max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            )
            yield self._apply_watermark(wav) if apply_watermark else wav

    def infer_batch(
        self,
        texts: List[str],
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        return [self.infer(t, apply_watermark=apply_watermark, **kwargs) for t in texts]

    def close(self) -> None:
        self.engine = None
