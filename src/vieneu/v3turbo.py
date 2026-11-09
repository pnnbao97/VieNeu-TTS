"""
VieNeu-TTS v3 Turbo backend (PyTorch).
======================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    wav = tts.infer("Xin chào", ref_audio="ref.wav")   # clone a voice
    wav = tts.infer("Xin chào", voice="Xuân Vĩnh")     # preset voice
    tts.save(wav, "out.wav")

``style`` is DEPRECATED on v3 Turbo: the speaking style is already implied by the
reference voice (speaker embedding + reference codes), so generation is always the
natural style. The argument is still accepted everywhere for backward compatibility,
but it is ignored.
"""
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from ._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import (
    join_audio_chunks,
    gaps_to_silence,
    max_expected_frames,
    pause_pad_samples,
    BABBLE_MAX_RETRIES,
    strip_encoder_pad_frame,
)


def _cap_frames(sampling: dict, cap: int) -> dict:
    """Bản sao ``sampling`` với ``max_new_frames`` chặn trên bởi ``cap``.

    Chặn phần "nói thêm" khi stop token bắn trượt trên chunk ngắn (xem
    :func:`vieneu_utils.core_utils.max_expected_frames`); chunk dài không bị
    ảnh hưởng vì cap luôn vượt ``max_new_frames`` mặc định.
    """
    out = dict(sampling)
    out["max_new_frames"] = min(out.get("max_new_frames", 300), cap)
    return out

logger = logging.getLogger("Vieneu.V3Turbo")


def _featured_rank(v: dict) -> Optional[int]:
    """``featured`` from a voices JSON entry as a positive int, else ``None``."""
    try:
        r = int(v.get("featured"))
    except (TypeError, ValueError):
        return None
    return r if r > 0 else None


def sorted_voices(presets: Dict[str, dict]) -> List[Tuple[str, dict]]:
    """``(name, entry)`` pairs: editors' picks first (by ``featured`` rank), then the
    rest in insertion order."""
    return sorted(presets.items(),
                  key=lambda kv: (kv[1].get("featured") is None, kv[1].get("featured") or 0))


def voice_label(name: str, v: dict) -> str:
    """Dropdown / CLI label: ``⭐ Name — description`` for editors' picks."""
    label = f"{name} — {v['description']}" if v.get("description") else name
    return f"⭐ {label}" if v.get("featured") is not None else label


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo (PyTorch)."""

    @staticmethod
    def _preclean_reference_audio(
        ref_audio: Union[str, Path],
        *,
        top_db: int = 30,
        out_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Trim silence at the edges and save a cleaned reference clip.
        """
        import os
        import tempfile
        import soundfile as sf

        src = Path(ref_audio)
        if not src.exists():
            return str(src)

        wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)

        try:
            import librosa
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        except Exception:
            nonzero = np.flatnonzero(np.abs(wav) > 1e-5)
            if nonzero.size == 0:
                wav_trimmed = wav
            else:
                start = max(int(nonzero[0]) - int(0.01 * sr), 0)
                end = min(int(nonzero[-1]) + int(0.01 * sr), len(wav))
                wav_trimmed = wav[start:end]

        if out_path is None:
            try:
                secure_dir = Path.home() / ".cache" / "vieneu" / "tmp"
            except Exception:
                secure_dir = Path("./outputs/tmp")

            secure_dir.mkdir(parents=True, exist_ok=True)
            try:
                secure_dir.chmod(0o700)
            except Exception:
                pass

            fd, out_path = tempfile.mkstemp(
                prefix="temp_clone_optimized_",
                suffix=".wav",
                dir=str(secure_dir)
            )
            os.close(fd)
            out_path = Path(out_path)

        sf.write(str(out_path), wav_trimmed, sr)
        return str(out_path)

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
        precision: str = "fp32",   # ONNX/CPU backbone: "fp32" (mặc định, chất-lượng-tối-đa) | "int8" (nhanh ~3x/frame, nhỏ 4x; cần CPU hỗ trợ VNNI để không bị méo)
        onnx_subfolder: Optional[str] = None,   # override thủ công subfolder; None → suy từ `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = mặc định engine (~nhân vật lý, cap 8). Đặt số cụ thể để tinh chỉnh.
        max_batch_size: int = 32,   # GPU/PyTorch: trần số chunk gộp vào một forward (static batching). Batch thực = min(số_chunk, max_batch_size). Bỏ qua trên CPU/ONNX.
        max_streams: int = 16,   # GPU/PyTorch: số luồng `infer_stream` phục vụ đồng thời (continuous batching, một CUDA graph). Đo trên RTX 3060: 8 luồng chunk đầu ~135 ms, 16 ~200 ms, 32 ~260 ms. Bỏ qua trên CPU/ONNX.
        babble_retries: int = BABBLE_MAX_RETRIES,   # chunk <= 3 tiếng mà "nói thêm" (nhiều cụm âm hơn số tiếng) thì sinh lại tối đa N lần; 0 = tắt
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = 48_000
        self.babble_retries = max(0, int(babble_retries))

        # `precision` chỉ áp cho đường ONNX/CPU (chọn subfolder graph int8 vs fp32).
        # Đường PyTorch/GPU dùng torch fp32/bf16, không liên quan.
        if onnx_subfolder is None:
            onnx_subfolder = {"int8": "onnx_int8", "fp32": "onnx_update"}.get(str(precision).lower(), "onnx_update")

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
        self.engine.babble_retries = self.babble_retries   # guard chạy ở tầng engine
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")

        # Style is deprecated on v3 Turbo: it is implied by the reference (speaker
        # embedding + ref codes), so every generation uses the natural style. Kept
        # only as voice metadata / for backward-compatible call signatures.
        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        # Old names that still resolve to a (renamed) preset, e.g. "Minh Quân Pro" →
        # "Hải Đăng". Filled from the voices JSON ("aliases" per entry) so
        # existing API clients and saved scripts keep working after a rename.
        self._voice_aliases: dict = {}
        self._default_voice: Optional[str] = None
        # Enrolled references, keyed by clip CONTENT (blake2b of the file bytes) +
        # enrol flags. Enrolling = denoise + x-vector + codec encode ≈ 2.7 s at
        # 5-10 cores on a 6-core CPU (measured 2026-09-15), and a document's
        # chunks / repeated requests reuse the same clip. Content-keyed so a
        # fresh temp path per upload still hits.
        from collections import OrderedDict
        self._ref_cache: "OrderedDict[tuple, Tuple[np.ndarray, Optional[np.ndarray]]]" = OrderedDict()
        self.backbone_repo = backbone_repo
        self._load_v3_voices()
        self._load_repo_voices(backbone_repo)

        # Static-batching (GPU/PyTorch). Dựng lười ở lần batch đầu; None trên CPU/ONNX.
        self.max_batch_size = max(1, int(max_batch_size))
        self._batch_engine = None
        # One batch engine per model: a second one would capture, and keep, a
        # second set of CUDA graphs.
        self._batch_lock = threading.Lock()
        # Streaming scheduler (GPU/PyTorch): một worker thread + một CUDA graph
        # phục vụ mọi `infer_stream` đang chạy. Dựng lười ở lần stream đầu.
        self.max_streams = max(1, int(max_streams))
        self._stream_sched = None
        self._stream_lock = threading.Lock()

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
                "featured": _featured_rank(v),
                "aliases": list(v.get("aliases") or []),
                "speaker_emb": np.asarray(emb, dtype=np.float32) if emb is not None else None,
                "codes": strip_encoder_pad_frame(np.asarray(codes, dtype=np.int64)) if codes is not None else None,
            }
            self._register_aliases(name, v.get("aliases"))
        self._default_voice = data.get("default_voice")
        logger.info(f"📢 Loaded {len(self._preset_voices)} preset voices (default: {self._default_voice})")

    def _load_repo_voices(self, backbone_repo: Optional[str]) -> None:
        """Voices shipped WITH a model (fine-tunes): ``voices_v3_turbo.json`` at the root of
        the model folder / Hub repo, same layout as the built-in file. They are added on
        top of the built-ins (same name = override) and the file's ``default_voice`` wins.
        Missing file = nothing happens."""
        if not backbone_repo:
            return
        import json
        path = None
        local = Path(backbone_repo)
        if local.is_dir():
            if (local / "voices_v3_turbo.json").is_file():
                path = local / "voices_v3_turbo.json"
        else:
            try:
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(backbone_repo, "voices_v3_turbo.json"))
            except Exception:
                path = None
        if path is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Ignoring unreadable voices file {path}: {e}")
            return
        n = 0
        for name, v in data.get("presets", {}).items():
            emb, codes = v.get("speaker_emb"), v.get("codes")
            if emb is None:
                continue
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "featured": _featured_rank(v),
                "aliases": list(v.get("aliases") or []),
                "speaker_emb": np.asarray(emb, dtype=np.float32),
                "codes": strip_encoder_pad_frame(np.asarray(codes, dtype=np.int64)) if codes is not None else None,
            }
            self._register_aliases(name, v.get("aliases"))
            n += 1
        if data.get("default_voice") in self._preset_voices:
            self._default_voice = data["default_voice"]
        if n:
            logger.info("📢 Loaded %d extra voice(s) shipped with the model.", n)

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for the built-in voices.

        Editors' picks (``featured`` 1..N in the voices JSON) come first in that
        order with a ⭐ prefix, then the remaining voices in file order.
        """
        return [(voice_label(n, v), n) for n, v in sorted_voices(self._preset_voices)]

    def _register_aliases(self, name: str, aliases) -> None:
        for a in aliases or []:
            if a and a != name:
                self._voice_aliases[str(a)] = name

    def resolve_voice_name(self, name: Optional[str]) -> Optional[str]:
        """Canonical preset name for ``name`` (itself, or the preset an alias points
        to); ``None`` if unknown. A real preset always wins over an alias."""
        if name is None:
            return None
        if name in self._preset_voices:
            return name
        target = self._voice_aliases.get(name)
        return target if target in self._preset_voices else None

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = self.resolve_voice_name(voice_name or self._default_voice)
        if name is None:
            raise ValueError(f"Voice '{voice_name or self._default_voice}' not found. "
                             f"Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    REF_CACHE_MAX = 32   # distinct clips kept (LRU); each entry is a few hundred KB

    def _enroll_reference(self, ref_audio: Union[str, Path], denoise: bool, use_ref_codes: bool
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Trim → engine.prepare_reference, memoised on the clip's content."""
        import hashlib
        import os
        key = None
        try:
            # blake2b: fast content fingerprint (not a security boundary, but keeps
            # the weak-hash linter quiet); 16-byte digest is plenty for a cache key.
            digest = hashlib.blake2b(Path(ref_audio).read_bytes(), digest_size=16).hexdigest()
            key = (digest, bool(denoise), bool(use_ref_codes))
        except OSError:
            pass   # unreadable path: let the engine raise its own error below
        if key is not None and key in self._ref_cache:
            self._ref_cache.move_to_end(key)
            return self._ref_cache[key]
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            out = self.engine.prepare_reference(str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass
        if key is not None:
            self._ref_cache[key] = out
            while len(self._ref_cache) > self.REF_CACHE_MAX:
                self._ref_cache.popitem(last=False)
        return out

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice from a wav → ``(speaker_emb, ref_codes)``.
        """
        return self._enroll_reference(ref_audio, denoise=denoise, use_ref_codes=True)

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
                  style: Any = None, save: bool = False) -> str:
        """Register a custom voice under ``name`` for use as ``infer(..., voice=name)``.

        ``ref_audio`` is enrolled once (denoised + trimmed, then speaker embedding +
        reference codes are extracted). Pass ``denoise=False`` if the clip is already
        clean. Set ``save=True`` to persist it to the voices file for later sessions.

        ``style`` is deprecated: it is stored as metadata only and never changes how
        the voice is synthesized (the style is implied by the reference itself).
        """
        if not name or not str(name).strip():
            raise ValueError("Tên giọng không được để trống.")
        import os
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            speaker_emb, ref_codes = self.engine.prepare_reference(
                str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass
        self._preset_voices[name] = {
            "description": description,
            "gender": gender,
            "style": style or self.default_style,   # metadata only (deprecated)
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
        name = self.resolve_voice_name(name) or name
        self._preset_voices.pop(name, None)
        self._voice_aliases = {a: t for a, t in self._voice_aliases.items() if t != name}
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
            if v.get("featured") is not None:
                presets[n]["featured"] = v["featured"]
            aliases = [a for a, t in self._voice_aliases.items() if t == n]
            if aliases:
                presets[n]["aliases"] = aliases
        data = {"meta": {"note": "v3 turbo voices: speaker embedding + reference codes; "
                                 "`featured` 1..N marks the editors' picks in display order"},
                "default_voice": self._default_voice, "presets": presets}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Saved {len(presets)} voices → {path}")
        return str(path)

    def _resolve_ref(self, voice, ref_audio, denoise, use_ref_codes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve the requested voice to ``(speaker_emb, ref_codes)``.

        Precedence: cloned ``ref_audio`` → preset ``voice`` (name or dict) → default preset.
        """
        if ref_audio is not None:
            return self._enroll_reference(ref_audio, denoise=denoise, use_ref_codes=use_ref_codes)
        preset = None
        if isinstance(voice, str):
            name = self.resolve_voice_name(voice)
            preset = self._preset_voices.get(name) if name else None
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

    # ── Batched inference (GPU/PyTorch) ──────────────────────────────────────
    def _get_batch_engine(self):
        """Lazily build the static-batching engine. Returns ``None`` on CPU/ONNX.

        The batching runtime (``vieneu.v3_turbo_serve``) is pure PyTorch/CUDA, so it
        only applies to the ``pytorch`` backend. Importing it is deferred until the
        first batched call to keep the torch-free CPU path from touching torch.
        """
        if self.backend != "pytorch":
            return None
        if self._batch_engine is None:
            # A server's loader thread (warm-up) and its first request can get
            # here together; both must end up with the same engine.
            with self._batch_lock:
                if self._batch_engine is None:
                    from .v3_turbo_serve import V3TurboBatchEngine
                    engine = V3TurboBatchEngine(self.engine)
                    engine.babble_retries = self.babble_retries
                    self._batch_engine = engine
        return self._batch_engine

    def _fused_available(self) -> bool:
        """The batch engine runs frames as CUDA graphs here (see fused.py)."""
        eng = self._get_batch_engine()
        return bool(eng is not None and getattr(eng, "use_fused", False)
                    and getattr(self.engine.device, "type", "") == "cuda")

    def _get_stream_scheduler(self):
        """Lazily build the continuous-batching stream scheduler (CUDA only).

        Returns ``None`` on CPU/ONNX or when the fused graphs are disabled
        (``VIENEU_FUSED_FRAME=0``); ``infer_stream`` then takes the engine's
        own single-sequence path.
        """
        if not self._fused_available():
            return None
        # The scheduler is captured under _stream_lock: a thread holding
        # CAPTURE_LOCK must not wait for it.
        from ._v3_turbo_engine.capture_lock import CAPTURE_LOCK
        CAPTURE_LOCK.ensure_not_held("infer_stream")
        with self._stream_lock:
            if self._stream_sched is None:
                from .v3_turbo_serve.stream import V3TurboStreamScheduler
                self._stream_sched = V3TurboStreamScheduler(
                    self._get_batch_engine(), max_streams=self.max_streams,
                )
                logger.info(f"✅ v3 Turbo stream scheduler ready (max_streams={self.max_streams})")
        return self._stream_sched

    def _infer_chunks(
        self,
        chunks: List[str],
        speaker_emb: np.ndarray,
        ref_codes: Optional[np.ndarray],
        use_ref_codes: bool,
        batch_size: int,
        sampling: dict,
    ) -> List[np.ndarray]:
        """Synthesize one waveform per chunk (no join, no watermark).

        On the PyTorch/GPU backend with ``batch_size > 1`` and more than one chunk,
        the chunks are pushed through the static-batching engine in groups of
        ``batch_size`` (chunks share each forward step — the throughput win). Groups
        are formed over chunks SORTED by phoneme length (length bucketing): batching
        similar-length prompts together minimizes left-padding, which cuts wasted
        prefill compute and padding-induced numeric noise. On CPU (ONNX) or a single
        chunk, each chunk goes through the single-sequence engine sequentially.
        Output order always matches ``chunks``.
        """
        n = len(chunks)
        engine = self._get_batch_engine() if (batch_size > 1 and n > 1) else None
        # A single chunk (or batch_size=1) used to take the single-sequence
        # engine. On CUDA the batch engine's fused CUDA graph beats it even
        # at B=1 — measured 2.3 s → 0.38 s for one sentence — so every chunk
        # goes through the batch engine there; groups of ``batch_size`` still
        # decide how many share a forward.
        if engine is None and self._fused_available():
            engine = self._get_batch_engine()
        phs = [phonemize_text_with_emotions(c) for c in chunks]

        def _one(ph: str) -> np.ndarray:
            return self.engine.infer(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                use_ref_codes=use_ref_codes,
                **_cap_frames(sampling, max_expected_frames(ph)),
            )

        if engine is None:
            wavs: List[np.ndarray] = [_one(ph) for ph in phs]
            return wavs

        order = sorted(range(n), key=lambda i: len(phs[i]))
        wavs = [None] * n
        for i in range(0, n, batch_size):
            idxs = order[i:i + batch_size]
            reqs = [{
                "phonemes": phs[j],
                "speaker_emb": speaker_emb, "ref_codes": ref_codes,
                "use_ref_codes": use_ref_codes,
            } for j in idxs]
            # Trần chung cho cả group = trần của row dài nhất — group đã được
            # bucket theo độ dài phoneme nên trần vẫn sát với từng row.
            group_cap = max(max_expected_frames(phs[j]) for j in idxs)
            for j, w in zip(idxs, engine.generate_batch(reqs, **_cap_frames(sampling, group_cap))):
                wavs[j] = w
        return wavs

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua — style đã nằm trong ref code (luôn tự nhiên)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,   # GPU: trần chunk/forward (None → self.max_batch_size; 1 → tắt batch)
        **kwargs: Any,
    ) -> np.ndarray:
        """Synthesize ``text`` into one 48 kHz waveform.

        ``style`` is deprecated and ignored (kept only so older code keeps running):
        the reading style comes from the reference voice, so output is always the
        natural style.
        """
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)

        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        # GPU gộp các chunk vào cùng forward; CPU/1-chunk chạy tuần tự (xem _infer_chunks).
        all_wavs = self._infer_chunks(
            chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

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
        style: Any = None,   # DEPRECATED: bỏ qua (xem `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize ``text`` and yield 48 kHz float32 audio as it is generated.

        On CUDA every call goes through one shared scheduler (see
        ``v3_turbo_serve/stream.py``): concurrent calls from different threads
        share a CUDA graph and one codec session, so a server can stream to
        many listeners at once — first audio in ~110-135 ms with up to 8
        streams on an RTX 3060. On CPU/ONNX the engine's own frame-level
        streaming runs, one call at a time.
        """
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        pauses = gaps_to_silence(gaps)
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
        )
        sched = self._get_stream_scheduler()
        if sched is not None:
            per_chunk = self._stream_chunks_gpu(sched, chunks, speaker_emb, ref_codes,
                                                use_ref_codes, max_new_frames, sampling)
        else:
            per_chunk = self._stream_chunks_engine(chunks, speaker_emb, ref_codes,
                                                   use_ref_codes, max_new_frames, sampling)
        sr = self.sample_rate
        last_out: Optional[np.ndarray] = None   # mẩu audio cuối đã phát của chunk trước
        for ci, subs in enumerate(per_chunk):
            first = True
            for sub in subs:
                if sub is None or len(sub) == 0:
                    continue
                if first and last_out is not None:
                    # Khoảng nghỉ giữa hai text chunk = TỔNG theo loại ranh giới, cùng
                    # bảng với join_audio_chunks. Audio chunk trước đã phát đi nên không
                    # trim được, chỉ bù zeros cho đủ (đuôi model tự phát dài hơn thì giữ).
                    pad = pause_pad_samples(last_out, sub, sr, pauses[ci - 1])
                    if pad > 0:
                        yield np.zeros(pad, dtype=np.float32)
                first = False
                last_out = sub
                yield self._apply_watermark(sub) if apply_watermark else sub

    def _stream_chunks_engine(self, chunks, speaker_emb, ref_codes, use_ref_codes,
                              max_new_frames, sampling):
        """One audio iterator per text chunk from the engine's own streaming
        (PyTorch single path or ONNX); a full ``infer`` per chunk if it has none."""
        stream_fn = getattr(self.engine, "infer_stream", None)
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            gen_kwargs = dict(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                use_ref_codes=use_ref_codes,
                max_new_frames=min(max_new_frames, max_expected_frames(ph)), **sampling,
            )
            yield stream_fn(**gen_kwargs) if stream_fn is not None else (self.engine.infer(**gen_kwargs),)

    def _stream_chunks_gpu(self, sched, chunks, speaker_emb, ref_codes, use_ref_codes,
                           max_new_frames, sampling):
        """One audio iterator per text chunk through the stream scheduler.

        Chunk ``i+1`` is submitted the moment chunk ``i``'s last frame is
        generated (its tail audio is still being decoded), so one call holds
        one scheduler slot and the join between chunks costs a prefill, not a
        whole chunk's wait. Abandoning the iterator cancels whatever is queued.
        """
        def submit(ci):
            ph = phonemize_text_with_emotions(chunks[ci])
            return sched.submit(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                use_ref_codes=use_ref_codes,
                max_new_frames=min(max_new_frames, max_expected_frames(ph)), **sampling,
            )

        state = {"next": None}

        def audio_of(handle, ci):
            nxt = None
            try:
                for sub in handle:
                    if nxt is None and ci + 1 < len(chunks) and handle.gen_done.is_set():
                        nxt = submit(ci + 1)
                    yield sub
                if nxt is None and ci + 1 < len(chunks):
                    nxt = submit(ci + 1)
            finally:
                handle.close()
                state["next"] = nxt

        if not chunks:
            return
        handle = submit(0)
        try:
            for ci in range(len(chunks)):
                state["next"] = None
                yield audio_of(handle, ci)
                handle = state["next"]
                if handle is None and ci + 1 < len(chunks):
                    handle = submit(ci + 1)
        finally:
            for h in (handle, state["next"]):
                if h is not None:
                    h.close()

    def infer_batch(
        self,
        texts: List[str],
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua (xem `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Synthesize many texts, returning one waveform each (same order as ``texts``).

        All texts share one voice (resolved once). On the PyTorch/GPU backend the
        chunks from *every* text are flattened and batched together in groups of
        ``batch_size`` — so even many short texts fill the batch and share forward steps.
        On CPU (ONNX) this runs sequentially. Output equals per-text ``infer`` (equivalent,
        not bit-identical to the single path, since sampling runs batched).
        """
        if not texts:
            return []

        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )

        # Cắt chunk từng text, nhớ chunk thuộc text nào (owner) và gaps để join lại.
        per_text_gaps: List[Any] = []
        flat_chunks: List[str] = []
        owner: List[int] = []
        for ti, t in enumerate(texts):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(t, max_chars=max_chars)
            per_text_gaps.append(gaps)
            for c in chunks:
                flat_chunks.append(c)
                owner.append(ti)

        empty = np.array([], dtype=np.float32)
        if not flat_chunks:
            return [empty for _ in texts]

        flat_wavs = self._infer_chunks(
            flat_chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Gom wav về đúng text (thứ tự flat_wavs khớp flat_chunks/owner), rồi join từng text.
        grouped: List[List[np.ndarray]] = [[] for _ in texts]
        for w, ti in zip(flat_wavs, owner):
            grouped[ti].append(w)

        results: List[np.ndarray] = []
        for ti in range(len(texts)):
            if not grouped[ti]:
                results.append(empty)
                continue
            joined = join_audio_chunks(
                grouped[ti], self.sample_rate, silence_ps=gaps_to_silence(per_text_gaps[ti])
            )
            results.append(self._apply_watermark(joined) if apply_watermark else joined)
        return results

    def close(self) -> None:
        if self._stream_sched is not None:
            self._stream_sched.close()
            self._stream_sched = None
        self.engine = None
        self._batch_engine = None
