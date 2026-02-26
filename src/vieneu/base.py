from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Dict, Any, Union, Optional
import json
import re
import gc
import numpy as np
import torch
import librosa
from huggingface_hub import hf_hub_download
from vieneu_utils.normalize_text import VietnameseTTSNormalizer

# Pre-compiled regex for speech token extraction
SPEECH_TOKEN_RE = re.compile(r"<\|speech_(\d+)\|>")

def _linear_overlap_add(frames: List[np.ndarray], stride: int) -> np.ndarray:
    """Join audio chunks using linear overlap-add (Optimized)."""
    if not frames:
        return np.array([], dtype=np.float32)

    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    # Calculate total size efficiently
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros((*shape, total_size), dtype=dtype)

    # Pre-calculate weight and avoid redundant operations in the loop
    last_fl = -1
    weight = None

    for i, frame in enumerate(frames):
        fl = frame.shape[-1]
        if fl != last_fl:
            # Re-calculate weight only when frame length changes
            t = np.linspace(0, 1, fl + 2, dtype=dtype)[1:-1]
            weight = np.abs(0.5 - (t - 0.5))
            last_fl = fl

        offset = i * stride
        out[..., offset : offset + fl] += weight * frame
        sum_weight[offset : offset + fl] += weight

    # Vectorized normalization with safety check for division by zero
    return out / np.maximum(sum_weight, 1e-6)

def _compile_codec_with_triton(codec):
    """Compile codec with Triton for faster decoding (Windows/Linux compatible)"""
    try:
        import triton

        if hasattr(codec, 'dec') and hasattr(codec.dec, 'resblocks'):
            if len(codec.dec.resblocks) > 2:
                codec.dec.resblocks[2].forward = torch.compile(
                    codec.dec.resblocks[2].forward,
                    mode="reduce-overhead",
                    dynamic=True
                )
                print("   ✅ Triton compilation enabled for codec")
        return True

    except ImportError:
        return False
    except Exception:
        return False

class VieneuBase(ABC):
    """Abstract base class for VieNeu-TTS implementations."""

    def __init__(self):
        # Constants
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480

        # Internal state
        self._preset_voices = {}
        self._default_voice = None
        self.assets_dir = Path(__file__).parent / "assets"
        self.normalizer = VietnameseTTSNormalizer()

        # Models (to be initialized by subclasses)
        self.backbone = None
        self.codec = None
        self.tokenizer = None

        # Watermarker
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
            print("   🔒 Audio watermarking initialized (Perth)")
        except (ImportError, AttributeError):
            self.watermarker = None

    @abstractmethod
    def _load_backbone(self, **kwargs):
        pass

    @abstractmethod
    def _load_codec(self, **kwargs):
        pass

    @abstractmethod
    def infer(self, text: str, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def infer_stream(self, text: str, **kwargs) -> Generator[np.ndarray, None, None]:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Explicitly release model resources."""
        _gc = globals().get("gc", None)
        _torch = globals().get("torch", None)

        try:
            if hasattr(self, "backbone") and self.backbone is not None:
                # For GGUF models, call close()
                close_fn = getattr(self.backbone, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                self.backbone = None

            if hasattr(self, "codec") and self.codec is not None:
                self.codec = None

            if _gc is not None:
                _gc.collect()

            if _torch is not None:
                if hasattr(_torch, "cuda") and _torch.cuda is not None:
                    if callable(getattr(_torch.cuda, "is_available", None)) and _torch.cuda.is_available():
                        if callable(getattr(_torch.cuda, "empty_cache", None)):
                            _torch.cuda.empty_cache()
        except Exception:
            pass

    def save(self, audio, output_path: str):
        """Save audio to file."""
        import soundfile as sf
        sf.write(output_path, audio, self.sample_rate)

    def _load_voices(self, backbone_repo, hf_token=None, clear_existing=False):
        """Unified voice loading for Local and Remote paths."""
        if not backbone_repo:
            return

        path_obj = Path(backbone_repo)
        if path_obj.exists():
            if path_obj.is_dir():
                json_path = path_obj / "voices.json"
            else:
                json_path = path_obj.parent / "voices.json"

            if json_path.exists():
                self._load_voices_from_file(json_path, clear_existing=clear_existing)
            else:
                if clear_existing:
                     self._preset_voices.clear()
                print(f"   ⚠️ Validation Warning: Local path '{backbone_repo}' missing 'voices.json'.")
        else:
            if clear_existing:
                self._preset_voices.clear()

            try:
                self._load_voices_from_repo(backbone_repo, hf_token)
            except Exception as e:
                print(f"   ⚠️ Warning: Could not load voices from repo '{backbone_repo}': {e}")

    def _load_voices_from_file(self, file_path: Path, clear_existing=False):
        """Load voices from a local JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "presets" in data:
                if clear_existing:
                    self._preset_voices.clear()

                self._preset_voices.update(data["presets"])
                print(f"   📢 Loaded {len(data['presets'])} voices from {file_path.name}")

            if "default_voice" in data and data["default_voice"]:
                self._default_voice = data["default_voice"]

        except Exception as e:
            print(f"   ⚠️ Failed to load voices from {file_path}: {e}")

    def _load_voices_from_repo(self, repo_id: str, hf_token=None):
        """Download and load voices.json from a HuggingFace repo."""
        voices_file = None
        try:
            voices_file = hf_hub_download(
                repo_id=repo_id,
                filename="voices.json",
                token=hf_token,
                repo_type="model"
            )
        except Exception:
            try:
                voices_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="voices.json",
                    token=hf_token,
                    repo_type="model",
                    local_files_only=True
                )
            except Exception:
                pass

        if voices_file:
            self._load_voices_from_file(Path(voices_file))

    def list_preset_voices(self):
        """List available preset voices as (description, id)."""
        return [
            (v.get("description", k) if isinstance(v, dict) else str(v), k)
            for k, v in self._preset_voices.items()
        ]

    def get_preset_voice(self, voice_name: str = None):
        """Get reference codes and text for a preset voice."""
        if voice_name is None:
            voice_name = self._default_voice
            if voice_name is None:
                if self._preset_voices:
                    voice_name = next(iter(self._preset_voices))
                else:
                    raise ValueError("No voice specified and no preset voices available.")

        if voice_name not in self._preset_voices:
            available = [name for _, name in self.list_preset_voices()]
            raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")

        voice_data = self._preset_voices[voice_name]
        codes = voice_data["codes"]
        if isinstance(codes, list):
            codes = torch.tensor(codes, dtype=torch.long)

        return {"codes": codes, "text": voice_data["text"]}

    def encode_reference(self, ref_audio_path: Union[str, Path]):
        """Encode reference audio to codes."""
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

        device = getattr(self.codec, "device", "cpu")
        wav_tensor = wav_tensor.to(device)

        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes
