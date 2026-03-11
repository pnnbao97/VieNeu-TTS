from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Generator
import json
import torch
import numpy as np
import gc
import logging
from huggingface_hub import hf_hub_download
from sea_g2p import Normalizer

# Configure logging
logger = logging.getLogger("Vieneu")

class BaseVieneuTTS(ABC):
    """
    Abstract base class for VieNeu-TTS implementations.
    Provides shared functionality for voice management and common operations.
    """

    def __init__(self):
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480

        self.assets_dir = Path(__file__).parent / "assets"
        self._preset_voices: Dict[str, Any] = {}
        self._default_voice: Optional[str] = None
        self.normalizer = Normalizer()
        self._ref_phoneme_cache: Dict[str, str] = {}

        # Watermarker placeholder
        self.watermarker = None
        self._init_watermarker()

    def _init_watermarker(self) -> None:
        """Initialize optional audio watermarker."""
        try:
            import perth
            self.watermarker = perth.PerthImplicitWatermarker()
            logger.info("🔒 Audio watermarking initialized (Perth)")
        except (ImportError, AttributeError):
            self.watermarker = None

    def _load_voices(self, backbone_repo: Optional[str], hf_token: Optional[str] = None, clear_existing: bool = False) -> None:
        """Unified voice loading for Local and Remote paths."""
        if not backbone_repo:
            return

        path_obj = Path(backbone_repo)
        if path_obj.exists():
            # Local Path (Dir or File)
            if path_obj.is_dir():
                json_path = path_obj / "voices.json"
            else:
                json_path = path_obj.parent / "voices.json"

            if json_path.exists():
                self._load_voices_from_file(json_path, clear_existing=clear_existing)
            else:
                if clear_existing:
                     self._preset_voices.clear()
                logger.warning(f"Validation Warning: Local path '{backbone_repo}' missing 'voices.json'.")
                logger.warning(f"Falling back to Custom Voice Cloning mode.")
        else:
            # Remote Repo
            if clear_existing:
                self._preset_voices.clear()

            try:
                self._load_voices_from_repo(backbone_repo, hf_token)
            except Exception as e:
                logger.warning(f"Could not load voices from repo '{backbone_repo}': {e}")
                logger.warning(f"Falling back to Custom Voice Cloning mode.")

    def _load_voices_from_file(self, file_path: Path, clear_existing: bool = False) -> None:
        """Load voices from a local JSON file."""
        try:
            if not file_path.exists():
                logger.error(f"Voice file not found: {file_path}")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in voice file {file_path}: {e}")
                    return

            if "presets" in data:
                if clear_existing:
                    self._preset_voices.clear()
                    logger.info("🧹 Cleared existing voices for replacement")

                # Merge into existing presets
                self._preset_voices.update(data["presets"])
                logger.info(f"📢 Loaded {len(data['presets'])} voices from {file_path.name}")

            # Update default voice if provided
            if "default_voice" in data and data["default_voice"]:
                self._default_voice = data["default_voice"]

        except Exception as e:
            logger.error(f"Failed to load voices from {file_path}: {e}")

    def _load_voices_from_repo(self, repo_id: str, hf_token: Optional[str] = None) -> None:
        """Download and load voices.json from a HuggingFace repo."""
        voices_file = None
        try:
            # 1. Try normal download (checks for updates from server)
            voices_file = hf_hub_download(
                repo_id=repo_id,
                filename="voices.json",
                token=hf_token,
                repo_type="model"
            )
        except Exception:
            # 2. Network error? Try to use cached version if available
            logger.warning(f"Network check failed for voices.json. Trying local cache...")
            try:
                voices_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="voices.json",
                    token=hf_token,
                    repo_type="model",
                    local_files_only=True
                )
                logger.info(f"✅ Using cached voices.json")
            except Exception:
                # 3. No cache available either
                pass

        if voices_file:
            self._load_voices_from_file(Path(voices_file))
        else:
            logger.warning(f"Repository '{repo_id}' is missing 'voices.json'. Falling back to Custom Voice mode.")

    def list_preset_voices(self) -> List[tuple[str, str]]:
        """List available preset voices as (description, id)."""
        return [
            (v.get("description", k) if isinstance(v, dict) else str(v), k)
            for k, v in self._preset_voices.items()
        ]

    def get_preset_voice(self, voice_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get reference codes and text for a preset voice.

        Args:
            voice_name: Name of voice. If None, uses default_voice.

        Returns:
            dict: { 'codes': torch.Tensor, 'text': str }
        """
        if voice_name is None:
            voice_name = self._default_voice
            if voice_name is None:
                if self._preset_voices:
                    voice_name = next(iter(self._preset_voices))
                else:
                    raise ValueError("No voice specified and no preset voices available.")

        if voice_name not in self._preset_voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {self.list_preset_voices()}")

        voice_data = self._preset_voices[voice_name]
        codes = voice_data["codes"]
        if isinstance(codes, list):
            codes = torch.tensor(codes, dtype=torch.long)

        return {"codes": codes, "text": voice_data["text"]}

    def get_ref_phonemes(self, ref_text: str) -> str:
        """
        Get phonemized version of reference text, using cache if available.
        """
        if ref_text not in self._ref_phoneme_cache:
            from vieneu_utils.phonemize_text import phonemize_with_dict
            self._ref_phoneme_cache[ref_text] = phonemize_with_dict(ref_text)
        return self._ref_phoneme_cache[ref_text]

    def save(self, audio: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save audio waveform to a file."""
        import soundfile as sf
        sf.write(str(output_path), audio, self.sample_rate)

    def encode_reference(self, ref_audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Encode reference audio to codes.

        Args:
            ref_audio_path: Path to the reference audio file.

        Returns:
            torch.Tensor: Encoded codes.
        """
        import librosa
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]

        # Ensure device and dtype compatibility
        if hasattr(self.codec, "device"):
            wav_tensor = wav_tensor.to(self.codec.device)

        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes

    def _decode(self, codes_str: str) -> np.ndarray:
        """
        Decode speech tokens to audio waveform.

        Args:
            codes_str: String containing speech tokens.

        Returns:
            np.ndarray: Decoded audio waveform.
        """
        from .utils import extract_speech_ids
        speech_ids = extract_speech_ids(codes_str)

        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in the output.")

        # Onnx decode
        if getattr(self, "_is_onnx_codec", False):
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        # Torch decode
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :]
                if hasattr(self.codec, "device"):
                    codes = codes.to(self.codec.device)

                recon = self.codec.decode_code(codes)
                if hasattr(recon, "cpu"):
                    recon = recon.cpu()
                if hasattr(recon, "numpy"):
                    recon = recon.numpy()


        return recon[0, 0, :]

    def _resolve_ref_voice(
        self,
        voice: Optional[Dict[str, Any]] = None,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_text: Optional[str] = None
    ) -> tuple[Union[np.ndarray, torch.Tensor], str]:
        """Resolve reference voice codes and text."""
        if voice is not None:
            ref_codes = voice.get('codes', ref_codes)
            ref_text = voice.get('text', ref_text)

        if ref_audio is not None and ref_codes is None:
            ref_codes = self.encode_reference(ref_audio)
        elif self._default_voice and (ref_codes is None or ref_text is None):
            try:
                voice_data = self.get_preset_voice(None)
                ref_codes = voice_data['codes']
                ref_text = voice_data['text']
            except Exception:
                pass

        if ref_codes is None or ref_text is None:
            raise ValueError("Must provide either 'voice' dict or both 'ref_codes' and 'ref_text'.")

        return ref_codes, ref_text

    def _apply_watermark(self, wav: np.ndarray) -> np.ndarray:
        """Apply watermark to audio if enabled."""
        if self.watermarker:
            return self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        return wav

    def _format_prompt(
        self,
        ref_codes: Union[List[int], torch.Tensor, np.ndarray],
        ref_text: str,
        input_text: str,
        ref_phonemes: Optional[str] = None,
        input_phonemes: Optional[str] = None
    ) -> str:
        """
        Format the prompt for the TTS model.
        Common implementation for LMDeploy (Fast) and Remote backends.
        Standard backend uses a specialized chat template via tokenizer.
        """
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes_list = ref_codes.flatten().tolist()
        else:
            ref_codes_list = ref_codes

        # Import inside method to avoid potential circular dependencies between
        # base TTS and phonemization utilities.
        from vieneu_utils.phonemize_text import phonemize_with_dict

        ref_text_phones = ref_phonemes if ref_phonemes else self.get_ref_phonemes(ref_text)
        input_text_phones = input_phonemes if input_phonemes else phonemize_with_dict(input_text, skip_normalize=True)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])

        return (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

    @abstractmethod
    def infer(self, text: str, **kwargs: Any) -> np.ndarray:
        """Main inference method for single text."""
        pass

    @abstractmethod
    def infer_batch(self, texts: List[str], apply_watermark: bool = True, **kwargs: Any) -> List[np.ndarray]:
        """Main inference method for batch processing."""
        pass

    def close(self) -> None:
        """Release resources."""
        pass

    def __enter__(self) -> 'BaseVieneuTTS':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
