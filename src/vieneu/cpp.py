import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import soundfile as sf

from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

# Shared literals defined as module-level constants to avoid duplication
CLI_REL_PATH = "audio.cpp/build/linux-cpu-release/bin/audiocpp_cli"
MODEL_REL_PATH = "audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf"
REF_AUDIO_REL_PATH = "audio.cpp/assets/resources/sample.wav"

def _resolve_binary_path(binary_path: Optional[str]) -> str:
    """Helper to resolve C++ CLI binary path while keeping cognitive complexity low."""
    if binary_path is not None:
        return str(binary_path)

    env_path = os.environ.get("VIENEU_CPP_BINARY_PATH")
    if env_path:
        return env_path

    possible_roots = [Path.cwd(), Path.home(), Path.home() / "git"]
    for root in possible_roots:
        p = root / CLI_REL_PATH
        if p.exists():
            return str(p)

    pkg_fallback = Path(__file__).parents[4] / CLI_REL_PATH
    if pkg_fallback.exists():
        return str(pkg_fallback)

    pkg_fallback_ref = Path(__file__).parents[3] / "ref" / CLI_REL_PATH
    if pkg_fallback_ref.exists():
        return str(pkg_fallback_ref)

    raise ValueError(
        "C++ CLI binary `audiocpp_cli` was not found in default paths.\n"
        "Please either:\n"
        "1. Provide `binary_path` explicitly during initialization: Vieneu(mode='cpp', binary_path='path/to/audiocpp_cli')\n"
        "2. Set the environment variable `VIENEU_CPP_BINARY_PATH` to point to it.\n"
        "3. Clone and build the C++ model runner from the vietneu-tts-v3-turbo branch:\n"
        "   git clone https://github.com/phuocnguyen90/audio.cpp.git\n"
        "   cd audio.cpp && git checkout vietneu-tts-v3-turbo && ./scripts/build_linux.sh --backend cpu --target audiocpp_cli"
    )

def _resolve_model_path(model_path: Optional[str]) -> str:
    """Helper to resolve GGUF model path while keeping cognitive complexity low."""
    if model_path is not None:
        return str(model_path)

    env_path = os.environ.get("VIENEU_CPP_MODEL_PATH")
    if env_path:
        return env_path

    possible_roots = [Path.cwd(), Path.home(), Path.home() / "git"]
    for root in possible_roots:
        p = root / MODEL_REL_PATH
        if p.exists():
            return str(p)

    pkg_fallback = Path(__file__).parents[4] / MODEL_REL_PATH
    if pkg_fallback.exists():
        return str(pkg_fallback)

    pkg_fallback_ref = Path(__file__).parents[3] / "ref" / MODEL_REL_PATH
    if pkg_fallback_ref.exists():
        return str(pkg_fallback_ref)

    raise ValueError(
        "GGUF model file was not found in default paths.\n"
        "Please either:\n"
        "1. Provide `model_path` explicitly: Vieneu(mode='cpp', model_path='path/to/model.gguf')\n"
        "2. Set the environment variable `VIENEU_CPP_MODEL_PATH`.\n"
        "3. Download the GGUF weights to `audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf`."
    )

def _resolve_ref_audio(default_ref_audio: Optional[str]) -> Optional[str]:
    """Helper to resolve default reference audio path while keeping cognitive complexity low."""
    if default_ref_audio:
        return default_ref_audio

    env_path = os.environ.get("VIENEU_CPP_REF_AUDIO")
    if env_path:
        return env_path

    possible_roots = [Path.cwd(), Path.home(), Path.home() / "git"]
    for root in possible_roots:
        p = root / REF_AUDIO_REL_PATH
        if p.exists():
            return str(p)

    pkg_fallback = Path(__file__).parents[4] / REF_AUDIO_REL_PATH
    if pkg_fallback.exists():
        return str(pkg_fallback)

    pkg_fallback_ref = Path(__file__).parents[3] / "ref" / REF_AUDIO_REL_PATH
    if pkg_fallback_ref.exists():
        return str(pkg_fallback_ref)

    return None

class CppVieNeuTTS(BaseVieneuTTS):
    """
    C++ Backend wrapper for VieNeu-TTS v3 Turbo.
    Invokes the compiled `audiocpp_cli` binary under the hood for highly optimized,
    lightweight, and torch-free generation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        binary_path: Optional[str] = None,
        threads: int = 4,
        **kwargs: Any
    ):
        super().__init__()
        self.binary_path = _resolve_binary_path(binary_path)
        self.model_path = _resolve_model_path(model_path)
        self.threads = threads
        self.sample_rate = 48000
        self.default_ref_audio = _resolve_ref_audio(kwargs.get("default_ref_audio"))
        self.default_ref_text = (
            "Some call me nature. Others call me Mother Nature. "
            "I've been here for over 4.5 billion years. 22,500 times longer than you."
        )

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.8,
        subtalker_temperature: float = 0.8,
        **kwargs: Any
    ) -> np.ndarray:
        if ref_audio is None:
            ref_audio = self.default_ref_audio
            if ref_audio is None:
                raise ValueError("Must provide either a default reference audio or pass it via `ref_audio`.")
        if reference_text is None:
            reference_text = self.default_ref_text

        # 1. Run the phonemizer on the text input
        phonemes = phonemize_text_with_emotions(text)

        # 2. Build secure temporary output file (CWE-377 / CWE-379)
        fd, temp_out = tempfile.mkstemp(suffix=".wav")
        try:
            # Close file descriptor so subprocess can write to it
            os.close(fd)

            # 3. Construct and execute the CLI call
            cmd = [
                self.binary_path,
                "--task", "tts",
                "--family", "vietneu_tts",
                "--model", self.model_path,
                "--backend", "cpu",
                "--voice-ref", str(ref_audio),
                "--reference-text", reference_text,
                "--text", phonemes,
                "--temperature", str(temperature),
                "--request-option", f"subtalker_temperature={subtalker_temperature}",
                "--threads", str(self.threads),
                "--out", temp_out
            ]

            ref_emb = Path(str(ref_audio) + ".emb.txt")
            if ref_emb.exists():
                cmd.extend(["--request-option", f"speaker_embedding_file={ref_emb}"])

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 4. Load the generated audio output
            audio, _ = sf.read(temp_out, dtype="float32")
            return audio
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"C++ TTS execution failed: {stderr_output}") from e
        finally:
            if os.path.exists(temp_out):
                try:
                    os.remove(temp_out)
                except OSError:
                    pass

    def infer_batch(self, texts: List[str], **kwargs: Any) -> List[np.ndarray]:
        return [self.infer(t, **kwargs) for t in texts]
