import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import soundfile as sf

from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

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

        # Resolve C++ CLI binary path
        if binary_path is None:
            binary_path = os.environ.get("VIENEU_CPP_BINARY_PATH")
            if not binary_path:
                possible_paths = [
                    Path.cwd() / "audio.cpp/build/linux-cpu-release/bin/audiocpp_cli",
                    Path.home() / "git/audio.cpp/build/linux-cpu-release/bin/audiocpp_cli",
                    Path.home() / "audio.cpp/build/linux-cpu-release/bin/audiocpp_cli",
                    Path(__file__).parents[3] / "ref/audio.cpp/build/linux-cpu-release/bin/audiocpp_cli",
                    Path(__file__).parents[4] / "audio.cpp/build/linux-cpu-release/bin/audiocpp_cli",
                ]
                for p in possible_paths:
                    if p.exists():
                        binary_path = str(p)
                        break
            if not binary_path:
                raise ValueError(
                    "C++ CLI binary `audiocpp_cli` was not found in default paths.\n"
                    "Please either:\n"
                    "1. Provide `binary_path` explicitly during initialization: Vieneu(mode='cpp', binary_path='path/to/audiocpp_cli')\n"
                    "2. Set the environment variable `VIENEU_CPP_BINARY_PATH` to point to it.\n"
                    "3. Clone and build the C++ model runner from the vietneu-tts-v3-turbo branch:\n"
                    "   git clone https://github.com/phuocnguyen90/audio.cpp.git\n"
                    "   cd audio.cpp && git checkout vietneu-tts-v3-turbo && ./scripts/build_linux.sh --backend cpu --target audiocpp_cli"
                )
        self.binary_path = str(binary_path)

        # Resolve GGUF model path
        if model_path is None:
            model_path = os.environ.get("VIENEU_CPP_MODEL_PATH")
            if not model_path:
                possible_paths = [
                    Path.cwd() / "audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf",
                    Path.home() / "git/audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf",
                    Path.home() / "audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf",
                    Path(__file__).parents[3] / "ref/audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf",
                    Path(__file__).parents[4] / "audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf",
                ]
                for p in possible_paths:
                    if p.exists():
                        model_path = str(p)
                        break
            if not model_path:
                raise ValueError(
                    "GGUF model file was not found in default paths.\n"
                    "Please either:\n"
                    "1. Provide `model_path` explicitly: Vieneu(mode='cpp', model_path='path/to/model.gguf')\n"
                    "2. Set the environment variable `VIENEU_CPP_MODEL_PATH`.\n"
                    "3. Download the GGUF weights to `audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf`."
                )
        self.model_path = str(model_path)

        self.threads = threads
        self.sample_rate = 48000

        # Resolve default voice-cloning reference files
        self.default_ref_audio = os.environ.get("VIENEU_CPP_REF_AUDIO")
        if not self.default_ref_audio:
            possible_ref_paths = [
                Path.cwd() / "audio.cpp/assets/resources/sample.wav",
                Path.home() / "git/audio.cpp/assets/resources/sample.wav",
                Path.home() / "audio.cpp/assets/resources/sample.wav",
                Path(__file__).parents[3] / "ref/audio.cpp/assets/resources/sample.wav",
                Path(__file__).parents[4] / "audio.cpp/assets/resources/sample.wav",
            ]
            for p in possible_ref_paths:
                if p.exists():
                    self.default_ref_audio = str(p)
                    break
        
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

        # 2. Build temporary output file
        temp_out = "/tmp/cpp_out_tmp.wav"
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass

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
            "--subtalker-temperature", str(subtalker_temperature),
            "--threads", str(self.threads),
            "--out", temp_out
        ]

        try:
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 4. Load the generated audio output
            audio, sr = sf.read(temp_out, dtype="float32")
            if os.path.exists(temp_out):
                os.remove(temp_out)
            return audio
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"C++ TTS execution failed: {stderr_output}") from e

    def infer_batch(self, texts: List[str], **kwargs: Any) -> List[np.ndarray]:
        return [self.infer(t, **kwargs) for t in texts]
