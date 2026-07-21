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


def _resolve_asset_path(
    explicit_path: Optional[Union[str, Path]],
    env_var: str,
    rel_path: str,
    asset_name: str,
    required: bool = True,
    error_instructions: str = "",
) -> Optional[str]:
    """Helper to resolve asset paths cleanly while avoiding code duplication."""
    if explicit_path is not None:
        return str(explicit_path)

    env_path = os.environ.get(env_var)
    if env_path:
        return env_path

    possible_roots = [Path.cwd(), Path.home(), Path.home() / "git"]
    for root in possible_roots:
        p = root / rel_path
        if p.exists():
            return str(p)

    try:
        repo_root = Path(__file__).resolve().parents[2]
        p = repo_root / rel_path
        if p.exists():
            return str(p)
    except IndexError:
        pass

    if required:
        raise ValueError(
            f"{asset_name} was not found in default paths.\n"
            f"Please either:\n"
            f"1. Provide `{asset_name.lower().replace(' ', '_')}` explicitly during initialization.\n"
            f"2. Set the environment variable `{env_var}` to point to it.\n"
            f"{error_instructions}"
        )
    return None


class CppVieNeuTTS(BaseVieneuTTS):
    """
    C++ Backend wrapper for VieNeu-TTS v3 Turbo.
    Invokes the compiled `audiocpp_cli` binary under the hood for highly optimized,
    lightweight, and torch-free generation.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        binary_path: Optional[Union[str, Path]] = None,
        threads: int = 4,
        timeout: float = 60.0,
        default_ref_audio: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.binary_path = _resolve_asset_path(
            binary_path,
            "VIENEU_CPP_BINARY_PATH",
            CLI_REL_PATH,
            "C++ CLI binary `audiocpp_cli`",
            required=True,
            error_instructions=(
                "3. Clone and build the C++ model runner:\n"
                "   git clone https://github.com/0xShug0/audio.cpp.git\n"
                "   cd audio.cpp && ./scripts/build_linux.sh --backend cpu --target audiocpp_cli"
            ),
        )
        self.model_path = _resolve_asset_path(
            model_path,
            "VIENEU_CPP_MODEL_PATH",
            MODEL_REL_PATH,
            "GGUF model file",
            required=True,
            error_instructions="3. Download the GGUF weights to `audio.cpp/models/VieNeu-TTS-v3-Turbo/model.gguf`.",
        )
        self.default_ref_audio = _resolve_asset_path(
            default_ref_audio or kwargs.get("default_ref_audio"),
            "VIENEU_CPP_REF_AUDIO",
            REF_AUDIO_REL_PATH,
            "Default reference audio",
            required=False,
        )
        self.threads = threads
        self.timeout = timeout
        self.sample_rate = 48000

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        temperature: float = 0.8,
        subtalker_temperature: float = 0.8,
        apply_watermark: bool = True,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        audio_ref = ref_audio or self.default_ref_audio
        if audio_ref is None:
            raise ValueError("Must provide either a default reference audio or pass it via `ref_audio`.")

        eff_timeout = timeout if timeout is not None else self.timeout

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
                "--voice-ref", str(audio_ref),
                "--text", phonemes,
                "--temperature", str(temperature),
                "--request-option", f"subtalker_temperature={subtalker_temperature}",
                "--threads", str(self.threads),
                "--out", temp_out,
            ]

            ref_emb = Path(str(audio_ref) + ".emb.txt")
            if ref_emb.exists():
                cmd.extend(["--request-option", f"speaker_embedding_file={ref_emb}"])

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=eff_timeout,
            )

            # 4. Load the generated audio output
            audio, _ = sf.read(temp_out, dtype="float32")
            return self._apply_watermark(audio) if apply_watermark else audio
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"C++ TTS execution timed out after {eff_timeout} seconds.") from e
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
