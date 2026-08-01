import importlib
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class TestV3TurboClonePreclean:
    def test_preclean_reference_audio_trims_silence_and_returns_path(self):
        from vieneu.v3turbo import V3TurboVieNeuTTS

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "raw_ref.wav"
            sr = 16000
            wav = np.zeros(sr * 2, dtype=np.float32)
            wav[sr // 2: sr * 3 // 2] = 0.5
            sf.write(src, wav, sr)

            cleaned = V3TurboVieNeuTTS._preclean_reference_audio(src)

            try:
                assert cleaned is not None
                assert Path(cleaned).exists()
                clean_wav, clean_sr = sf.read(cleaned, dtype="float32", always_2d=False)
                assert clean_sr == sr
                assert len(clean_wav) < len(wav)
            finally:
                if cleaned and Path(cleaned).exists():
                    try:
                        Path(cleaned).unlink()
                    except Exception:
                        pass
