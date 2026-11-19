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


def _write_ref(path: Path, sr: int = 16000) -> np.ndarray:
    wav = np.zeros(sr * 2, dtype=np.float32)
    wav[sr // 2: sr * 3 // 2] = 0.5
    sf.write(path, wav, sr)
    return wav


class TestV3TurboClonePrecleanTempDirFallback:
    """~/.cache/vieneu/tmp unusable → cleaned clip goes to the system temp dir."""

    def _run(self, tmp_path: Path, monkeypatch):
        from vieneu.v3turbo import V3TurboVieNeuTTS

        sys_tmp = tmp_path / "systemp"
        sys_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(sys_tmp))

        src = tmp_path / "raw_ref.wav"
        wav = _write_ref(src)
        cleaned = Path(V3TurboVieNeuTTS._preclean_reference_audio(src))

        assert cleaned.exists()
        assert cleaned.parent == sys_tmp
        assert cleaned.name.startswith("temp_clone_optimized_")
        clean_wav, _ = sf.read(cleaned, dtype="float32", always_2d=False)
        assert len(clean_wav) < len(wav)

    def test_access_denied_creating_cache_dir(self, tmp_path, monkeypatch):
        # Reproduces "[WinError 5] Access is denied: 'C:\\Users\\PC\\.cache\\vieneu'".
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

        real_mkdir = Path.mkdir

        def deny_under_home(self, *args, **kwargs):
            if fake_home in self.parents:
                raise PermissionError(13, "Access is denied", str(self))
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", deny_under_home)
        self._run(tmp_path, monkeypatch)
        assert not (fake_home / ".cache").exists()

    def test_cache_path_blocked_by_a_file(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        (fake_home / ".cache").write_text("not a directory")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        self._run(tmp_path, monkeypatch)

    def test_no_home_directory(self, tmp_path, monkeypatch):
        def no_home(cls):
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(Path, "home", classmethod(no_home))
        self._run(tmp_path, monkeypatch)
