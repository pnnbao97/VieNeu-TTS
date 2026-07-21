import os
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from vieneu.cpp import CppVieNeuTTS, _resolve_asset_path


def test_resolve_asset_path_explicit():
    path = _resolve_asset_path("/custom/path", "DUMMY_ENV", "dummy/rel", "Asset", required=True)
    assert path == "/custom/path"


def test_resolve_asset_path_env(monkeypatch):
    monkeypatch.setenv("TEST_ASSET_ENV", "/env/path/to/asset")
    path = _resolve_asset_path(None, "TEST_ASSET_ENV", "dummy/rel", "Asset", required=True)
    assert path == "/env/path/to/asset"


def test_resolve_asset_path_missing_required():
    with pytest.raises(ValueError, match="Asset was not found"):
        _resolve_asset_path(None, "NON_EXISTENT_ENV_VAR_12345", "non/existent/path/999.bin", "Asset", required=True)


def test_resolve_asset_path_missing_optional():
    path = _resolve_asset_path(None, "NON_EXISTENT_ENV_VAR_12345", "non/existent/path/999.bin", "Asset", required=False)
    assert path is None


def test_cpp_vieneu_tts_init_defaults():
    tts = CppVieNeuTTS(
        binary_path="/fake/audiocpp_cli",
        model_path="/fake/model.gguf",
        default_ref_audio="/fake/sample.wav",
        threads=8,
        timeout=45.0,
    )
    assert tts.binary_path == "/fake/audiocpp_cli"
    assert tts.model_path == "/fake/model.gguf"
    assert tts.default_ref_audio == "/fake/sample.wav"
    assert tts.threads == 8
    assert pytest.approx(tts.timeout) == 45.0
    assert tts.sample_rate == 48000


@patch("vieneu.cpp.sf.read")
@patch("vieneu.cpp.phonemize_text_with_emotions")
@patch("subprocess.run")
def test_cpp_vieneu_tts_infer_success(mock_run, mock_phonemize, mock_sf_read, tmp_path):
    mock_phonemize.return_value = "s i n c h a o"
    mock_sf_read.return_value = (np.zeros(48000, dtype=np.float32), 48000)

    dummy_ref = tmp_path / "ref.wav"
    dummy_ref.write_text("dummy audio")

    tts = CppVieNeuTTS(
        binary_path="/fake/audiocpp_cli",
        model_path="/fake/model.gguf",
        default_ref_audio=str(dummy_ref),
    )

    audio = tts.infer("Xin chào", apply_watermark=False)
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 48000

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    kwargs = mock_run.call_args[1]

    assert cmd[0] == "/fake/audiocpp_cli"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "/fake/model.gguf"
    assert "--text" in cmd and cmd[cmd.index("--text") + 1] == "s i n c h a o"
    assert "--voice-ref" in cmd and cmd[cmd.index("--voice-ref") + 1] == str(dummy_ref)
    assert "--reference-text" not in cmd
    assert pytest.approx(kwargs.get("timeout")) == 60.0


@patch("vieneu.cpp.phonemize_text_with_emotions")
@patch("subprocess.run")
def test_cpp_vieneu_tts_infer_timeout(mock_run, mock_phonemize, tmp_path):
    mock_phonemize.return_value = "s i n c h a o"
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["audiocpp_cli"], timeout=10.0)

    dummy_ref = tmp_path / "ref.wav"
    dummy_ref.write_text("dummy audio")

    tts = CppVieNeuTTS(
        binary_path="/fake/audiocpp_cli",
        model_path="/fake/model.gguf",
        default_ref_audio=str(dummy_ref),
        timeout=10.0,
    )

    with pytest.raises(RuntimeError, match=r"C\+\+ TTS execution timed out after 10.0 seconds."):
        tts.infer("Xin chào")
