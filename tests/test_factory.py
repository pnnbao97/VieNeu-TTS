import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock heavy modules before importing Vieneu
mock_torch = MagicMock()
# Ensure torch.Tensor is available for type hinting
mock_torch.Tensor = MagicMock
sys.modules["torch"] = mock_torch
sys.modules["torch.backends"] = mock_torch.backends
sys.modules["torch.backends.mps"] = mock_torch.backends.mps
sys.modules["llama_cpp"] = MagicMock()
sys.modules["lmdeploy"] = MagicMock()
sys.modules["neucodec"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()

from vieneu.factory import Vieneu

@patch("vieneu.turbo.TurboVieNeuTTS", create=True)
def test_factory_turbo(mock_turbo):
    Vieneu(mode="turbo")
    mock_turbo.assert_called_once()

@patch("vieneu.turbo.TurboGPUVieNeuTTS", create=True)
def test_factory_turbo_gpu(mock_turbo_gpu):
    Vieneu(mode="turbo_gpu")
    mock_turbo_gpu.assert_called_once()

@patch("vieneu.fast.FastVieNeuTTS", create=True)
def test_factory_fast(mock_fast):
    Vieneu(mode="fast")
    mock_fast.assert_called_once()

@patch("vieneu.standard.VieNeuTTS", create=True)
def test_factory_standard(mock_standard):
    Vieneu(mode="standard")
    mock_standard.assert_called_once()

@patch("vieneu.remote.RemoteVieNeuTTS", create=True)
def test_factory_remote(mock_remote):
    Vieneu(mode="remote")
    mock_remote.assert_called_once()

@patch("vieneu.core_xpu.XPUVieNeuTTS", create=True)
def test_factory_xpu(mock_xpu):
    Vieneu(mode="xpu")
    mock_xpu.assert_called_once()

def test_factory_invalid_mode():
    # Factory with unknown mode should return None (match-case without default)
    # The current implementation of factory.py ends with the last case
    assert Vieneu(mode="unknown") is None
