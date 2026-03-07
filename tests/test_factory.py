from unittest.mock import patch

import pytest

from vieneu.factory import Vieneu
from vieneu.fast import FastVieNeuTTS
from vieneu.remote import RemoteVieNeuTTS
from vieneu.standard import VieNeuTTS


def test_factory_standard():
    with patch('vieneu.standard.VieNeuTTS.__init__', return_value=None):
        tts = Vieneu(mode='standard')
        assert isinstance(tts, VieNeuTTS)

def test_factory_fast():
    with patch('vieneu.fast.FastVieNeuTTS.__init__', return_value=None):
        tts = Vieneu(mode='fast')
        assert isinstance(tts, FastVieNeuTTS)

def test_factory_remote():
    with patch('vieneu.remote.RemoteVieNeuTTS.__init__', return_value=None):
        tts = Vieneu(mode='remote')
        assert isinstance(tts, RemoteVieNeuTTS)

def test_factory_xpu_error():
    # XPU will likely fail in this environment due to lack of torch.xpu
    with pytest.raises(RuntimeError, match="Failed to load XPU backend"):
        Vieneu(mode='xpu')

def test_factory_xpu_success():
    # Mocking torch.xpu availability is complex, but we can mock the class instantiation
    with patch('torch.xpu.is_available', return_value=True), \
         patch('vieneu.core_xpu.XPUVieNeuTTS.__init__', return_value=None):
             tts = Vieneu(mode='xpu')
             from vieneu.core_xpu import XPUVieNeuTTS
             assert isinstance(tts, XPUVieNeuTTS)
