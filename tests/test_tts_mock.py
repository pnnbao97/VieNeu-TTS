from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vieneu.standard import VieNeuTTS


@pytest.fixture
def mock_tts():
    # We use create=True because Llama might not be importable if llama-cpp-python is missing
    with patch('vieneu.standard.Llama', create=True) as mock_llama, \
         patch('vieneu.standard.DistillNeuCodec', create=True) as mock_codec, \
         patch('vieneu.standard.NeuCodec', create=True) as mock_neucodec, \
         patch('vieneu.base.hf_hub_download') as mock_hf_download:

        # Setup mock codec
        mock_codec_instance = MagicMock()
        mock_codec.from_pretrained.return_value = mock_codec_instance
        mock_codec_instance.hop_length = 480
        mock_codec_instance.sample_rate = 24000
        mock_codec_instance.decode_code.return_value = torch.zeros((1, 1, 1000))

        # Setup mock llama
        mock_llama_instance = MagicMock()
        mock_llama.from_pretrained.return_value = mock_llama_instance
        mock_llama_instance.return_value = {"choices": [{"text": "<|speech_1|><|speech_2|>"}]}

        # Setup mock voices.json
        mock_hf_download.return_value = "dummy_voices.json"

        with patch.object(VieNeuTTS, '_load_backbone'), \
             patch.object(VieNeuTTS, '_load_codec'), \
             patch.object(VieNeuTTS, '_warmup_model'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', MagicMock()), \
             patch('json.load', return_value={"presets": {"test_voice": {"codes": [1, 2], "text": "test"}}, "default_voice": "test_voice"}):
                tts = VieNeuTTS(backbone_repo="dummy-gguf", codec_repo="neuphonic/distill-neucodec")
                # Manually set some attributes that would have been set by _load_backbone/_load_codec
                tts.codec = mock_codec_instance
                tts.backbone = mock_llama_instance
                tts._is_quantized_model = True

                # Re-run _load_voices because it might have failed in __init__ due to patches not being active yet
                # Actually, the patches ARE active because they are in the 'with' block above.
                # But Path.exists(dummy_voices.json) might have failed if dummy_voices.json is not a real file.
                # Let's manually inject the voices
                tts._preset_voices = {"test_voice": {"codes": [1, 2], "text": "test"}}
                tts._default_voice = "test_voice"

                return tts

def test_tts_infer_mock(mock_tts):
    with patch.object(VieNeuTTS, '_decode', return_value=np.zeros(1000)):
        wav = mock_tts.infer("Xin chào", max_chars=10)
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0

def test_tts_infer_batch_mock(mock_tts):
    with patch.object(VieNeuTTS, '_decode', return_value=np.zeros(1000)):
        texts = ["Xin chào", "Chào buổi sáng"]
        results = mock_tts.infer_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2
        for res in results:
            assert isinstance(res, np.ndarray)

def test_tts_infer_multi_chunk_mock(mock_tts):
    with patch.object(VieNeuTTS, '_decode', return_value=np.zeros(1000)):
        # Force multiple chunks by using very small max_chars
        wav = mock_tts.infer("Xin chào Việt Nam thân yêu của tôi", max_chars=5)
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0

def test_tts_list_voices(mock_tts):
    voices = mock_tts.list_preset_voices()
    assert len(voices) > 0
    assert voices[0][1] == "test_voice"
