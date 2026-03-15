import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from vieneu.standard import VieNeuTTS
from pathlib import Path

@pytest.fixture
def mock_torch_components():
    tokenizer = MagicMock()
    tokenizer.pad.return_value = {"input_ids": torch.zeros((2, 10), dtype=torch.long)}
    tokenizer.convert_tokens_to_ids.side_effect = lambda x: 1001 if x.startswith("<|") else 999
    tokenizer.decode.return_value = "<|speech_1|><|speech_2|>"
    
    def mocked_encode(text, **kwargs):
        if "TEXT_REPLACE" in text or "TEXT_PROMPT_START" in text:
             return [10, 1003, 11, 1001]
        return [1, 2, 3]
    tokenizer.encode.side_effect = mocked_encode

    model = MagicMock()
    model.device = torch.device("cpu")
    model.to.return_value = model
    model.generate.return_value = torch.zeros((2, 20), dtype=torch.long)

    codec = MagicMock()
    codec.device = torch.device("cpu")
    codec.sample_rate = 24000
    codec.decode_code.return_value = torch.zeros((1, 1, 4800))
    codec.encode_code.return_value = torch.zeros((1, 1, 10), dtype=torch.long)

    return {"tokenizer": tokenizer, "model": model, "codec": codec}

@pytest.fixture
def mock_tts_instance(mock_torch_components):
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_torch_components["tokenizer"]), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_torch_components["model"]), \
         patch("vieneu.standard.NeuCodec.from_pretrained", return_value=mock_torch_components["codec"]), \
         patch("vieneu.standard.DistillNeuCodec.from_pretrained", return_value=mock_torch_components["codec"]), \
         patch("vieneu.base.hf_hub_download", return_value="dummy_path"), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"presets": {"test_voice": {"codes": [1, 2], "text": "test"}}, "default_voice": "test_voice"}), \
         patch.object(VieNeuTTS, '_warmup_model'):
        
        tts = VieNeuTTS(backbone_repo="dummy", backbone_device="cpu")
        tts._preset_voices = {"test_voice": {"codes": [1, 2, 3], "text": "test"}}
        return tts

def test_vieneu_tts_init(mock_tts_instance):
    assert mock_tts_instance.backbone is not None
    assert mock_tts_instance.codec is not None

def test_vieneu_tts_infer(mock_tts_instance):
    with patch("vieneu.standard.phonemize_with_dict", return_value="phonemes"):
        audio = mock_tts_instance.infer("Xin chào", ref_codes=[1, 2, 3], ref_text="Chào")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 4800

def test_vieneu_tts_infer_with_voice_preset(mock_tts_instance):
    with patch("vieneu.standard.phonemize_with_dict", return_value="phonemes"):
        audio = mock_tts_instance.infer("Xin chào", voice=mock_tts_instance.get_preset_voice("test_voice"))
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 4800

def test_vieneu_tts_streaming(mock_tts_instance):
    with patch("vieneu.standard.phonemize_with_dict", return_value="phonemes"):
        stream = mock_tts_instance.infer_stream("Xin chào", ref_codes=[1, 2, 3], ref_text="Chào")
        chunks = list(stream)
        assert len(chunks) > 0
        assert isinstance(chunks[0], np.ndarray)

def test_vieneu_tts_infer_batch(mock_tts_instance):
    mock_tts_instance._is_quantized_model = False # Force torch path
    texts = ["Text 1", "Text 2"]
    with patch("vieneu.standard.phonemize_batch", return_value=["p1", "p2"]), \
         patch.object(mock_tts_instance, '_decode', return_value=np.zeros(1000)):
        results = mock_tts_instance.infer_batch(texts, ref_codes=[1], ref_text="ref")
        assert len(results) == 2

def test_lora_loading_logic(mock_tts_instance):
    with patch("peft.PeftModel.from_pretrained") as mock_peft:
        mock_tts_instance.load_lora_adapter("lora_repo")
        assert mock_tts_instance._lora_loaded is True
        mock_peft.assert_called_once()

        with patch.object(mock_tts_instance.backbone, 'unload', return_value=mock_tts_instance.backbone):
            mock_tts_instance.unload_lora_adapter()
            assert mock_tts_instance._lora_loaded is False

def test_base_encode_reference_device(mock_tts_instance):
    with patch("librosa.load", return_value=(np.zeros(16000), 16000)):
        with patch.object(torch.Tensor, 'to', wraps=torch.zeros(1).to) as mock_to:
            mock_tts_instance.encode_reference("dummy.wav")
            mock_to.assert_called()
