import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from vieneu import VieNeuTTS, RemoteVieNeuTTS

@pytest.fixture
def mock_codec():
    codec = MagicMock()
    codec.device = "cpu"
    codec.to.return_value = codec
    codec.eval.return_value = codec
    codec.encode_code.return_value = torch.zeros((1, 1, 10), dtype=torch.long)
    codec.decode_code.return_value = torch.zeros((1, 1, 4800))
    return codec

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    def encode_side_effect(text, **kwargs):
        if "<|TEXT_REPLACE|>" in text and "<|SPEECH_REPLACE|>" in text:
            return [10, 100, 20, 200]
        return [1, 2, 3]
    tokenizer.encode.side_effect = encode_side_effect
    tokenizer.decode.return_value = "<|speech_1|><|speech_2|><|speech_3|>"
    def convert_side_effect(token):
        mapping = {"<|TEXT_REPLACE|>": 100, "<|SPEECH_REPLACE|>": 200, "<|SPEECH_GENERATION_START|>": 300, "<|TEXT_PROMPT_START|>": 400, "<|TEXT_PROMPT_END|>": 500, "<|SPEECH_GENERATION_END|>": 600}
        return mapping.get(token, 999)
    tokenizer.convert_tokens_to_ids.side_effect = convert_side_effect
    return tokenizer

@pytest.fixture
def mock_backbone():
    backbone = MagicMock()
    backbone.device = "cpu"
    backbone.to.return_value = backbone
    backbone.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    return backbone

@patch('huggingface_hub.hf_hub_download')
@patch('neucodec.DistillNeuCodec.from_pretrained')
@patch('transformers.AutoTokenizer.from_pretrained')
@patch('transformers.AutoModelForCausalLM.from_pretrained')
def test_viennu_tts_torch_init(mock_model_fn, mock_tok_fn, mock_codec_fn, mock_hf, mock_codec, mock_tokenizer, mock_backbone):
    mock_model_fn.return_value = mock_backbone
    mock_tok_fn.return_value = mock_tokenizer
    mock_codec_fn.return_value = mock_codec

    tts = VieNeuTTS(backbone_repo="some/repo", backbone_device="cpu")

    with patch('vieneu.standard.phonemize_with_dict', return_value="phonemes"):
        audio = tts.infer("Xin chào", ref_codes=[1, 2, 3], ref_text="chào")
        assert isinstance(audio, np.ndarray)

@patch('requests.post')
@patch('neucodec.DistillNeuCodec.from_pretrained')
def test_remote_vieneu_tts(mock_codec_fn, mock_post, mock_codec):
    mock_codec_fn.return_value = mock_codec
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "<|speech_1|><|speech_2|>"}}]}
    mock_post.return_value = mock_response
    tts = RemoteVieNeuTTS(api_base="http://fake-api")
    audio = tts.infer("Xin chào", ref_codes=[1, 2, 3], ref_text="chào")
    assert isinstance(audio, np.ndarray)
