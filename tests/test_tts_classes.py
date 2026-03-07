import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from vieneu.standard import VieNeuTTS
from vieneu.remote import RemoteVieNeuTTS


@pytest.fixture
def mock_codec():
    codec = MagicMock()
    codec.sample_rate = 24000
    codec.device = "cpu"
    # Mock encode_code to return a tensor of codes
    codec.encode_code.return_value = torch.zeros((1, 1, 10), dtype=torch.long)
    # Mock decode_code to return a tensor of audio
    codec.decode_code.return_value = torch.zeros((1, 1, 4800), dtype=torch.float32)
    return codec


@pytest.fixture
def mock_backbone():
    backbone = MagicMock()
    backbone.device = torch.device("cpu")
    backbone.to.return_value = backbone
    # Mock generate for torch backend
    backbone.generate.return_value = torch.tensor([[0, 1, 2, 3]])  # Dummy output tokens
    # Mock llama-cpp call
    backbone.return_value = {"choices": [{"text": "<|speech_1|><|speech_2|>"}]}
    return backbone


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    # Map special tokens to specific IDs
    token_to_id = {
        "<|SPEECH_REPLACE|>": 1001,
        "<|SPEECH_GENERATION_START|>": 1002,
        "<|TEXT_REPLACE|>": 1003,
        "<|TEXT_PROMPT_START|>": 1004,
        "<|TEXT_PROMPT_END|>": 1005,
        "<|SPEECH_GENERATION_END|>": 1006,
    }
    tokenizer.convert_tokens_to_ids.side_effect = lambda x: token_to_id.get(x, 999)

    # Mock encode to include the TEXT_REPLACE and SPEECH_REPLACE tokens when expected
    def mocked_encode(text, **kwargs):
        if "TEXT_REPLACE" in text:
            return [10, 1003, 11, 1001]
        return [1, 2, 3]

    tokenizer.encode.side_effect = mocked_encode
    tokenizer.decode.return_value = "<|speech_1|><|speech_2|>"
    return tokenizer


def test_vieneu_tts_init(mock_codec, mock_backbone, mock_tokenizer):
    with (
        patch("vieneu.standard.NeuCodec.from_pretrained", return_value=mock_codec),
        patch("vieneu.standard.DistillNeuCodec.from_pretrained", return_value=mock_codec),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_backbone),
    ):
        tts = VieNeuTTS(backbone_repo="some/repo", backbone_device="cpu")
        assert tts.backbone is not None
        assert tts.codec is not None


def test_vieneu_tts_infer(mock_codec, mock_backbone, mock_tokenizer):
    with (
        patch("vieneu.standard.NeuCodec.from_pretrained", return_value=mock_codec),
        patch("vieneu.standard.DistillNeuCodec.from_pretrained", return_value=mock_codec),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_backbone),
    ):
        tts = VieNeuTTS(backbone_repo="some/repo", backbone_device="cpu")

        with patch("vieneu.standard.phonemize_with_dict", return_value="phonemes"):
            audio = tts.infer("Xin chào", ref_codes=[1, 2, 3], ref_text="Chào")
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
            assert len(audio) == 4800


def test_vieneu_tts_infer_with_voice_preset(mock_codec, mock_backbone, mock_tokenizer):
    with (
        patch("vieneu.standard.NeuCodec.from_pretrained", return_value=mock_codec),
        patch("vieneu.standard.DistillNeuCodec.from_pretrained", return_value=mock_codec),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_backbone),
    ):
        tts = VieNeuTTS(backbone_repo="some/repo", backbone_device="cpu")
        tts._preset_voices = {"test_voice": {"codes": [1, 2, 3], "text": "test"}}

        with patch("vieneu.standard.phonemize_with_dict", return_value="phonemes"):
            audio = tts.infer("Xin chào", voice=tts.get_preset_voice("test_voice"))
            assert isinstance(audio, np.ndarray)
            assert len(audio) == 4800


def test_remote_vieneu_tts_infer(mock_codec):
    with patch("vieneu.standard.DistillNeuCodec.from_pretrained", return_value=mock_codec):
        tts = RemoteVieNeuTTS(api_base="http://mock-api", model_name="mock-model")

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "<|speech_1|><|speech_2|>"}}]}
        mock_response.raise_for_status = MagicMock()

        with (
            patch("requests.post", return_value=mock_response),
            patch("vieneu.remote.phonemize_with_dict", return_value="phonemes"),
        ):
            audio = tts.infer("Xin chào", ref_codes=[1, 2, 3], ref_text="Chào")
            assert isinstance(audio, np.ndarray)
            assert len(audio) == 4800


def test_vieneu_tts_streaming(mock_codec, mock_backbone, mock_tokenizer):
    with (
        patch("vieneu.standard.NeuCodec.from_pretrained", return_value=mock_codec),
        patch("vieneu.standard.DistillNeuCodec.from_pretrained", return_value=mock_codec),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_backbone),
    ):
        tts = VieNeuTTS(backbone_repo="some/repo", backbone_device="cpu")

        with patch("vieneu.standard.phonemize_with_dict", return_value="phonemes"):
            stream = tts.infer_stream("Xin chào", ref_codes=[1, 2, 3], ref_text="Chào")
            chunks = list(stream)
            assert len(chunks) > 0
            assert isinstance(chunks[0], np.ndarray)
