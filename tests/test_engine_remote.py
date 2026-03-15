import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from vieneu.remote import RemoteVieNeuTTS

@pytest.fixture
def remote_tts():
    with patch("vieneu.base.hf_hub_download") as mock_hf:
        mock_hf.return_value = None  # Mock voice loading
        return RemoteVieNeuTTS(api_base="http://fake-api:23333/v1", model_name="fake-model")

def test_remote_format_prompt(remote_tts):
    ref_codes = [1, 2, 3]
    ref_text = "Chào bạn"
    input_text = "Thế giới"

    with patch.object(remote_tts, "get_ref_phonemes", return_value="ch-ao b-an"):
        with patch("vieneu_utils.phonemize_text.phonemize_with_dict", return_value="th-e g-io-i"):
            prompt = remote_tts._format_prompt(ref_codes, ref_text, input_text)
            assert "<|TEXT_PROMPT_START|>ch-ao b-an th-e g-io-i<|TEXT_PROMPT_END|>" in prompt
            assert "assistant:<|SPEECH_GENERATION_START|><|speech_1|><|speech_2|><|speech_3|>" in prompt

@patch("requests.post")
def test_remote_infer_single_chunk(mock_post, remote_tts):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "<|speech_100|><|speech_101|>"}}]}
    mock_post.return_value = mock_response

    remote_tts.codec = MagicMock()
    remote_tts.codec.device = torch.device("cpu")
    remote_tts.codec.decode_code.return_value = np.zeros((1, 1, 1000))

    with patch.object(remote_tts, "_resolve_ref_voice", return_value=([1, 2], "ref")):
        audio = remote_tts.infer("Xin chào")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000

def test_remote_parallel_chunking():
    with patch("vieneu.standard.DistillNeuCodec.from_pretrained"):
        tts = RemoteVieNeuTTS(api_base="http://mock", model_name="mock")
        with patch("vieneu.remote.split_text_into_chunks", return_value=["chunk1", "chunk2"]), \
             patch.object(RemoteVieNeuTTS, 'infer_async', return_value=np.zeros(2000)) as mock_async, \
             patch.object(RemoteVieNeuTTS, '_resolve_ref_voice', return_value=(torch.zeros(10), "ref")):
            tts.infer("Long text")
            mock_async.assert_called_once()

@pytest.mark.asyncio
async def test_remote_infer_async_chunk(remote_tts):
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    
    class AsyncContextManager:
        async def __aenter__(self): return mock_resp
        async def __aexit__(self, *args): pass
    mock_session.post.return_value = AsyncContextManager()
    mock_resp.raise_for_status = MagicMock()

    async def mock_json(): return {"choices": [{"message": {"content": "<|speech_200|>"}}]}
    mock_resp.json.side_effect = mock_json

    remote_tts.codec = MagicMock()
    remote_tts.codec.device = torch.device("cpu")
    remote_tts.codec.decode_code.return_value = np.zeros((1, 1, 500))

    audio = await remote_tts._infer_chunk_async(
        mock_session, "chunk", [1], "ref_text", 1.0, 50,
        ref_phonemes="ref", chunk_phonemes="chunk"
    )
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 500
