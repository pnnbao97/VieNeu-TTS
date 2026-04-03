import pytest
import numpy as np
import torch
import sys
from unittest.mock import patch, MagicMock, AsyncMock
from vieneu.remote import RemoteVieNeuTTS

@pytest.fixture
def remote_tts():
    with patch("vieneu.base.hf_hub_download") as mock_hf:
        mock_hf.return_value = None  # Mock voice loading
        # Mock codec loading in BaseVieneuTTS
        with patch("vieneu.base.BaseVieneuTTS._load_codec"):
            tts = RemoteVieNeuTTS(api_base="https://fake-api:23333/v1", model_name="fake-model")
            tts.codec = MagicMock()
            tts.codec.device = "cpu"
            tts.codec.sample_rate = 24000
            tts.codec.decode_code.return_value = np.zeros((1, 1, 1000))
            return tts

def test_remote_format_prompt(remote_tts):
    ref_codes = [1, 2, 3]
    ref_text = "Chào bạn"
    input_text = "Thế giới"

    with patch.object(remote_tts, "get_ref_phonemes", return_value="ch-ao b-an"):
        with patch("sea_g2p.Normalizer.normalize", side_effect=lambda x: x):
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

    with patch.object(remote_tts, "_resolve_ref_voice", return_value=([1, 2], "ref")):
        audio = remote_tts.infer("Xin chào")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000

@pytest.mark.asyncio
async def test_remote_infer_async_chunk(remote_tts):
    # Setup mock session and response
    mock_session = MagicMock()
    # Mock aiohttp in sys.modules
    with patch.dict(sys.modules, {"aiohttp": MagicMock()}):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = AsyncMock(return_value={"choices": [{"message": {"content": "<|speech_200|>"}}]})

        # Mock the 'async with session.post(...)' pattern
        # calling session.post() returns an object whose __aenter__ returns mock_resp
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        audio = await remote_tts._infer_chunk_async(
            mock_session, "chunk", [1], "ref_text", 1.0, 50
        )
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000
