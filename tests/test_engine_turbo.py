import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, patch

# Mock heavy modules before importing turbo
mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
sys.modules["torch"] = mock_torch
sys.modules["torch.backends"] = mock_torch.backends
sys.modules["torch.backends.mps"] = mock_torch.backends.mps
sys.modules["llama_cpp"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["lmdeploy"] = MagicMock()

from vieneu.turbo import TurboVieNeuTTS, TurboGPUVieNeuTTS

@pytest.fixture
def mock_onnx_session():
    session = MagicMock()
    # Mock return value for decoder/encoder run (returns a list of outputs)
    session.run.return_value = [np.zeros((1, 1, 48000), dtype=np.float32)]
    return session

@pytest.fixture
def mock_llama_instance():
    llama = MagicMock()
    llama.return_value = {
        "choices": [{"text": "<|speech_1|><|speech_2|><|speech_3|>"}]
    }
    return llama

@patch("onnxruntime.InferenceSession")
@patch("llama_cpp.Llama")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
def test_turbo_gguf_init(mock_hf, mock_llama, mock_ort):
    tts = TurboVieNeuTTS(backbone_repo="dummy", device="cpu")
    assert tts.backbone is not None
    assert tts.decoder_sess is not None
    mock_llama.assert_called_once()

@patch("onnxruntime.InferenceSession")
@patch("llama_cpp.Llama")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
def test_turbo_gguf_infer(mock_hf, mock_llama, mock_ort, mock_onnx_session, mock_llama_instance):
    mock_ort.return_value = mock_onnx_session
    mock_llama.return_value = mock_llama_instance
    
    tts = TurboVieNeuTTS(backbone_repo="dummy", device="cpu")
    tts._preset_voices = {"test": {"codes": np.zeros(128), "text": "test"}}
    
    with patch("vieneu_utils.phonemize_text.phonemize_text", return_value="p-h-o-n-e-m-e-s"):
        audio = tts.infer("Xin chào", voice={"codes": np.zeros(128), "text": "test"})
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        mock_llama_instance.assert_called()

@patch("onnxruntime.InferenceSession")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_turbo_gpu_standard_init(mock_model, mock_tokenizer, mock_hf, mock_ort):
    # Mock model and tokenizer
    mock_tokenizer.return_value = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model.return_value = mock_model_instance

    tts = TurboGPUVieNeuTTS(backbone_repo="dummy", device="cuda", backend="standard")
    assert tts.backend == "standard"
    assert tts.backbone is not None
    assert tts.device == "cuda"

@patch("onnxruntime.InferenceSession")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
@patch("lmdeploy.pipeline")
def test_turbo_gpu_lmdeploy_init(mock_pipeline, mock_hf, mock_ort):
    mock_pipeline_instance = MagicMock()
    mock_pipeline.return_value = mock_pipeline_instance

    tts = TurboGPUVieNeuTTS(backbone_repo="dummy", device="cuda", backend="lmdeploy")
    assert tts.backend == "lmdeploy"
    assert tts.backbone is not None

@patch("onnxruntime.InferenceSession")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_turbo_gpu_infer(mock_model, mock_tokenizer, mock_hf, mock_ort, mock_onnx_session):
    mock_ort.return_value = mock_onnx_session

    # Mock standard Transformers path
    mock_tokenizer_instance = MagicMock()
    mock_token_tensor = MagicMock()
    mock_token_tensor.to.return_value = mock_token_tensor
    mock_tokenizer_instance.return_value = {"input_ids": mock_token_tensor}
    mock_tokenizer_instance.decode.return_value = "<|speech_100|><|speech_101|>"
    mock_tokenizer.return_value = mock_tokenizer_instance

    mock_model_instance = MagicMock()
    mock_gen_output = MagicMock()
    mock_gen_output.cpu.return_value = mock_gen_output
    # Mock __getitem__ for inputs['input_ids'].shape[-1]
    mock_token_tensor.shape = [1, 5]

    mock_model_instance.generate.return_value = mock_gen_output
    mock_model_instance.to.return_value = mock_model_instance
    mock_model.return_value = mock_model_instance

    tts = TurboGPUVieNeuTTS(backbone_repo="dummy", device="cpu", backend="standard")
    tts._preset_voices = {"test": {"codes": np.zeros(128), "text": "test"}}

    with patch("vieneu_utils.phonemize_text.phonemize_text", return_value="p-h-o-n-e-m-e-s"):
        audio = tts.infer("Xin chào", voice={"codes": np.zeros(128), "text": "test"})
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

@patch("onnxruntime.InferenceSession")
@patch("llama_cpp.Llama")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
def test_turbo_voice_cloning_encode(mock_hf, mock_llama, mock_ort, mock_onnx_session):
    mock_ort.return_value = mock_onnx_session
    # Mock encoder return
    mock_onnx_session.run.return_value = [np.zeros((1, 128), dtype=np.float32)]
    
    tts = TurboVieNeuTTS(backbone_repo="dummy", device="cpu")
    with patch("librosa.load", return_value=(np.zeros(24000), 24000)):
        emb = tts.encode_reference("dummy.wav")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1, 128)

@patch("onnxruntime.InferenceSession")
@patch("llama_cpp.Llama")
@patch("huggingface_hub.hf_hub_download", return_value="dummy_path")
def test_turbo_array_truth_value_fix(mock_hf, mock_llama, mock_ort, mock_onnx_session, mock_llama_instance):
    """Verify that passing numpy arrays doesn't cause 'truth value of an array is ambiguous' error."""
    mock_ort.return_value = mock_onnx_session
    mock_llama.return_value = mock_llama_instance

    tts = TurboVieNeuTTS(backbone_repo="dummy", device="cpu")

    with patch("vieneu_utils.phonemize_text.phonemize_text", return_value="p-h-o-n-e-m-e-s"):
        # Test with numpy array for ref_codes
        audio = tts.infer("Xin chào", ref_codes=np.zeros(128))
        assert isinstance(audio, np.ndarray)
