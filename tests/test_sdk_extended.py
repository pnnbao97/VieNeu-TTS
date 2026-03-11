import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from vieneu.factory import Vieneu
from vieneu.standard import VieNeuTTS
from vieneu.remote import RemoteVieNeuTTS

@pytest.fixture
def mock_torch_backbone():
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("neucodec.NeuCodec.from_pretrained") as mock_codec, \
         patch("neucodec.DistillNeuCodec.from_pretrained") as mock_distill_codec:

        tokenizer = MagicMock()
        tokenizer.pad.return_value = {"input_ids": torch.zeros((2, 10), dtype=torch.long)}
        tokenizer.convert_tokens_to_ids.return_value = 100
        tokenizer.decode.return_value = "<|speech_1|><|speech_2|>"
        mock_tok.return_value = tokenizer

        model = MagicMock()
        model.device = torch.device("cpu")
        model.to.return_value = model
        model.generate.return_value = torch.zeros((2, 20), dtype=torch.long)
        mock_model.return_value = model

        codec = MagicMock()
        codec.device = torch.device("cpu")
        codec.sample_rate = 24000
        codec.decode_code.return_value = torch.zeros((1, 1, 4800))
        mock_codec.return_value = codec
        mock_distill_codec.return_value = codec

        yield {"tokenizer": tokenizer, "model": model, "codec": codec}


def test_base_encode_reference_device(mock_torch_backbone):
    """Test BaseVieneuTTS.encode_reference moves tensor to correct device."""
    from vieneu.standard import VieNeuTTS

    with patch("librosa.load", return_value=(np.zeros(16000), 16000)), \
         patch.object(VieNeuTTS, '_warmup_model'):
        tts = VieNeuTTS(backbone_repo="dummy", backbone_device="cpu")
        tts.codec.device = torch.device("cpu")

        with patch.object(torch.Tensor, 'to', wraps=torch.zeros(1).to) as mock_to:
            tts.encode_reference("dummy.wav")
            # Verify .to() was called to move the tensor to codec device
            mock_to.assert_called()

def test_standard_true_batching(mock_torch_backbone):
    """Test that VieNeuTTS.infer_batch uses true batch generation for Torch backend."""
    from vieneu.standard import VieNeuTTS

    with patch.object(VieNeuTTS, '_warmup_model'), \
         patch("vieneu.standard.phonemize_batch", return_value=["p1", "p2"]):
        tts = VieNeuTTS(backbone_repo="dummy", backbone_device="cpu")
        tts._is_quantized_model = False # Force torch backend

        texts = ["Text 1", "Text 2"]
        with patch.object(tts, '_decode', return_value=np.zeros(1000)):
            results = tts.infer_batch(texts, ref_codes=[1], ref_text="ref")

            assert len(results) == 2

def test_fast_infer_batch_phonemize_called(mock_torch_backbone):
    """Test that FastVieNeuTTS.infer_batch calls phonemize_batch."""
    from vieneu.fast import FastVieNeuTTS

    with patch("lmdeploy.pipeline") as mock_pipeline, \
         patch("lmdeploy.GenerationConfig"), \
         patch.object(FastVieNeuTTS, '_warmup_model'):

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.return_value = [MagicMock(text="codes")]

        tts = FastVieNeuTTS(backbone_device="cuda")

        texts = ["Text 1", "Text 2"]
        with patch("vieneu.fast.phonemize_batch", return_value=["p1", "p2"]) as mock_ph_batch, \
             patch.object(tts, '_decode', return_value=np.zeros(1000)):
            tts.infer_batch(texts, ref_codes=[1], ref_text="ref")
            mock_ph_batch.assert_called_once()
            # Verify LMDeploy pipeline was called
            mock_pipeline_instance.assert_called()

def test_remote_parallel_chunking():
    """Test that RemoteVieNeuTTS.infer uses parallel async processing for multi-chunks."""
    with patch("neucodec.DistillNeuCodec.from_pretrained"):
        tts = RemoteVieNeuTTS(api_base="http://mock", model_name="mock")

        # Force 2 chunks
        with patch("vieneu.remote.split_text_into_chunks", return_value=["chunk1", "chunk2"]), \
             patch.object(RemoteVieNeuTTS, 'infer_async', return_value=np.zeros(2000)) as mock_async, \
             patch.object(RemoteVieNeuTTS, '_resolve_ref_voice', return_value=(torch.zeros(10), "ref")):

            res = tts.infer("Long text that splits")
            assert mock_async.called
            # Should be called once via asyncio.run(self.infer_async(...))
            # which then parallelizes internal chunks
            mock_async.assert_called_once()

def test_lora_loading_logic(mock_torch_backbone):
    """Test LoRA adapter loading and unloading."""
    from vieneu.standard import VieNeuTTS

    with patch.object(VieNeuTTS, '_warmup_model'), \
         patch("peft.PeftModel.from_pretrained") as mock_peft:
        tts = VieNeuTTS(backbone_repo="dummy", backbone_device="cpu")

        # Load LoRA
        tts.load_lora_adapter("lora_repo")
        assert tts._lora_loaded is True
        mock_peft.assert_called_once()

        # Unload LoRA
        with patch.object(tts.backbone, 'unload', return_value=mock_torch_backbone["model"]):
            tts.unload_lora_adapter()
            assert tts._lora_loaded is False
