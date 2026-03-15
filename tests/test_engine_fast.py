import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from vieneu.fast import FastVieNeuTTS

def test_fast_infer_batch_phonemize_called():
    with patch("lmdeploy.pipeline") as mock_pipeline, \
         patch("lmdeploy.GenerationConfig"), \
         patch.object(FastVieNeuTTS, '_warmup_model'), \
         patch("vieneu.standard.DistillNeuCodec.from_pretrained"):

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.return_value = [MagicMock(text="codes")]

        tts = FastVieNeuTTS(backbone_device="cuda")

        texts = ["Text 1", "Text 2"]
        with patch("vieneu.fast.phonemize_batch", return_value=["p1", "p2"]) as mock_ph_batch, \
             patch.object(tts, '_decode', return_value=np.zeros(1000)):
            tts.infer_batch(texts, ref_codes=[1], ref_text="ref")
            mock_ph_batch.assert_called_once()
            mock_pipeline_instance.assert_called()
