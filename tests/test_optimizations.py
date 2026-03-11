import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Thêm src vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from vieneu_utils.phonemize_text import phonemize_batch, phonemize_with_dict

def test_base_ref_phoneme_cache():
    # Only import if base exists, otherwise skip or mock
    try:
        from vieneu.base import BaseVieneuTTS
    except ImportError:
        pytest.skip("BaseVieneuTTS not found")

    class MockTTS(BaseVieneuTTS):
        def infer(self, text, **kwargs):
            return None
        def infer_batch(self, texts, **kwargs):
            return [self.infer(t, **kwargs) for t in texts]

    tts = MockTTS()
    ref_text = "Giọng đọc mẫu số 1"

    with patch("vieneu_utils.phonemize_text.phonemize_with_dict") as mock_phonemize:
        mock_phonemize.return_value = "cached-phonemes"

        p1 = tts.get_ref_phonemes(ref_text)
        p2 = tts.get_ref_phonemes(ref_text)

        assert p1 == p2
        assert mock_phonemize.call_count == 1
        assert ref_text in tts._ref_phoneme_cache
