import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Thêm src vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'src')))

from vieneu_utils.phonemize_text import phonemize_batch, phonemize_with_dict

def test_phonemize_batch_deduplication():
    # Use 3 texts with overlapping words and English segments
    texts = [
        "Cái Bàn <en>world</en>",
        "Cái Bàn <en>world</en>",
        "Cái Ghế <en>world</en>"
    ]

    # Patch the actual espeak phonemize call
    with patch("vieneu_utils.phonemize_text.phonemize") as mock_phonemize:
        # All unique unknown words (bàn, cái, ghế, world) are sent in one batch.
        # sorted(list({'bàn', 'cái', 'ghế', 'world'})) -> ['bàn', 'cái', 'ghế', 'world']
        mock_phonemize.return_value = ["ban", "kai", "ge", "w-o-r-l-d"]

        # Use an empty custom dict (and no system dictionary) 
        # to ensure all words are treated as unknown fallback.
        results = phonemize_batch(texts, phoneme_dict={})

        # Verify deduplication: 1 single call to espeak backend
        assert mock_phonemize.call_count == 1

        # Check if all unique unknown words were sent in that one call
        call_args = mock_phonemize.call_args[0][0]
        assert len(call_args) == 4
        
        # Convert list elements to lower case for insensitive comparison
        call_args_lower = [w.lower() for w in call_args]
        assert "world" in call_args_lower
        assert "cái" in call_args_lower

def test_phonemize_with_dict_caching():
    from vieneu_utils.phonemize_text import _phonemize_with_dict_cached
    text = "Câu này sẽ được cache"

    # Clear cache before test
    _phonemize_with_dict_cached.cache_clear()

    with patch("vieneu_utils.phonemize_text.phonemize_batch") as mock_batch:
        mock_batch.return_value = ["p-h-o-n-e-m-e-s"]

        # First call
        res1 = phonemize_with_dict(text)
        # Second call
        res2 = phonemize_with_dict(text)

        assert res1 == res2
        # Should only be called once due to LRU cache
        assert mock_batch.call_count == 1

def test_base_ref_phoneme_cache():
    # Only import if base exists, otherwise skip or mock
    try:
        from vieneu.base import BaseVieneuTTS
    except ImportError:
        pytest.skip("BaseVieneuTTS not found")

    class MockTTS(BaseVieneuTTS):
        def infer(self, text, **kwargs):
            return None

    tts = MockTTS()
    ref_text = "Giọng đọc mẫu số 1"

    with patch("vieneu_utils.phonemize_text.phonemize_with_dict") as mock_phonemize:
        mock_phonemize.return_value = "cached-phonemes"

        p1 = tts.get_ref_phonemes(ref_text)
        p2 = tts.get_ref_phonemes(ref_text)

        assert p1 == p2
        assert mock_phonemize.call_count == 1
        assert ref_text in tts._ref_phoneme_cache
