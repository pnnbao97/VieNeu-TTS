import pytest
import os
import sys
from unittest.mock import patch

# Thêm src vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

from vieneu_utils.phonemize_text import phonemize_batch, phonemize_with_dict


def test_phonemize_batch_deduplication():
    # Use 3 texts with overlapping words and English segments
    texts = ["Cái Bàn <en>world</en>", "Cái Bàn <en>world</en>", "Cái Ghế <en>world</en>"]

    # Patch the actual espeak phonemize call
    with patch("vieneu_utils.phonemize_text.phonemize") as mock_phonemize:
        # Call 1: force_espeak (EN từ <en> tag) → ['world']
        # Call 2: global_unknown VI accented  → ['bàn', 'cái', 'ghế']
        mock_phonemize.side_effect = [
            ["w-o-r-l-d"],  # Result for force_espeak EN words
            ["ban", "kai", "ge"],  # Result for VI unknown words
        ]

        results = phonemize_batch(texts, phoneme_dict={})

        assert mock_phonemize.call_count == 2

        all_calls = mock_phonemize.call_args_list

        # Call 1 = force_espeak (en-us), Call 2 = vi unknown
        en_call_words = all_calls[0][0][0]
        vi_call_words = all_calls[1][0][0]

        assert len(en_call_words) == 1  # chỉ 'world'
        assert len(vi_call_words) == 3  # 'bàn', 'cái', 'ghế'

        assert "world" in [w.lower() for w in en_call_words]
        assert "cái" in [w.lower() for w in vi_call_words]


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
