import numpy as np
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks


def test_split_text_into_chunks():
    text = "Đây là một câu ngắn. Đây là một câu dài hơn một chút để kiểm tra xem nó có bị chia ra không nếu chúng ta đặt giới hạn ký tự thấp."
    chunks = split_text_into_chunks(text, max_chars=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50


def test_split_text_paragraphs():
    text = "Đoạn 1.\n\nĐoạn 2."
    chunks = split_text_into_chunks(text, max_chars=100)
    assert len(chunks) == 2
    assert "Đoạn 1" in chunks[0]
    assert "Đoạn 2" in chunks[1]


def test_join_audio_chunks():
    sr = 16000
    chunk1 = np.ones(1600, dtype=np.float32)  # 0.1s
    chunk2 = np.ones(1600, dtype=np.float32) * 0.5

    # Simple join
    joined = join_audio_chunks([chunk1, chunk2], sr)
    assert len(joined) == 3200
    assert np.all(joined[:1600] == 1.0)
    assert np.all(joined[1600:] == 0.5)


def test_join_audio_chunks_with_silence():
    sr = 16000
    chunk1 = np.ones(1600, dtype=np.float32)
    chunk2 = np.ones(1600, dtype=np.float32)

    # Join with 0.1s silence (1600 samples)
    joined = join_audio_chunks([chunk1, chunk2], sr, silence_p=0.1)
    assert len(joined) == 1600 + 1600 + 1600
    assert np.all(joined[1600:3200] == 0.0)
