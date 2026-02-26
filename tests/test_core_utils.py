import pytest
import numpy as np
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

def test_split_text_into_chunks_basic():
    text = "Đây là một câu ngắn. Đây là một câu khác."
    chunks = split_text_into_chunks(text, max_chars=30)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 30

def test_split_text_into_chunks_long_sentence():
    # A single very long sentence without punctuation
    text = "Một chuỗi cực kỳ dài không có dấu câu nào cả và nó phải bị cắt ra theo từ để đảm bảo độ dài tối đa của mỗi đoạn không vượt quá giới hạn cho phép của hệ thống tts"
    max_chars = 20
    chunks = split_text_into_chunks(text, max_chars=max_chars)
    for chunk in chunks:
        assert len(chunk) <= max_chars
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

def test_join_audio_chunks_basic():
    chunks = [np.ones(100), np.zeros(100)]
    result = join_audio_chunks(chunks, sr=16000)
    assert len(result) == 200
    assert np.all(result[:100] == 1)
    assert np.all(result[100:] == 0)

def test_join_audio_chunks_silence():
    chunks = [np.ones(100), np.ones(100)]
    # 0.01s silence @ 16000Hz = 160 samples
    result = join_audio_chunks(chunks, sr=16000, silence_p=0.01)
    assert len(result) == 200 + 160
    assert np.all(result[100:260] == 0)

def test_join_audio_chunks_crossfade():
    chunks = [np.ones(100), np.zeros(100)]
    # 10 samples crossfade
    result = join_audio_chunks(chunks, sr=16000, crossfade_p=10/16000)
    assert len(result) == 190
    # Overlap at samples 90-100
    # Values should be between 0 and 1
    assert np.all(result[90:100] >= 0)
    assert np.all(result[90:100] <= 1)
