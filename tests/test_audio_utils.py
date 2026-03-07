import numpy as np
from vieneu.utils import _linear_overlap_add
from vieneu_utils.core_utils import join_audio_chunks

def test_linear_overlap_add():
    # Create two overlapping frames
    frame_len = 100
    stride = 50
    frame1 = np.ones(frame_len, dtype=np.float32)
    frame2 = np.ones(frame_len, dtype=np.float32)

    frames = [frame1, frame2]
    out = _linear_overlap_add(frames, stride)

    # Total length should be stride * (len(frames) - 1) + frame_len = 50 * 1 + 100 = 150
    assert out.shape == (150,)

    # Check that it's not all zeros
    assert np.any(out != 0)

    # With all ones and linear OLA, the result should be close to 1.0 where it overlaps
    # (weight1 * 1 + weight2 * 1) / (weight1 + weight2) = 1.0
    assert np.allclose(out[50:100], 1.0)

def test_linear_overlap_add_empty():
    assert _linear_overlap_add([], 50).shape == (0,)

def test_join_audio_chunks_simple():
    chunks = [np.ones(100), np.zeros(100)]
    joined = join_audio_chunks(chunks, sr=16000)
    assert joined.shape == (200,)
    assert np.array_equal(joined[:100], np.ones(100))
    assert np.array_equal(joined[100:], np.zeros(100))

def test_join_audio_chunks_silence():
    chunks = [np.ones(100), np.ones(100)]
    sr = 16000
    silence_p = 0.1 # 0.1s * 16000 = 1600 samples
    joined = join_audio_chunks(chunks, sr=sr, silence_p=silence_p)
    assert joined.shape == (100 + 1600 + 100,)
    assert np.all(joined[100:1700] == 0)

def test_join_audio_chunks_crossfade():
    chunks = [np.ones(1000), np.zeros(1000)]
    sr = 16000
    crossfade_p = 0.01 # 0.01s * 16000 = 160 samples
    joined = join_audio_chunks(chunks, sr=sr, crossfade_p=crossfade_p)
    # Length should be 1000 + 1000 - 160 = 1840
    assert joined.shape == (1840,)
    # Check transition
    assert joined[0] == 1.0
    assert joined[-1] == 0.0
    # Mid-point of crossfade should be 0.5
    assert np.allclose(joined[1000 - 80], 0.5, atol=0.01)

def test_join_audio_chunks_empty():
    assert join_audio_chunks([], 16000).shape == (0,)

def test_join_audio_chunks_single():
    chunk = np.ones(100)
    assert np.array_equal(join_audio_chunks([chunk], 16000), chunk)
