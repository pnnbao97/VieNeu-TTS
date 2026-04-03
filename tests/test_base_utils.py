import unittest
import numpy as np
import sys
from unittest.mock import MagicMock, patch

# Mock torch BEFORE importing utilities that use it
sys.modules["torch"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()
sys.modules["torch.backends.mps"] = MagicMock()

from vieneu.utils import normalize_device
from vieneu.base import BaseVieneuTTS

class DummyTTS(BaseVieneuTTS):
    def infer(self, text, **kwargs):
        return np.zeros(100)
    def infer_batch(self, texts, **kwargs):
        return [np.zeros(100) for _ in texts]

class TestBaseUtils(unittest.TestCase):
    def test_normalize_device(self):
        self.assertEqual(normalize_device("cuda:0"), "cuda")
        self.assertEqual(normalize_device("GPU"), "cuda")
        self.assertEqual(normalize_device("cpu"), "cpu")
        self.assertEqual(normalize_device("xpu"), "xpu")

        # Mocking torch.backends.mps.is_available needs to handle the fact that
        # torch might not be imported yet or might be missing
        with patch("torch.backends.mps.is_available", return_value=True, create=True):
            self.assertEqual(normalize_device("mps"), "mps")
        with patch("torch.backends.mps.is_available", return_value=False, create=True):
            self.assertEqual(normalize_device("mps"), "cpu")

    def test_to_list(self):
        tts = DummyTTS()
        # Test list
        self.assertEqual(tts.to_list([1, 2, 3]), [1, 2, 3])
        # Test numpy
        self.assertEqual(tts.to_list(np.array([4, 5])), [4, 5])
        # Test multi-dim numpy
        self.assertEqual(tts.to_list(np.array([[6], [7]])), [6, 7])

        # Test torch if available or mocked
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value.tolist.return_value = [8, 9]
        with patch("torch.Tensor", (MagicMock,), create=True):
             # We need to simulate isinstance check. This is tricky with MagicMock.
             # Let's test the fallback hasattr(codes, "tolist")
             mock_obj = MagicMock()
             mock_obj.flatten.return_value.tolist.return_value = [10, 11]
             self.assertEqual(tts.to_list(mock_obj), [10, 11])

    def test_base_streaming_config(self):
        tts = DummyTTS()
        self.assertTrue(hasattr(tts, "streaming_overlap_frames"))
        self.assertEqual(tts.streaming_lookforward, 5)

if __name__ == "__main__":
    unittest.main()
