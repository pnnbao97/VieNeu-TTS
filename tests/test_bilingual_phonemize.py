import unittest
import os
import sys

# Thêm src vào path để import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vieneu_utils.phonemize_text import phonemize_text

class TestBilingualPhonemize(unittest.TestCase):
    """
    Test suite kiểm tra logic phonemize song ngữ và cơ chế lan truyền (propagation).
    """

    def assert_propagation(self, natural_text, tagged_text):
        """Kiểm tra câu tự nhiên phải cho ra phoneme giống hệt câu có tag."""
        p_nat = phonemize_text(natural_text)
        p_tag = phonemize_text(tagged_text)
        self.assertEqual(p_nat, p_tag, f"\nFAIL:\n  Nat: {p_nat}\n  Tag: {p_tag}")

    def test_bridge_propagation(self):
        """Kiểm tra từ 'common' đi theo mỏ neo English."""
        # 'go to the' được kéo sang EN bởi 'market'
        self.assert_propagation(
            "Tôi muốn go to the market", 
            "Tôi muốn <en>go to the market</en>"
        )

    def test_punctuation_boundary(self):
        """Dấu câu (dấu chấm) phải ngăn cách ngữ cảnh."""
        # Chữ 'to' ở túi to (VI) không được bị 'market' (EN) ở câu trước kéo đi
        self.assert_propagation(
            "go to the market. Mua một cái túi to.", 
            "<en>go to the market</en>. Mua một cái túi to."
        )

    def test_closest_neighbor_priority(self):
        """Từ common đi theo mỏ neo gần nhất."""
        # 'to' kẹp giữa 'túi' (VI) và 'market' (EN), nhưng sát 'túi' hơn
        self.assert_propagation("cái túi to market", "cái túi to <en>market</en>")
        
        # 'go to' kẹp giữa 'muốn' (VI) và 'market' (EN), khoảng cách bằng nhau -> ưu tiên bên phải (EN)
        self.assert_propagation("muốn go to market", "muốn <en>go to market</en>")

    def test_common_word_disambiguation(self):
        """Kiểm tra các cặp từ đa nghĩa phổ biến."""
        # me (EN) vs me (VI)
        self.assert_propagation("give it to me", "<en>give it to me</en>")
        self.assert_propagation("ăn quả me", "ăn quả me")
        
        # no (EN) vs no (VI)
        self.assert_propagation("say no more", "<en>say no more</en>")
        self.assert_propagation("ăn cho no", "ăn cho no")
        
        # can (EN) vs can (VI)
        self.assert_propagation("I can do it", "<en>I can do it</en>")
        self.assert_propagation("can ngăn", "can ngăn")

if __name__ == '__main__':
    unittest.main()
