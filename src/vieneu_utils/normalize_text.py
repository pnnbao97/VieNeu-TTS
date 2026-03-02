import re
import unicodedata
from .cleaner import clean_vietnamese_text

class VietnameseTTSNormalizer:
    """
    A text normalizer for Vietnamese Text-to-Speech systems.
    Converts numbers, dates, units, and special characters into readable Vietnamese text.
    All core logic is implemented in the cleaner module.
    """
    
    def __init__(self) -> None:
        pass
    
    def normalize(self, text: str) -> str:
        """Main normalization pipeline with EN tag protection."""
        if not text:
            return ""

        # Pre-normalization: Ensure NFC format for Vietnamese characters
        text = unicodedata.normalize('NFC', text)

        # Step 1: Detect and protect EN tags
        en_contents = []
        placeholder_pattern = "___PROTECTED_EN_TAG_{}___"
        
        def extract_en(match):
            en_contents.append(match.group(0))
            return placeholder_pattern.format(len(en_contents) - 1)
        
        text = re.sub(r'<en>.*?</en>', extract_en, text, flags=re.IGNORECASE)
        
        # Step 2: Core Normalization
        text = clean_vietnamese_text(text)
        
        # Final cleanup - preserve newlines
        text = text.lower()
        text = re.sub(r'[ \t\xA0]+', ' ', text).strip()
        
        # Step 3: Restore EN tags
        for idx, en_content in enumerate(en_contents):
            placeholder = placeholder_pattern.format(idx).lower()
            text = text.replace(placeholder, en_content + ' ')
        
        # Final whitespace cleanup - preserve newlines
        text = re.sub(r'[ \t\xA0]+', ' ', text).strip()
        
        return text

if __name__ == "__main__":
    normalizer = VietnameseTTSNormalizer()
    
    test_texts = [
        "Lễ kỷ niệm 70 năm Chiến thắng Điện Biên Phủ (07/5/1954 - 07/5/2024).",
        "Phiên bản 1.0.4, tốc độ 60km/h, nhiệt độ 30°C.",
        "Số điện thoại: +84 912.345.678"
    ]
    
    for t in test_texts:
        print(f"In:  {t}")
        print(f"Out: {normalizer.normalize(t)}\n")