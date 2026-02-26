import pytest
from vieneu_utils.normalize_text import VietnameseTTSNormalizer

@pytest.fixture
def normalizer():
    return VietnameseTTSNormalizer()

test_cases = [
    # ─── 1. SỐ THÔNG THƯỜNG ──────────────────
    ("0", "không"),
    ("1", "một"),
    ("10", "mười"),
    ("11", "mười một"),
    ("21", "hai mươi mốt"),
    ("100", "một trăm"),
    ("1000", "một nghìn"),
    ("1001", "một nghìn không trăm lẻ một"),
    ("1234567", "một triệu hai trăm ba mươi bốn nghìn năm trăm sáu mươi bảy"),

    # ─── 2. SỐ THẬP PHÂN ─────────────────────
    ("1.000", "một nghìn"),
    ("3,14", "ba phẩy mười bốn"),
    ("1.3", "một chấm ba"),

    # ─── 3. SỐ ĐIỆN THOẠI ────────────────────
    ("0912345678", "không chín một hai ba bốn năm sáu bảy tám"),
    ("+84912345678", "cộng tám bốn chín một hai ba bốn năm sáu bảy tám"),

    # ─── 4. SỐ THỨ TỰ ────────────────────────
    ("thứ 1", "thứ nhất"),
    ("thứ 4", "thứ tư"),
    ("hạng 1", "hạng nhất"),

    # ─── 5. PHÉP NHÂN ────────────────────────
    ("3 x 4", "ba nhân bốn"),

    # ─── 6. NGÀY THÁNG ───────────────────────
    ("21/02/2025", "ngày hai mươi mốt tháng hai năm hai nghìn không trăm hai mươi lăm"),
    ("01/12", "ngày một tháng mười hai"),
    ("02/2025", "tháng hai năm hai nghìn không trăm hai mươi lăm"),

    # ─── 7. THỜI GIAN ────────────────────────
    ("14h30", "mười bốn giờ ba mươi phút"),
    ("8h05", "tám giờ không năm phút"),
    ("23:59", "hai mươi ba giờ năm mươi chín phút"),
    ("12:00:00", "mười hai giờ không phút không giây"),

    # ─── 8. TIỀN TỆ & PHẦN TRĂM ──────────────
    ("100$", "một trăm đô la Mỹ"),
    ("500 VND", "năm trăm đồng"),
    ("75%", "bảy mươi lăm phần trăm"),
    ("370 tỷ USD", "ba trăm bảy mươi tỷ đô la Mỹ"),

    # ─── 9. ĐƠN VỊ ĐO LƯỜNG ──────────────────
    ("50km", "năm mươi ki lô mét"),
    ("100m", "một trăm mét"),
    ("75kg", "bảy mươi lăm ki lô gam"),
    ("10ha", "mười héc ta"),
    ("50m2", "năm mươi mét vuông"),

    # ─── 10. SỐ LA MÃ ────────────────────────
    ("Thế kỷ XXI", "thế kỷ hai mươi mốt"),
    ("Chương IV", "chương bốn"),

    # ─── 11. CHỮ CÁI ─────────────────────────
    ("chữ B", "chữ bê"),
    ("ký tự 'C'", "ký tự xê"),

    # ─── 12. VIẾT TẮT TIẾNG VIỆT ──────────────
    ("UBND", "uỷ ban nhân dân"),
    ("TP.HCM", "thành phố hồ chí minh"),
    ("CSGT", "cảnh sát giao thông"),

    # ─── 13. EN TAG ──────────────────────────
    ("<en>Hello</en>", "<en>Hello</en>"),
    ("Xin chào <en>Good morning</en>", "xin chào <en>Good morning</en>"),

    # ─── 14. DẤU CÂU & KÝ TỰ ─────────────────
    ("A & B", "a và bê"),
    ("A + B", "a cộng bê"),
    ("#1", "thăng một"),

    # ─── 15. HỖN HỢP ─────────────────────────
    ("Ngày 21/02/2025 lúc 14h30, giá vàng đạt 100$ tại TPHCM",
     "ngày hai mươi mốt tháng hai năm hai nghìn không trăm hai mươi lăm lúc mười bốn giờ ba mươi phút, giá vàng đạt một trăm đô la Mỹ tại thành phố hồ chí minh"),
]

@pytest.mark.parametrize("input_text, expected", test_cases)
def test_normalization(normalizer, input_text, expected):
    actual = normalizer.normalize(input_text)
    # Basic cleanup for comparison
    actual_clean = " ".join(actual.split()).lower()
    expected_clean = " ".join(expected.split()).lower()
    assert actual_clean == expected_clean
