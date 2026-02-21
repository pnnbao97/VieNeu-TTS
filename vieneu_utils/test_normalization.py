# -*- coding: utf-8 -*-
"""
Comprehensive test suite for VietnameseTTSNormalizer.
Tests all special cases across all normalization categories.
"""
import sys
import os
import io

# Ensure the parent directory is in sys.path so we can import vieneu_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from vieneu_utils.normalize_text import VietnameseTTSNormalizer

normalizer = VietnameseTTSNormalizer()

test_groups = {

    # ─── 1. SỐ THÔNG THƯỜNG ────────────────────────────────────────────────────
    "Số thông thường": [
        ("0",        "không"),
        ("1",        "một"),
        ("10",       "mười"),
        ("11",       "mười một"),
        ("21",       "hai mươi mốt"),
        ("100",      "một trăm"),
        ("1000",     "một nghìn"),
        ("1001",     "một nghìn không trăm lẻ một"),
        ("1000000",  "một triệu"),
        ("1234567",  "một triệu hai trăm ba mươi bốn nghìn năm trăm sáu mươi bảy"),
    ],

    # ─── 2. SỐ THẬP PHÂN / SỐ CÓ DẤU PHÂN CÁCH ──────────────────────────────
    "Số thập phân": [
        ("1.000",    "một nghìn"),
        ("1.000.000","một triệu"),
        ("3,14",     "ba phẩy mười bốn"),
    ],

    # ─── 3. SỐ ĐIỆN THOẠI ────────────────────────────────────────────────────
    "Số điện thoại": [
        ("0912345678",    "không chín một hai ba bốn năm sáu bảy tám"),
        ("+84912345678",  "cộng tám bốn chín một hai ba bốn năm sáu bảy tám"),
    ],

    # ─── 4. SỐ THỨ TỰ ─────────────────────────────────────────────────────────
    "Số thứ tự đặc biệt": [
        ("thứ 1",  "thứ nhất"),
        ("thứ 4",  "thứ tư"),
        ("thứ 5",  "thứ năm"),
        ("hạng 1", "hạng nhất"),
    ],

    # ─── 5. PHÉP NHÂN ─────────────────────────────────────────────────────────
    "Phép nhân": [
        ("3 x 4",  "ba nhân bốn"),
        ("10 x 20","mười nhân hai mươi"),
    ],

    # ─── 6. NGÀY THÁNG ─────────────────────────────────────────────────────────
    "Ngày tháng năm (đầy đủ)": [
        ("21/02/2025", "ngày hai mươi mốt tháng hai năm hai nghìn không trăm hai mươi lăm"),
        ("01-01-2024", "ngày một tháng một năm hai nghìn không trăm hai mươi bốn"),
        ("31.12.2023", "ngày ba mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi ba"),
    ],

    "Ngày tháng (ngắn)": [
        ("21/02", "ngày hai mươi mốt tháng hai"),
        ("01/12", "ngày một tháng mười hai"),
    ],

    "Tháng năm": [
        ("02/2025", "tháng hai năm hai nghìn không trăm hai mươi lăm"),
        ("12/2024", "tháng mười hai năm hai nghìn không trăm hai mươi bốn"),
    ],

    "Ngày tháng không hợp lệ (phải giữ nguyên)": [
        ("32/01", "ba mươi hai lẻ một"),
        ("01/13", "lẻ một mười ba"),
    ],

    # ─── 7. THỜI GIAN ─────────────────────────────────────────────────────────
    "Thời gian": [
        ("14h30",   "mười bốn giờ ba mươi phút"),
        ("8h05",    "tám giờ lẻ năm phút"),
        ("0h00",    "không giờ không phút"),
        ("23:59",   "hai mươi ba giờ năm mươi chín phút"),
        ("12:00:00","mười hai giờ không phút không giây"),
    ],

    # ─── 8. TIỀN TỆ ──────────────────────────────────────────────────────────
    "Tiền tệ": [
        ("100$",   "một trăm đô la"),
        ("$50",    "năm mươi đô la"),
        ("200 USD","hai trăm đô la"),
        ("500 VND","năm trăm đồng"),
        ("50 euro","năm mươi ơ rô"),
        ("1000đ",  "một nghìn đồng"),
        ("75%",    "bảy mươi lăm phần trăm"),
    ],

    # ─── 9. ĐƠN VỊ ĐO LƯỜNG ─────────────────────────────────────────────────
    "Đơn vị đo lường": [
        ("50km",  "năm mươi ki lô mét"),
        ("100m",  "một trăm mét"),
        ("30cm",  "ba mươi xen ti mét"),
        ("5mm",   "năm mi li mét"),
        ("75kg",  "bảy mươi lăm ki lô gam"),
        ("500g",  "năm trăm gam"),
        ("250ml", "hai trăm năm mươi mi li lít"),
        ("2l",    "hai lít"),
        ("10ha",  "mười héc ta"),
        ("50m2",  "năm mươi mét vuông"),
        ("20m3",  "hai mươi mét khối"),
        ("300.000km", "ba trăm nghìn ki lô mét"),
    ],

    # ─── 10. SỐ LA MÃ ────────────────────────────────────────────────────────
    "Số La Mã": [
        ("Thế kỷ XXI",  "thế kỷ hai mươi mốt"),
        ("Chương IV",   "chương bốn"),
        ("Hồi IX",      "hồi chín"),
        ("Phần III",    "phần ba"),
        ("Thế kỷ XX",   "thế kỷ hai mươi"),
    ],

    # ─── 11. CHỮ CÁI ─────────────────────────────────────────────────────────
    "Chữ cái": [
        ("ký tự A",     "ký tự ây"),
        ("chữ B",       "chữ bê"),
        ("ký tự 'C'",   "ký tự xê"),
        ("chữ cái Z",   "chữ cái dét"),
        ("kí tự w",     "kí tự vê kép"),
    ],

    # ─── 12. TỪ VIẾT TẮT TIẾNG VIỆT ─────────────────────────────────────────
    "Viết tắt tiếng Việt": [
        ("UBND",  "uỷ ban nhân dân"),
        ("TP.HCM","thành phố hồ chí minh"),
        ("TPHCM", "thành phố hồ chí minh"),
        ("CSGT",  "cảnh sát giao thông"),
        ("LHQ",   "liên hợp quốc"),
        ("CLB",   "câu lạc bộ"),
        ("HLV",   "huấn luyện viên"),
        ("TS",    "tiến sĩ"),
        ("GS",    "giáo sư"),
        ("THPT",  "trung học phổ thông"),
        ("THCS",  "trung học cơ sở"),
    ],

    # ─── 13. THẺ TIẾNG ANH (EN TAG) ─────────────────────────────────────────
    "Bảo vệ thẻ tiếng Anh <en>": [
        ("<en>Hello</en>",                              "<en>Hello</en>"),
        ("<en>Hello 123</en>",                          "<en>Hello 123</en>"),
        ("Xin chào <en>Good morning</en>",              "xin chào <en>Good morning</en>"),
        ("Ngày 21/02 <en>February 21</en>",             "ngày hai mươi mốt tháng hai <en>February 21</en>"),
        ("<en>AI</en> là trí tuệ nhân tạo",             "<en>AI</en> là trí tuệ nhân tạo"),
    ],

    # ─── 14. DẤU CÂU ─────────────────────────────────────────────────────────
    "Dấu câu và ký tự đặc biệt": [
        ("A & B",              "a và b"),
        ("A + B",              "a cộng b"),
        ("A = B",              "a bằng b"),
        ("#1",                 "thăng một"),
        ("(text in brackets)", ", text in brackets,"),
        ("[text in brackets]", ", text in brackets,"),
    ],

    # ─── 15. VIẾT TẮT ĐƠN GIẢN ──────────────────────────────────────────────
    "Viết tắt đơn giản": [
        ("v.v",  "vân vân"),
        ("v/v",  "về việc"),
        ("ko",   "không"),
        ("đ/c",  "địa chỉ"),
    ],

    # ─── 16. TRƯỜNG HỢP HỖN HỢP ──────────────────────────────────────────────
    "Hỗn hợp": [
        ("Ngày 21/02/2025 lúc 14h30, giá vàng đạt 100$ tại TPHCM",
         "ngày hai mươi mốt tháng hai năm hai nghìn không trăm hai mươi lăm lúc mười bốn giờ ba mươi phút, giá vàng đạt một trăm đô la tại thành phố hồ chí minh"),
        ("Thế kỷ XXI chứng kiến sự phát triển của <en>AI</en> và vũ trụ học",
         "thế kỷ hai mươi mốt chứng kiến sự phát triển của <en>AI</en> và vũ trụ học"),
    ],
}

def run_tests():
    total = 0
    passed = 0
    failed_cases = []

    for group_name, cases in test_groups.items():
        group_pass = 0
        group_fail = 0
        fails = []

        for input_text, expected in cases:
            actual = normalizer.normalize(input_text)
            actual_clean = " ".join(actual.split()).lower()
            expected_clean = " ".join(expected.split()).lower()
            total += 1
            if actual_clean == expected_clean:
                passed += 1
                group_pass += 1
            else:
                group_fail += 1
                failed_cases.append((group_name, input_text, expected, actual))
                fails.append((input_text, expected, actual))

        icon = "✅" if group_fail == 0 else "❌"
        print(f"\n{icon} [{group_name}] — Passed: {group_pass}/{group_pass+group_fail}")
        if fails:
            for input_text, expected, actual in fails:
                print(f"   Input:    {input_text}")
                print(f"   Expected: {expected}")
                print(f"   Actual:   {actual}")

    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} passed")
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
    print("=" * 60)

if __name__ == "__main__":
    results_path = os.path.join(parent_dir, 'test_results.txt')
    with io.open(results_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        run_tests()
    sys.stdout = sys.__stdout__
    print(f"Done. Results saved to {results_path}")
