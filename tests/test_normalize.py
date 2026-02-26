import pytest
from vieneu_utils.normalize_text import VietnameseTTSNormalizer

@pytest.fixture
def normalizer():
    return VietnameseTTSNormalizer()

# Combined test cases from multiple categories
TEST_CASES = [
    # ─── 1. SỐ THÔNG THƯỜNG ────────────────────────────────────────────────────
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

    # ─── 2. SỐ THẬP PHÂN / SỐ CÓ DẤU PHÂN CÁCH ──────────────────────────────
    ("1.000",    "một nghìn"),
    ("1.000.000","một triệu"),
    ("3,14",     "ba phẩy mười bốn"),
    ("1.3",      "một chấm ba"),

    # ─── 3. SỐ ĐIỆN THOẠI ────────────────────────────────────────────────────
    ("0912345678",    "không chín một hai ba bốn năm sáu bảy tám"),
    ("+84912345678",  "cộng tám bốn chín một hai ba bốn năm sáu bảy tám"),

    # ─── 4. SỐ THỨ TỰ ─────────────────────────────────────────────────────────
    ("thứ 1",  "thứ nhất"),
    ("thứ 4",  "thứ tư"),
    ("thứ 5",  "thứ năm"),
    ("hạng 1", "hạng nhất"),

    # ─── 5. PHÉP NHÂN ─────────────────────────────────────────────────────────
    ("3 x 4",  "ba nhân bốn"),
    ("10 x 20","mười nhân hai mươi"),

    # ─── 6. NGÀY THÁNG ─────────────────────────────────────────────────────────
    ("21/02/2025", "ngày hai mươi mốt tháng hai năm hai nghìn không trăm hai mươi lăm"),
    ("01-01-2024", "ngày một tháng một năm hai nghìn không trăm hai mươi bốn"),
    ("31.12.2023", "ngày ba mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi ba"),
    ("31.12.1997", "ngày ba mươi mốt tháng mười hai năm một nghìn chín trăm chín mươi bảy"),
    ("21/02", "ngày hai mươi mốt tháng hai"),
    ("01/12", "ngày một tháng mười hai"),
    ("02/2025", "tháng hai năm hai nghìn không trăm hai mươi lăm"),
    ("12/2024", "tháng mười hai năm hai nghìn không trăm hai mươi bốn"),
    ("32/01", "ba mươi hai xẹt không một"),
    ("01/13", "không một trên mười ba"),

    # ─── 7. THỜI GIAN ─────────────────────────────────────────────────────────
    ("14h30",   "mười bốn giờ ba mươi phút"),
    ("8h05",    "tám giờ không năm phút"),
    ("0h00",    "không giờ không phút"),
    ("23:59",   "hai mươi ba giờ năm mươi chín phút"),
    ("12:00:00","mười hai giờ không phút không giây"),
    ("10:20 phút", "mười giờ hai mươi phút"),
    ("12:00:00 giây", "mười hai giờ không phút không giây"),

    # ─── 8. TIỀN TỆ ──────────────────────────────────────────────────────────
    ("100$",   "một trăm đô la Mỹ"),
    ("$50",    "năm mươi đô la Mỹ"),
    ("200 USD","hai trăm đô la Mỹ"),
    ("500 VND","năm trăm đồng"),
    ("50 euro","năm mươi ơ rô"),
    ("1000đ",  "một nghìn đồng"),
    ("75%",    "bảy mươi lăm phần trăm"),
    ("15,4% xuống còn 8,3%", "mười lăm phẩy bốn phần trăm xuống còn tám phẩy ba phần trăm"),
    ("370 tỷ USD", "ba trăm bảy mươi tỷ đô la Mỹ"),
    ("5 triệu VND", "năm triệu đồng"),
    ("10 nghìn USD", "mười nghìn đô la Mỹ"),
    ("8,92 tỷ USD", "tám phẩy chín mươi hai tỷ đô la Mỹ"),

    # ─── 9. ĐƠN VỊ ĐO LƯỜNG ─────────────────────────────────────────────────
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
    ("5 triệu km", "năm triệu ki lô mét"),
    ("1,5 ha", "một phẩy năm héc ta"),
    ("1.5 ha", "một chấm năm héc ta"),
    ("AN/ASQ", "an trên asq"),

    # ─── 10. KHOẢNG / DÃY SỐ ──────────────────────────────────────────────────
    ("700-900", "bảy trăm đến chín trăm"),
    ("0,5-0,9", "không phẩy năm đến không phẩy chín"),

    # ─── 11. SỐ LA MÃ ────────────────────────────────────────────────────────
    ("Thế kỷ XXI",  "thế kỷ hai mươi mốt"),
    ("Chương IV",   "chương bốn"),
    ("Hồi IX",      "hồi chín"),
    ("Phần III",    "phần ba"),
    ("Thế kỷ XX",   "thế kỷ hai mươi"),

    # ─── 12. CHỮ CÁI ─────────────────────────────────────────────────────────
    ("ký tự A",     "ký tự a"),
    ("chữ B",       "chữ bê"),
    ("ký tự 'C'",   "ký tự xê"),
    ("chữ cái Z",   "chữ cái dét"),
    ("kí tự w",     "kí tự vê kép"),

    # ─── 13. TỪ VIẾT TẮT TIẾNG VIỆT ─────────────────────────────────────────
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

    # ─── 14. THẺ TIẾNG ANH (EN TAG) ─────────────────────────────────────────
    ("<en>Hello</en>",                              "<en>Hello</en>"),
    ("<en>Hello 123</en>",                          "<en>Hello 123</en>"),
    ("Xin chào <en>Good morning</en>",              "xin chào <en>Good morning</en>"),
    ("Ngày 21/02 <en>February 21</en>",             "ngày hai mươi mốt tháng hai <en>February 21</en>"),
    ("<en>AI</en> là trí tuệ nhân tạo",             "<en>AI</en> là trí tuệ nhân tạo"),

    # ─── 15. DẤU CÂU ─────────────────────────────────────────────────────────
    ("A & B",              "a và bê"),
    ("A + B",              "a cộng bê"),
    ("A = B",              "a bằng bê"),
    ("#1",                 "thăng một"),
    ("(text in brackets)", "text in brackets"),
    ("[text in brackets]", "text in brackets"),
    ("(giờ Mỹ)", "giờ mỹ"),
    ("hiệu lực từ 0h01 (giờ Mỹ), trong vòng", "hiệu lực từ không giờ không một phút, giờ mỹ, trong vòng"),
    ("hiệu lực từ 0h01 (giờ Mỹ) trong vòng", "hiệu lực từ không giờ không một phút, giờ mỹ, trong vòng"),
    ("kết thúc (0h01).", "kết thúc, không giờ không một phút."),
    ("chỉ số là 7,05 - đường huyết là 1.8", "chỉ số là bảy phẩy không năm, đường huyết là một chấm tám"),

    # ─── 16. CẤU TRÚC VĂN BẢN ──────────────────────────────────────────────
    ("Đoạn 1.\nĐoạn 2.", "đoạn một.\nđoạn hai."),
    ("\n12 tiêm kích", "\nmười hai tiêm kích"),

    # ─── 17. VIẾT TẮT ĐƠN GIẢN ──────────────────────────────────────────────
    ("v.v",  "vân vân"),
    ("v/v",  "về việc"),
    ("ko",   "không"),
    ("đ/c",  "địa chỉ"),

    # ─── 18. TRƯỜNG HỢP HỖN HỢP ──────────────────────────────────────────────
    ("Ngày 21/02/2025 lúc 14h30, giá vàng đạt 100$ tại TPHCM",
     "ngày hai mươi mốt tháng hai năm hai nghìn không trăm hai mươi lăm lúc mười bốn giờ ba mươi phút, giá vàng đạt một trăm đô la Mỹ tại thành phố hồ chí minh"),
    ("Thế kỷ XXI chứng kiến sự phát triển của <en>AI</en> và vũ trụ học",
     "thế kỷ hai mươi mốt chứng kiến sự phát triển của <en>AI</en> và vũ trụ học"),
    ("Đề án 06 và Chỉ thị 04", "đề án không sáu và chỉ thị không bốn"),

    # ─── 19. AN TOÀN (KHÔNG NHẦM LẪN) ──────────────────────────────────────────
    ("Anh M. đi bộ", "anh mờ đi bộ"),
    ("Vitamin G", "vitamin gờ"),
    ("L. là tên riêng", "lờ là tên riêng"),
    ("5m chiều dài", "năm mét chiều dài"),
    ("Đơn vị km", "đơn vị ki lô mét"),

    # ─── 20. EMAIL ───────────────────────────────────────────────────────────
    ("Liên hệ qua email pnnbao@gmail.com nhé.", "liên hệ qua email phê nờ nờ bê a o a còng gờ meo chấm com nhé."),
    ("Email: contact@example.com", "email: xê o nờ tê a xê tê a còng e ích xì a mờ phê lờ e chấm xê o mờ"),

    # ─── 21. VIẾT TẮT ALPHANUMERIC (ENGLISH STYLE) ──────────────────────────
    ("Mô hình B2B rất phổ biến.", "mô hình bi two bi rất phổ biến."),
    ("Tôi dùng camera K3.", "tôi dùng camera cây three."),
    ("Mã số A1B.", "mã số ây one bi."),
    ("Tôi đang học về AI.", "tôi đang học về ây ai."),
    ("Dự án VYE.", "dự án vi quai i"),

    # ─── 22. PHÂN BIỆT HOA THƯỜNG TRONG CÂU ─────────────────────────────────
    ("TÔI ĐI HỌC", "tôi đi học"),
    ("Chào mừng bạn đến với CTY.", "chào mừng bạn đến với xi ti quai."),

    # ─── 23. TOÀN DIỆN (CẢI TIẾN MỚI) ──────────────────────────────────────────
    # URLs
    ("Truy cập https://vieneu.io để biết thêm chi tiết.", "truy cập hát tê tê phê sờ hai chấm xẹt xẹt vờ i e nờ e u chấm i o để biết thêm chi tiết."),
    ("Website www.google.com rất hữu ích.", "website vê kép vê kép vê kép chấm gờ o o gờ lờ e chấm xê o mờ rất hữu ích."),

    # Slashes / Địa chỉ
    ("Địa chỉ nhà tôi là 123/4 đường Nguyễn Trãi.", "địa chỉ nhà tôi là một trăm hai mươi ba xẹt bốn đường nguyễn trãi."),
    ("Tỷ lệ là 100/2.", "tỷ lệ là một trăm xẹt hai."),

    # Ký hiệu toán học
    ("Nếu x > 5 và y ≤ 10 thì xấp xỉ ≈ 0.", "nếu ích xì lớn hơn năm và i dài nhỏ hơn hoặc bằng mười thì xấp xỉ xấp xỉ không."),
    ("Nhiệt độ là 30°C ± 2°C.", "nhiệt độ là ba mươi độ xê cộng trừ hai độ xê."),
    ("Biểu thức ≥ 10.", "biểu thức lớn hơn hoặc bằng mười."),

    # Đơn vị đo lường mở rộng
    ("Dung lượng 16GB.", "dung lượng mười sáu gi ga bai."),
    ("File nặng 50MB.", "file nặng năm mươi mê ga bai."),
    ("Ổ cứng 1TB.", "ổ cứng một tê ra bai."),
    ("Căn hộ 75sqm.", "căn hộ bảy mươi lăm mét vuông."),
    ("Bể bơi 100cum.", "bể bơi một trăm mét khối."),
    ("Âm thanh 80db.", "âm thanh tám mươi đê xi ben."),
    ("Trọng lượng 10lb.", "trọng lượng mười pao."),
    ("Màn hình 24in.", "màn hình hai mươi bốn ins."),
    ("Độ phân giải 300dpi.", "độ phân giải ba trăm đê phê i"),

    # Emails mở rộng
    ("Email công việc: admin@fpt.vn", "email công việc: a dê mờ i nờ a còng ép phê tê chấm vê nờ"),
    ("Liên hệ hotmail: test@hotmail.com", "liên hệ hotmail: tê e sờ tê a còng hót meo chấm com"),

    # Redundant expansion (symbol + unit)
    ("#1kg", "thăng một ki lô gam"),
]

@pytest.mark.parametrize("input_text, expected", TEST_CASES)
def test_normalize(normalizer, input_text, expected):
    actual = normalizer.normalize(input_text)
    # Clean up whitespace for comparison
    actual_clean = " ".join(actual.split()).lower()
    expected_clean = " ".join(expected.split()).lower()
    
    # Special handling for brackets/punctuation to match expectation
    # Note: brackets are sometimes kept or removed depending on normalizer state
    # We follow the user provided expected strings
    assert actual_clean == expected_clean
