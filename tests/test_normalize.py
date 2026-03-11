import pytest
from sea_g2p import Normalizer

@pytest.fixture
def normalizer():
    return Normalizer()

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
    ("tháng 3/2026", "tháng ba năm hai nghìn không trăm hai mươi sáu"),

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
    ("200 USD","hai trăm <en>u s d</en>"),
    ("500 VND","năm trăm đồng"),
    ("50 euro","năm mươi euro"),
    ("1000đ",  "một nghìn đồng"),
    ("75%",    "bảy mươi lăm phần trăm"),
    ("15,4% xuống còn 8,3%", "mười lăm phẩy bốn phần trăm xuống còn tám phẩy ba phần trăm"),
    ("370 tỷ USD", "ba trăm bảy mươi tỷ <en>u s d</en>"),
    ("5 triệu VND", "năm triệu đồng"),
    ("10 nghìn USD", "mười nghìn <en>u s d</en>"),
    ("8,92 tỷ USD", "tám phẩy chín mươi hai tỷ <en>u s d</en>"),

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
    ("kí tự w",     "kí tự đắp liu"),

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
    ("L là tên riêng", "lờ là tên riêng"),
    ("5m chiều dài", "năm mét chiều dài"),
    ("Đơn vị km", "đơn vị ki lô mét"),

    # ─── 20. EMAIL ───────────────────────────────────────────────────────────
    ("Liên hệ qua email pnnbao@gmail.com nhé.", "liên hệ qua email <en>pnnbao</en> a còng <en>gmail</en> chấm com nhé."),
    ("Email: contact@example.com", "email, <en>contact</en> a còng <en>example</en> chấm com"),

    # ─── 21. VIẾT TẮT ALPHANUMERIC (ENGLISH STYLE) ──────────────────────────
    ("Mô hình B2B rất phổ biến.", "mô hình <en>b two b</en> rất phổ biến."),
    ("Tôi dùng camera K3.", "tôi dùng camera ca ba."),
    ("Mã số A1B.", "mã số a một bê."),
    ("Tôi đang học về AI.", "tôi đang học về <en>a i</en>."),
    ("Dự án VYE.", "dự án <en>v y e</en>."),

    # ─── 22. PHÂN BIỆT HOA THƯỜNG TRONG CÂU ─────────────────────────────────
    ("TÔI ĐI HỌC", "tôi đi học"),
    ("Chào mừng bạn đến với CTY.", "chào mừng bạn đến với <en>c t y</en>."),

    # ─── 23. TOÀN DIỆN (CẢI TIẾN MỚI) ──────────────────────────────────────────
    # URLs
    ("Truy cập https://vieneu.io để biết thêm chi tiết.", "truy cập <en>https</en> <en>vieneu</en> chấm <en>i o</en> để biết thêm chi tiết."),
    ("Website www.google.com rất hữu ích.", "website <en>www</en> chấm <en>google</en> chấm com rất hữu ích."),

    # Slashes / Địa chỉ
    ("Địa chỉ nhà tôi là 123/4 đường Nguyễn Trãi.", "địa chỉ nhà tôi là một trăm hai mươi ba xẹt bốn đường nguyễn trãi."),
    ("Tỷ lệ là 100/2.", "tỷ lệ là một trăm xẹt hai."),

    # Ký hiệu toán học
    ("Nếu x > 5 và y ≤ 10 thì xấp xỉ ≈ 0.", "nếu ích lớn hơn năm và y nhỏ hơn hoặc bằng mười thì xấp xỉ xấp xỉ không."),
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
    ("Độ phân giải 300dpi.", "độ phân giải ba trăm đi phi ai."),
    ("Độ pH của nước là 7.", "độ pê hát của nước là bảy."),

    # Emails mở rộng
    ("Email công việc: admin@fpt.vn", "email công việc, <en>admin</en> a còng <en>f p t</en> chấm <en>v n</en>"),
    ("Liên hệ hotmail: test@hotmail.com", "liên hệ hotmail, <en>test</en> a còng <en>hotmail</en> chấm com"),
    ("Liên hệ qua email pnnbao@proton.com nhé.", "liên hệ qua email <en>pnnbao</en> a còng <en>proton</en> chấm com nhé."),

    # Redundant expansion (symbol + unit)
    ("#1kg", "thăng một ki lô gam"),

    # ─── 24. CÂU TEST THỰC TẾ ──────────────────────────────────────────────────
    ("Ông Lưu Trung Thái, Chủ tịch HĐQT MB cho biết, vốn hóa của ngân hàng đã tăng gần 10 lần kể từ năm 2017, đạt khoảng 8,5 tỷ USD, tạo nền tảng cho mục tiêu 10 tỷ USD vào năm 2027.",
     "ông lưu trung thái, chủ tịch hội đồng quản trị <en>m b</en> cho biết, vốn hóa của ngân hàng đã tăng gần mười lần kể từ năm hai nghìn không trăm mười bảy, đạt khoảng tám phẩy năm tỷ <en>u s d</en>, tạo nền tảng cho mục tiêu mười tỷ <en>u s d</en> vào năm hai nghìn không trăm hai mươi bảy."),

    # ─── 25. THÊM MỚI (CẢI THIỆN) ──────────────────────────────────────────
    ("Số tiền là 1.000.000.000.000 đồng.", "số tiền là một nghìn tỷ đồng."),
    ("“Lời chào cao hơn mâm cỗ”", "lời chào cao hơn mâm cỗ"),
    ("‘Trân trọng’", "trân trọng"),
    ("Chào <en>world</en> xinh đẹp", "chào <en>world</en> xinh đẹp"),
    ("Nếu đã từng đọc cuốn sách trên của Simon, hoặc đã xem anh thuyết trình về khái niệm tại sao trên diễn đàn TED.com, thì có lẽ bạn không còn xa lạ với vòng tròn vàng.", "nếu đã từng đọc cuốn sách trên của simon, hoặc đã xem anh thuyết trình về khái niệm tại sao trên diễn đàn <en>t e d</en> chấm com, thì có lẽ bạn không còn xa lạ với vòng tròn vàng."),

    # ─── 26. TRƯỜNG HỢP CẠNH (EDGE CASES) ─────────────────────────────────────
    ("MixedCase Acronyms như ChatGPT hay Claude.", "mixedcase acronyms như chatgpt hay claude."),
    ("Unit mix: 10km/h và 5m/s.", "unit mix, mười ki lô mét trên giờ và năm mét trên giây."),
    ("Số cực lớn: 1.000.000.000.000.000.000", "số cực lớn, một tỷ tỷ"),
    ("Email với tên miền lạ: user@domain.tech", "email với tên miền lạ, <en>user</en> a còng <en>domain</en> chấm <en>tech</en>"),
    
    # ─── 27. PHẨY VÀ DẤU NHÁY (PRIMES & APOSTROPHES) ──────────────────────────
    ("A' là một ký tự đặc biệt", "a phẩy là một ký tự đặc biệt"),
    ("Đây là phút 1'", "đây là phút một phẩy"),
    ("I don't know why", "i don't know why"),
    ("It's a beautiful day", "it's a beautiful day"),
    ("Giá của 'Sản phẩm' này là $10", "giá của sản phẩm này là mười đô la mỹ"),
    ("Giá SP500 hôm nay là 4.200,5 điểm", "giá ét pê năm trăm hôm nay là bốn nghìn hai trăm phẩy năm điểm"),
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
