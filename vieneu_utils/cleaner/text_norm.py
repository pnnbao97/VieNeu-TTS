import re
from .num2vi import n2w
from .symbols import vietnamese_re, vietnamese_without_num_re

_measurement_key_vi = {
    "km": "ki lô mét", "dm": "đê xi mét", "cm": "xen ti mét", "mm": "mi li mét",
    "nm": "na nô mét", "µm": "mic rô mét", "μm": "mic rô mét", "m": "mét",
    "kg": "ki lô gam", "g": "gam", "mg": "mi li gam",
    "km2": "ki lô mét vuông", "m2": "mét vuông", "cm2": "xen ti mét vuông", "mm2": "mi li mét vuông",
    "ha": "héc ta",
    "km3": "ki lô mét khối", "m3": "mét khối", "cm3": "xen ti mét khối", "mm3": "mi li mét khối",
    "l": "lít", "dl": "đê xi lít", "ml": "mi li lít", "hl": "héc tô lít",
    "kw": "ki lô oát", "mw": "mê ga oát", "gw": "gi ga oát",
    "kwh": "ki lô oát giờ", "mwh": "mê ga oát giờ", "wh": "oát giờ",
    "hz": "héc", "khz": "ki lô héc", "mhz": "mê ga héc", "ghz": "gi ga héc",
    "pa": "pát cal", "kpa": "ki lô pát cal", "mpa": "mê ga pát cal",
    "bar": "ba", "mbar": "mi li ba", "atm": "át mốt phia", "psi": "pi ét xai",
    "j": "giun", "kj": "ki lô giun",
    "cal": "ca lo", "kcal": "ki lô ca lo",
    "h": "giờ", "p": "phút", "s": "giây"
}

_currency_key = {
    "usd": "đô la", "vnd": "đồng", "đ": "đồng", "euro": "ơ rô", "%": "phần trăm"
}

_letter_key_vi = {
    "a": "ây", "b": "bê", "c": "xê", "d": "dê", "đ": "đê", "f": "ép", "g": "gờ", "h": "hát",
    "i": "ai", "j": "chây", "k": "kây", "l": "lờ", "m": "mờ", "n": "nờ", "o": "ô",
    "p": "pê", "q": "kiu", "r": "rờ", "s": "ét", "t": "ti", "v": "vi", "w": "vê kép", "x": "ít", "z": "dét"
}

_acronyms_exceptions_vi = {
    "CĐV": "cổ động viên", "TV": "ti vi", "HĐND": "hội đồng nhân dân", "TAND": "toàn án nhân dân",
    "BHXH": "bảo hiểm xã hội", "BHTN": "bảo hiểm thất nghiệp", "TP.HCM": "thành phố hồ chí minh",
    "VN": "việt nam", "UBND": "uỷ ban nhân dân", "TP": "thành phố", "HCM": "hồ chí minh",
    "HN": "hà nội", "BTC": "ban tổ chức", "CLB": "câu lạc bộ", "HTX": "hợp tác xã",
    "NXB": "nhà xuất bản", "TW": "trung ương", "CSGT": "cảnh sát giao thông", "LHQ": "liên hợp quốc",
    "THCS": "trung học cơ sở", "THPT": "trung học phổ thông", "ĐH": "đại học", "HLV": "huấn luyện viên",
    "GS": "giáo sư", "TS": "tiến sĩ", "TNHH": "trách nhiệm hữu hạn", "VĐV": "vận động viên",
    "GDP": "gi đi pi", "FDI": "ép đê i", "ODA": "ô đê a", "covid": "cô vít", "youtube": "du túp",
    "TPHCM": "thành phố hồ chí minh", "ĐH": "đại học", "PGS": "phó giáo sư"
}

_roman_number_re = r"\b(?=[IVXLCDM]{2,})M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b"
_letter_re = r"(chữ|chữ cái|kí tự|ký tự)\s+(['\"]?)([a-z])(['\"]?)\b"

def _strip_dot_sep(num_str):
    if re.fullmatch(r"\d+(\.\d{3})+", num_str):
        return num_str.replace(".", "")
    return num_str

def expand_measurement(text):
    def _repl(m, full):
        num = _strip_dot_sep(m.group(1))
        return f"{n2w(num)} {full}"
    
    for unit, full in _measurement_key_vi.items():
        if len(unit) == 1:
            pattern = rf"((?:\d+\.)*\d+)\s*{unit}\b"
            text = re.sub(pattern, lambda m, f=full: _repl(m, f), text)
        else:
            pattern = rf"((?:\d+\.)*\d+)\s*{unit}\b"
            text = re.sub(pattern, lambda m, f=full: _repl(m, f), text, flags=re.IGNORECASE)
    return text

def expand_currency(text):
    def _repl(m, full):
        num = _strip_dot_sep(m.group(1))
        return f"{n2w(num)} {full}"
    text = re.sub(r"\$\s*((\d+\.)*\d+)", lambda m: f"{n2w(_strip_dot_sep(m.group(1)))} đô la", text)
    text = re.sub(r"((\d+\.)*\d+)\s*\$", lambda m: f"{n2w(_strip_dot_sep(m.group(1)))} đô la", text)
    text = re.sub(r"((\d+\.)*\d+)\s*%", lambda m: f"{n2w(_strip_dot_sep(m.group(1)))} phần trăm", text)
    for unit, full in _currency_key.items():
        if unit == "%": continue
        text = re.sub(rf"\b((\d+\.)*\d+)\s*{unit}\b", lambda m, f=full: _repl(m, f), text, flags=re.IGNORECASE)
    return text

def expand_compound_units(text):
    def _repl_compound(m):
        num = _strip_dot_sep(m.group(1)) if m.group(1) else ""
        u1 = m.group(2).lower()
        u2 = m.group(3).lower()
        full1 = _measurement_key_vi.get(u1, u1)
        full2 = _measurement_key_vi.get(u2, u2)
        res = f"{full1} trên {full2}"
        if num:
            res = f"{n2w(num)} {res}"
        return res

    pattern = r"(\d+(?:\.\d{3})*)?\s*\b([a-zμµ²³°]+)/([a-zμµ²³°0-9]+)\b"
    text = re.sub(pattern, _repl_compound, text, flags=re.IGNORECASE)
    return text

def expand_roman(match):
    roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    num = match.group(0).upper()
    if not num: return ""
    result = 0
    for i, c in enumerate(num):
        if (i + 1) == len(num) or roman_numerals[c] >= roman_numerals[num[i + 1]]:
            result += roman_numerals[c]
        else:
            result -= roman_numerals[c]
    return f" {n2w(str(result))} "

def expand_letter(match):
    prefix, q1, char, q2 = match.groups()
    if char.lower() in _letter_key_vi:
        return f"{prefix} {_letter_key_vi[char.lower()]} "
    return match.group(0)

def normalize_others(text):
    abbrs = {"v.v": " vân vân", "v/v": " về việc", "ko": " không", "đ/c": "địa chỉ"}
    for k, v in abbrs.items():
        text = text.replace(k, v)

    for k, v in _acronyms_exceptions_vi.items():
        text = re.sub(rf"\b{k}\b", v, text)
    
    text = expand_compound_units(text)
    text = expand_measurement(text)
    text = expand_currency(text)
    text = re.sub(_roman_number_re, expand_roman, text)
    text = re.sub(_letter_re, expand_letter, text, flags=re.IGNORECASE)
    
    def _expand_alphanumeric(m):
        num = m.group(1)
        char = m.group(2).lower()
        if char in _letter_key_vi:
            pronunciation = _letter_key_vi[char]
            if char == 'd' and ('quốc lộ' in text.lower() or 'ql' in text.lower()):
                pronunciation = 'đê'
            return f"{num} {pronunciation}"
        return m.group(0)
    
    text = re.sub(r'\b(\d+)([a-zA-Z])\b', _expand_alphanumeric, text)
    
    text = text.replace('"', '').replace("'", '').replace(''', '').replace(''', '')
    text = text.replace('&', ' và ').replace('+', ' cộng ').replace('=', ' bằng ').replace('#', ' thăng ')
    
    text = re.sub(r'[\(\[\{]\s*(.*?)\s*[\)\]\}]', r', \1, ', text)
    text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
    
    text = re.sub(r'-(\d+(?:[.,]\d+)?)\s*°\s*c\b', r'âm \1 độ xê', text, flags=re.IGNORECASE)
    text = re.sub(r'-(\d+(?:[.,]\d+)?)\s*°\s*f\b', r'âm \1 độ ép', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:[.,]\d+)?)\s*°\s*c\b', r'\1 độ xê', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:[.,]\d+)?)\s*°\s*f\b', r'\1 độ ép', text, flags=re.IGNORECASE)
    text = re.sub(r'°', ' độ ', text)

    def _expand_version(m):
        return ' chấm '.join(m.group(0).split('.'))
    text = re.sub(r'\b\d+(?:\.\d+){2,}\b', _expand_version, text)

    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữợỳýỷỹỵđ.,!?;:@%_]', ' ', text)
    
    return text
