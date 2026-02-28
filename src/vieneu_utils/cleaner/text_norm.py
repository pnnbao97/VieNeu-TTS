import re
from .num2vi import n2w, n2w_single
from .symbols import vietnamese_re, vietnamese_without_num_re

_en_letter_names = {
    "a": "ây", "b": "bi", "c": "xi", "d": "đi", "e": "i", "f": "ép", "g": "di",
    "h": "ếch", "i": "ai", "j": "giây", "k": "cây", "l": "eo", "m": "em", "n": "en",
    "o": "âu", "p": "pi", "q": "kiu", "r": "a", "s": "ét", "t": "ti", "u": "du",
    "v": "vi", "w": "đắp bờ liu", "x": "ích", "y": "quai", "z": "di",
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

_vi_letter_names = {
    "a": "a", "b": "bê", "c": "xê", "d": "dê", "đ": "đê", "e": "e", "ê": "ê",
    "f": "ép", "g": "gờ", "h": "hát", "i": "i", "j": "giây", "k": "ca", "l": "lờ",
    "m": "mờ", "n": "nờ", "o": "o", "ô": "ô", "ơ": "ơ", "p": "phê", "q": "qui",
    "r": "rờ", "s": "sờ", "t": "tê", "u": "u", "ư": "ư", "v": "vờ", "w": "vê kép",
    "x": "ích xì", "y": "i dài", "z": "dét"
}

_common_email_domains = {
    "gmail.com": "gờ meo chấm com",
    "yahoo.com": "da hu chấm com",
    "yahoo.com.vn": "da hu chấm com chấm vê nờ",
    "outlook.com": "aut lúc chấm com",
    "hotmail.com": "hót meo chấm com",
    "icloud.com": "ai clao chấm com",
    "fpt.vn": "ép phê tê chấm vê nờ",
    "fpt.com.vn": "ép phê tê chấm com chấm vê nờ",
}

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
    "h": "giờ", "p": "phút", "s": "giây",
    "sqm": "mét vuông", "cum": "mét khối",
    "gb": "gi ga bai", "mb": "mê ga bai", "kb": "ki lô bai", "tb": "tê ra bai",
    "db": "đê xi ben", "oz": "ao xơ", "lb": "pao", "lbs": "pao",
    "ft": "phít", "in": "ins", "dpi": "đê phê i"
}

_currency_key = {
    "usd": "đô la Mỹ", "vnd": "đồng", "đ": "đồng", "euro": "ơ rô", "%": "phần trăm"
}

_letter_key_vi = _vi_letter_names

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

# Compiled Regular Expressions
RE_ROMAN_NUMBER = re.compile(r"\b(?=[IVXLCDM]{2,})M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b")
RE_LETTER = re.compile(r"(chữ|chữ cái|kí tự|ký tự)\s+(['\"]?)([a-z])(['\"]?)\b", re.IGNORECASE)
RE_STANDALONE_LETTER = re.compile(r'\b([a-zA-Z])\b\.?')
RE_URL = re.compile(r'\b(?:https?://|www\.)[A-Za-z0-9.\-_~:/?#\[\]@!$&\'()*+,;=]+\b')
RE_SLASH_NUMBER = re.compile(r'\b(\d+)/(\d+)\b')
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
RE_SENTENCE_SPLIT = re.compile(r'([.!?]+(?:\s+|$))')
RE_ACRONYM = re.compile(r'\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,}\b')
RE_ALPHANUMERIC = re.compile(r'\b(\d+)([a-zA-Z])\b')
RE_BRACKETS = re.compile(r'[\(\[\{]\s*(.*?)\s*[\)\]\}]')
RE_STRIP_BRACKETS = re.compile(r'[\[\]\(\)\{\}]')
RE_TEMP_C_NEG = re.compile(r'-(\d+(?:[.,]\d+)?)\s*°\s*c\b', re.IGNORECASE)
RE_TEMP_F_NEG = re.compile(r'-(\d+(?:[.,]\d+)?)\s*°\s*f\b', re.IGNORECASE)
RE_TEMP_C = re.compile(r'(\d+(?:[.,]\d+)?)\s*°\s*c\b', re.IGNORECASE)
RE_TEMP_F = re.compile(r'(\d+(?:[.,]\d+)?)\s*°\s*f\b', re.IGNORECASE)
RE_DEGREE = re.compile(r'°')
RE_VERSION = re.compile(r'\b(\d+(?:\.\d+)+)\b')
RE_CLEAN_OTHERS = re.compile(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữỳýỷỹỵđ.,!?;:@%_]')

# Reusable patterns for measurement/currency
_MAGNITUDE_P = r"\s*(tỷ|triệu|nghìn|ngàn)?\s*"
_NUMERIC_P = r"((?:\d+[.,])*\d+)"

def _expand_number_with_sep(num_str):
    if not num_str: return ""
    if "," in num_str:
        # Standard Vietnamese float: 1,5 or 1.000,5
        clean_num = num_str.replace(".", "")
        parts = clean_num.split(",")
        if len(parts) == 2:
            return f"{n2w(parts[0])} phẩy {n2w(parts[1])}"
    
    if "." in num_str:
        # Check if it's a thousand separator format (e.g. 1.000, 1.000.000)
        # Vietnamese thousand sep is ALWAYS exactly 3 digits after the dot.
        if re.fullmatch(r"\d+(?:\.\d{3})+", num_str):
            return n2w(num_str.replace(".", ""))
        # Otherwise treat dot as "chấm" (e.g. version 1.3 or English-style decimal 1.5)
        return " chấm ".join([n2w(p) for p in num_str.split(".")])
        
    return n2w(num_str)

def expand_measurement(text):
    def _repl(m, full):
        num = m.group(1)
        mag = m.group(2) if m.group(2) else ""
        expanded_num = _expand_number_with_sep(num)
        return f"{expanded_num} {mag} {full}".replace("  ", " ").strip()
    
    for unit, full in sorted(_measurement_key_vi.items(), key=lambda x: len(x[0]), reverse=True):
        # Case with number
        pattern = re.compile(rf"\b{_NUMERIC_P}{_MAGNITUDE_P}{unit}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, f=full: _repl(m, f), text)
        
        # Standalone units
        safe_standalone = [
            "km", "cm", "mm", "kg", "mg",
            "m2", "km2", "usd", "vnd",
            "mhz", "khz", "ghz", "hz"
        ]
        if unit.lower() in safe_standalone:
            # First try with standard word boundaries
            standalone_pattern = re.compile(rf"(?<![\d.,])\b{unit}\b", re.IGNORECASE)
            text = standalone_pattern.sub(f" {full} ", text)
    return text

def expand_currency(text):
    def _repl(m, full):
        num = m.group(1)
        mag = m.group(2) if m.group(2) else ""
        expanded_num = _expand_number_with_sep(num)
        return f"{expanded_num} {mag} {full}".replace("  ", " ").strip()
        
    text = re.sub(rf"\$\s*{_NUMERIC_P}{_MAGNITUDE_P}", lambda m: _repl(m, "đô la Mỹ"), text)
    text = re.sub(rf"{_NUMERIC_P}{_MAGNITUDE_P}\$", lambda m: _repl(m, "đô la Mỹ"), text)
    text = re.sub(rf"{_NUMERIC_P}\s*%", lambda m: f"{_expand_number_with_sep(m.group(1))} phần trăm", text)
    
    for unit, full in _currency_key.items():
        if unit == "%": continue
        pattern = re.compile(rf"\b{_NUMERIC_P}{_MAGNITUDE_P}{unit}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, f=full: _repl(m, f), text)
    return text

def expand_compound_units(text):
    def _repl_compound(m):
        num_str = m.group(1) if m.group(1) else ""
        num = _expand_number_with_sep(num_str)
        u1 = m.group(2).lower()
        u2 = m.group(3).lower()
        full1 = _measurement_key_vi.get(u1, u1)
        full2 = _measurement_key_vi.get(u2, u2)
        res = f" {full1} trên {full2} "
        if num:
            res = f"{num} {res}"
        return res

    pattern = re.compile(rf"{_NUMERIC_P}?\s*\b([a-zμµ²³°]+)/([a-zμµ²³°0-9]+)\b", re.IGNORECASE)
    text = pattern.sub(_repl_compound, text)
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

def expand_abbreviations(text):
    abbrs = {"v.v": " vân vân", "v/v": " về việc", "ko": " không", "đ/c": "địa chỉ"}
    for k, v in abbrs.items():
        text = text.replace(k, v)
    return text

def expand_standalone_letters(text):
    def _repl_letter(m):
        char = m.group(1).lower()
        if char in _letter_key_vi:
            return f" {_letter_key_vi[char]} "
        return m.group(0)
    
    return RE_STANDALONE_LETTER.sub(_repl_letter, text)

def normalize_urls(text):
    def _repl_url(m):
        url = m.group(0)
        res = []
        for char in url.lower():
            if char.isalnum():
                if char.isdigit():
                    res.append(n2w_single(char))
                else:
                    res.append(_vi_letter_names.get(char, char))
            elif char == '.': res.append('chấm')
            elif char == '/': res.append('xẹt')
            elif char == ':': res.append('hai chấm')
            elif char == '-': res.append('gạch ngang')
            elif char == '_': res.append('gạch dưới')
            elif char == '?': res.append('hỏi')
            elif char == '&': res.append('và')
            elif char == '=': res.append('bằng')
            else: res.append(char)
        return " ".join(res)

    return RE_URL.sub(_repl_url, text)

def normalize_slashes(text):
    def _repl(m):
        n1 = m.group(1)
        n2 = m.group(2)
        # If it's likely an address (first number is large)
        if len(n1) > 2 or int(n1) > 31:
            return f"{n2w(n1)} xẹt {n2w(n2)}"
        return f"{n2w(n1)} trên {n2w(n2)}"
    return RE_SLASH_NUMBER.sub(_repl, text)

def normalize_emails(text):
    def _repl_email(m):
        email = m.group(0)
        parts = email.split('@')
        if len(parts) != 2: return email

        user_part, domain_part = parts

        # User part: spell out
        user_norm = []
        for char in user_part.lower():
            if char.isalnum():
                if char.isdigit():
                    user_norm.append(n2w_single(char))
                else:
                    user_norm.append(_vi_letter_names.get(char, char))
            elif char == '.': user_norm.append('chấm')
            elif char == '_': user_norm.append('gạch dưới')
            elif char == '-': user_norm.append('gạch ngang')
            else: user_norm.append(char)

        # Domain part
        domain_part_lower = domain_part.lower()
        if domain_part_lower in _common_email_domains:
            domain_norm = _common_email_domains[domain_part_lower]
        else:
            domain_parts = domain_part.split('.')
            norm_domain_parts = []
            for dp in domain_parts:
                dp_norm = []
                for char in dp.lower():
                    if char.isalnum():
                        if char.isdigit():
                            dp_norm.append(n2w_single(char))
                        else:
                            dp_norm.append(_vi_letter_names.get(char, char))
                    else: dp_norm.append(char)
                norm_domain_parts.append(" ".join(dp_norm))
            domain_norm = " chấm ".join(norm_domain_parts)

        return " ".join(user_norm) + " a còng " + domain_norm

    return RE_EMAIL.sub(_repl_email, text)

def normalize_acronyms(text):
    sentences = RE_SENTENCE_SPLIT.split(text)
    processed = []
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        sep = sentences[i+1] if i+1 < len(sentences) else ""
        if not s:
            processed.append(sep)
            continue

        words = s.split()
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        is_all_caps = len(alpha_words) > 0 and all(w.isupper() for w in alpha_words)

        if not is_all_caps:
            def _repl_acronym(m):
                word = m.group(0)
                if word.isdigit(): return word
                return " ".join(_en_letter_names.get(c.lower(), c) for c in word)

            s = RE_ACRONYM.sub(_repl_acronym, s)

        processed.append(s + sep)
    return "".join(processed)

def normalize_others(text):
    for k, v in _acronyms_exceptions_vi.items():
        text = re.sub(rf"\b{k}\b", v, text)
    
    text = normalize_urls(text)
    text = normalize_emails(text)
    text = normalize_slashes(text)

    text = RE_ROMAN_NUMBER.sub(expand_roman, text)
    text = RE_LETTER.sub(expand_letter, text)
    
    def _expand_alphanumeric(m):
        num = m.group(1)
        char = m.group(2).lower()
        if char in _letter_key_vi:
            pronunciation = _letter_key_vi[char]
            if char == 'd' and ('quốc lộ' in text.lower() or 'ql' in text.lower()):
                pronunciation = 'đê'
            return f"{num} {pronunciation}"
        return m.group(0)
    
    text = RE_ALPHANUMERIC.sub(_expand_alphanumeric, text)
    
    # Handle quotes and smart quotes
    text = re.sub(r'["\'“”‘’]', '', text)

    text = text.replace('&', ' và ').replace('+', ' cộng ').replace('=', ' bằng ').replace('#', ' thăng ')
    text = text.replace('>', ' lớn hơn ').replace('<', ' nhỏ hơn ')
    text = text.replace('≥', ' lớn hơn hoặc bằng ').replace('≤', ' nhỏ hơn hoặc bằng ')
    text = text.replace('±', ' cộng trừ ').replace('≈', ' xấp xỉ ')

    text = expand_compound_units(text)
    text = expand_measurement(text)
    text = expand_currency(text)
    
    text = RE_BRACKETS.sub(r', \1, ', text)
    text = RE_STRIP_BRACKETS.sub(' ', text)
    
    text = RE_TEMP_C_NEG.sub(r'âm \1 độ xê', text)
    text = RE_TEMP_F_NEG.sub(r'âm \1 độ ép', text)
    text = RE_TEMP_C.sub(r'\1 độ xê', text)
    text = RE_TEMP_F.sub(r'\1 độ ép', text)
    text = RE_DEGREE.sub(' độ ', text)

    text = normalize_acronyms(text)

    def _expand_version(m):
        return ' chấm '.join(m.group(1).split('.'))
    text = RE_VERSION.sub(_expand_version, text)

    text = RE_CLEAN_OTHERS.sub(' ', text)
    
    return text
