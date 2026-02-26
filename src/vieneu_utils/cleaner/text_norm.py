import re
from .num2vi import n2w, n2w_single
from .symbols import vietnamese_re, vietnamese_without_num_re

_en_letter_names = {
    "a": "ây", "b": "bi", "c": "xi", "d": "đi", "e": "y", "f": "ép", "g": "gi",
    "h": "êch", "i": "ai", "j": "chây", "k": "kê", "l": "eo", "m": "em", "n": "en",
    "o": "ô", "p": "pi", "q": "kiu", "r": "a", "s": "ét", "t": "ti", "u": "yu",
    "v": "vi", "w": "dắp bờ liu", "x": "ích", "y": "quai", "z": "zi",
    "0": "di râu", "1": "uăn", "2": "tu", "3": "tri", "4": "pho", "5": "fai",
    "6": "sích", "7": "xe vừn", "8": "eit", "9": "nai"
}

_vi_letter_names = {
    "a": "a", "b": "bờ", "c": "xê", "d": "dê", "đ": "đê", "e": "e", "ê": "ê",
    "f": "ép", "g": "gờ", "h": "hát", "i": "i", "j": "chây", "k": "ca", "l": "lờ",
    "m": "mờ", "n": "nờ", "o": "o", "ô": "ô", "ơ": "ơ", "p": "pê", "q": "kiu",
    "r": "rờ", "s": "ét", "t": "ti", "u": "u", "ư": "ư", "v": "vi", "w": "vê kép",
    "x": "ích", "y": "i dài", "z": "dét"
}

_common_email_domains = {
    "gmail.com": "gờ meo chấm com",
    "yahoo.com": "da hu chấm com",
    "yahoo.com.vn": "da hu chấm com chấm vê nờ",
    "outlook.com": "aut lúc chấm com",
    "hotmail.com": "hót meo chấm com",
    "icloud.com": "ai clao chấm com",
    "fpt.vn": "ép pê tê chấm vê nờ",
    "fpt.com.vn": "ép pê tê chấm com chấm vê nờ",
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
    "h": "giờ", "p": "phút", "s": "giây"
}

_currency_key = {
    "usd": "đô la Mỹ", "vnd": "đồng", "đ": "đồng", "euro": "ơ rô", "%": "phần trăm"
}

_letter_key_vi = {
    "a": "a", "b": "bê", "c": "xê", "d": "dê", "đ": "đê", "f": "ép", "g": "gờ", "h": "hát",
    "i": "i", "j": "chây", "k": "ca", "l": "lờ", "m": "mờ", "n": "nờ", "o": "o",
    "p": "pê", "q": "kiu", "r": "rờ", "s": "ét", "t": "ti", "u": "u", "v": "vi", "w": "vê kép", "x": "ít", "y": "y", "z": "dét"
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
    magnitude_p = r"\s*(tỷ|triệu|nghìn|ngàn)?\s*"
    numeric_p = r"((?:\d+[.,])*\d+)"
    
    def _repl(m, full):
        num = m.group(1)
        mag = m.group(2) if m.group(2) else ""
        expanded_num = _expand_number_with_sep(num)
        return f"{expanded_num} {mag} {full}".replace("  ", " ").strip()
    
    for unit, full in sorted(_measurement_key_vi.items(), key=lambda x: len(x[0]), reverse=True):
        # Case with number
        pattern = rf"\b{numeric_p}{magnitude_p}{unit}\b"
        text = re.sub(pattern, lambda m, f=full: _repl(m, f), text, flags=re.IGNORECASE)
        
        # Standalone units
        safe_standalone = [
            "km", "cm", "mm", "kg", "mg",
            "m2", "km2", "usd", "vnd",
            "mhz", "khz", "ghz", "hz"
        ]
        if unit.lower() in safe_standalone:
            # First try with standard word boundaries
            text = re.sub(rf"(?<![\d.,])\b{unit}\b", f" {full} ", text, flags=re.IGNORECASE)
    return text

def expand_currency(text):
    magnitude_p = r"\s*(tỷ|triệu|nghìn|ngàn)?\s*"
    numeric_p = r"((?:\d+[.,])*\d+)"
    
    def _repl(m, full):
        num = m.group(1)
        mag = m.group(2) if m.group(2) else ""
        expanded_num = _expand_number_with_sep(num)
        return f"{expanded_num} {mag} {full}".replace("  ", " ").strip()
        
    text = re.sub(rf"\$\s*{numeric_p}{magnitude_p}", lambda m: _repl(m, "đô la Mỹ"), text)
    text = re.sub(rf"{numeric_p}{magnitude_p}\$", lambda m: _repl(m, "đô la Mỹ"), text)
    text = re.sub(rf"{numeric_p}\s*%", lambda m: f"{_expand_number_with_sep(m.group(1))} phần trăm", text)
    
    for unit, full in _currency_key.items():
        if unit == "%": continue
        text = re.sub(rf"\b{numeric_p}{magnitude_p}{unit}\b", lambda m, f=full: _repl(m, f), text, flags=re.IGNORECASE)
    return text

def expand_compound_units(text):
    numeric_p = r"((?:\d+[.,])*\d+)"
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

    pattern = rf"{numeric_p}?\s*\b([a-zμµ²³°]+)/([a-zμµ²³°0-9]+)\b"
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
    
    # Match a standalone letter (optionally followed by a dot)
    # Using word boundaries to avoid matching letters inside words.
    return re.sub(r'\b([a-zA-Z])\b\.?', _repl_letter, text)

def normalize_emails(text):
    email_re = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

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

    return re.sub(email_re, _repl_email, text)

def normalize_acronyms(text):
    # Split into sentences to respect the "whole sentence is uppercase" rule
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)
    processed = []
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        sep = sentences[i+1] if i+1 < len(sentences) else ""
        if not s:
            processed.append(sep)
            continue

        # A sentence is "all uppercase" if all words with letters are uppercase
        words = s.split()
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        is_all_caps = len(alpha_words) > 0 and all(w.isupper() for w in alpha_words)

        if not is_all_caps:
            def _repl_acronym(m):
                word = m.group(0)
                if word.isdigit(): return word
                return " ".join(_en_letter_names.get(c.lower(), c) for c in word)

            # Match 2+ uppercase letters/digits, must contain at least one uppercase letter
            s = re.sub(r'\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,}\b', _repl_acronym, s)

        processed.append(s + sep)
    return "".join(processed)

def normalize_others(text):
    for k, v in _acronyms_exceptions_vi.items():
        text = re.sub(rf"\b{k}\b", v, text)
    
    text = normalize_emails(text)

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

    # Acronyms should be handled before converting version numbers like 1.2.3
    text = normalize_acronyms(text)

    def _expand_version(m):
        return ' chấm '.join(m.group(0).split('.'))
    text = re.sub(r'\b\d+(?:\.\d+)+\b', _expand_version, text)

    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữợỳýỷỹỵđ.,!?;:@%_]', ' ', text)
    
    return text
