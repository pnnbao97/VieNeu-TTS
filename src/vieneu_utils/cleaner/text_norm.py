import re
from .num2vi import n2w, n2w_single

_vi_letter_names = {
    "a": "a",
    "b": "bê",
    "c": "xê",
    "d": "đê",
    "đ": "đê",
    "e": "e",
    "ê": "ê",
    "f": "ép",
    "g": "gờ",
    "h": "hát",
    "i": "i",
    "j": "giây",
    "k": "ca",
    "l": "lờ",
    "m": "mờ",
    "n": "nờ",
    "o": "o",
    "ô": "ô",
    "ơ": "ơ",
    "p": "pê",
    "q": "qui",
    "r": "rờ",
    "s": "ét",
    "t": "tê",
    "u": "u",
    "ư": "ư",
    "v": "vê",
    "w": "vê kép",
    "x": "ích",
    "y": "y",
    "z": "dét",
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
    "km": "ki lô mét",
    "dm": "đê xi mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "nm": "na nô mét",
    "µm": "mic rô mét",
    "μm": "mic rô mét",
    "m": "mét",
    "kg": "ki lô gam",
    "g": "gam",
    "mg": "mi li gam",
    "km2": "ki lô mét vuông",
    "m2": "mét vuông",
    "cm2": "xen ti mét vuông",
    "mm2": "mi li mét vuông",
    "ha": "héc ta",
    "km3": "ki lô mét khối",
    "m3": "mét khối",
    "cm3": "xen ti mét khối",
    "mm3": "mi li mét khối",
    "l": "lít",
    "dl": "đê xi lít",
    "ml": "mi li lít",
    "hl": "héc tô lít",
    "kw": "ki lô oát",
    "mw": "mê ga oát",
    "gw": "gi ga oát",
    "kwh": "ki lô oát giờ",
    "mwh": "mê ga oát giờ",
    "wh": "oát giờ",
    "hz": "héc",
    "khz": "ki lô héc",
    "mhz": "mê ga héc",
    "ghz": "gi ga héc",
    "pa": "pát cal",
    "kpa": "ki lô pát cal",
    "mpa": "mê ga pát cal",
    "bar": "ba",
    "mbar": "mi li ba",
    "atm": "át mốt phia",
    "psi": "pi ét xai",
    "j": "giun",
    "kj": "ki lô giun",
    "cal": "ca lo",
    "kcal": "ki lô ca lo",
    "h": "giờ",
    "p": "phút",
    "s": "giây",
    "sqm": "mét vuông",
    "cum": "mét khối",
    "gb": "gi ga bai",
    "mb": "mê ga bai",
    "kb": "ki lô bai",
    "tb": "tê ra bai",
    "db": "đê xi ben",
    "oz": "ao xơ",
    "lb": "pao",
    "lbs": "pao",
    "ft": "phít",
    "in": "ins",
    "dpi": "đi phi ai",
    "pH": "pê hát",
}

_currency_key = {"usd": "đô la Mỹ", "vnd": "đồng", "đ": "đồng", "euro": "ơ rô", "%": "phần trăm"}

_letter_key_vi = _vi_letter_names

_acronyms_exceptions_vi = {
    "CĐV": "cổ động viên",
    "HĐND": "hội đồng nhân dân",
    "TAND": "toàn án nhân dân",
    "BHXH": "bảo hiểm xã hội",
    "BHTN": "bảo hiểm thất nghiệp",
    "TP.HCM": "thành phố hồ chí minh",
    "VN": "việt nam",
    "UBND": "uỷ ban nhân dân",
    "TP": "thành phố",
    "HCM": "hồ chí minh",
    "HN": "hà nội",
    "BTC": "ban tổ chức",
    "CLB": "câu lạc bộ",
    "HTX": "hợp tác xã",
    "NXB": "nhà xuất bản",
    "TW": "trung ương",
    "CSGT": "cảnh sát giao thông",
    "LHQ": "liên hợp quốc",
    "THCS": "trung học cơ sở",
    "THPT": "trung học phổ thông",
    "ĐH": "đại học",
    "HLV": "huấn luyện viên",
    "GS": "giáo sư",
    "TS": "tiến sĩ",
    "TNHH": "trách nhiệm hữu hạn",
    "VĐV": "vận động viên",
    "TPHCM": "thành phố hồ chí minh",
    "PGS": "phó giáo sư",
}

# Compiled Regular Expressions
RE_ROMAN_NUMBER = re.compile(r"\b(?=[IVXLCDM]{2,})M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b")
RE_LETTER = re.compile(r"(chữ|chữ cái|kí tự|ký tự)\s+(['\"]?)([a-z])(['\"]?)\b", re.IGNORECASE)
RE_STANDALONE_LETTER = re.compile(r"\b([a-zA-Z])\b(\.?)")
RE_URL = re.compile(r"\b(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=]+\b")
RE_SLASH_NUMBER = re.compile(r"\b(\d+)/(\d+)\b")
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*\.[A-Za-z]{2,}\b")
RE_SENTENCE_SPLIT = re.compile(r"([.!?]+(?:\s+|$))")
RE_ACRONYM = re.compile(r"\b[A-Z0-9]{2,}\b")
RE_ALPHANUMERIC = re.compile(r"\b(\d+)([a-zA-Z])\b")
RE_BRACKETS = re.compile(r"[(\[{]\s*([^[\](){}]+?)\s*[)\]}]")
RE_STRIP_BRACKETS = re.compile(r"[\[\]\(\)\{\}]")
RE_TEMP_C_NEG = re.compile(r"-(\d+(?:[.,]\d+)?)\s*°\s*c\b", re.IGNORECASE)
RE_TEMP_F_NEG = re.compile(r"-(\d+(?:[.,]\d+)?)\s*°\s*f\b", re.IGNORECASE)
RE_TEMP_C = re.compile(r"(\d+(?:[.,]\d+)?)\s*°\s*c\b", re.IGNORECASE)
RE_TEMP_F = re.compile(r"(\d+(?:[.,]\d+)?)\s*°\s*f\b", re.IGNORECASE)
RE_DEGREE = re.compile(r"°")
RE_VERSION = re.compile(r"\b(\d+(?:\.\d+)+)\b")
RE_CLEAN_OTHERS = re.compile(r"[^\w\s.,!?;:@%_]")
RE_CLEAN_QUOTES = re.compile(r'["\'“”‘’]')

# Reusable patterns for measurement/currency
_MAGNITUDE_P = r"\s*(?P<mag>tỷ|triệu|nghìn|ngàn)?\s*"
_NUMERIC_P = r"(?P<num>\d+(?:[.,]\d+)*)"

# Pre-compiled regex for compound units
RE_COMPOUND_UNIT = re.compile(rf"\b(?P<num>\d+(?:[.,]\d+)*)?\s*(?P<u1>[a-zμµ²³°]+)/(?P<u2>[a-zμµ²³°0-9]+)\b", re.IGNORECASE)

# Pre-compiled currency patterns
RE_CURRENCY_PREFIX_USD = re.compile(rf"\$\s*{_NUMERIC_P}{_MAGNITUDE_P}", re.IGNORECASE)
RE_CURRENCY_SUFFIX_USD = re.compile(rf"{_NUMERIC_P}{_MAGNITUDE_P}\$", re.IGNORECASE)
RE_PERCENTAGE = re.compile(rf"(?P<num>\d+(?:[.,]\d+)*)\s*%", re.IGNORECASE)

# Pre-compile measurement and currency unit patterns
_MEASUREMENT_PATTERNS = []
for unit, full in sorted(_measurement_key_vi.items(), key=lambda x: len(x[0]), reverse=True):
    pattern = re.compile(rf"\b{_NUMERIC_P}{_MAGNITUDE_P}{re.escape(unit)}\b", re.IGNORECASE)

    standalone_pattern = None
    safe_standalone = ["km", "cm", "mm", "kg", "mg", "m2", "km2", "usd", "vnd", "mhz", "khz", "ghz", "hz", "ph"]
    if unit.lower() in safe_standalone:
        standalone_pattern = re.compile(rf"(?<![\d.,])\b{unit}\b", re.IGNORECASE)

    _MEASUREMENT_PATTERNS.append((pattern, standalone_pattern, full))

_CURRENCY_PATTERNS = []
for unit, full in _currency_key.items():
    if unit == "%":
        continue
    pattern = re.compile(rf"\b{_NUMERIC_P}{_MAGNITUDE_P}{unit}\b", re.IGNORECASE)
    _CURRENCY_PATTERNS.append((pattern, full))

# Pre-compile acronyms exceptions
_ACRONYMS_EXCEPTIONS_RE = [(re.compile(rf"\b{re.escape(k)}\b"), v) for k, v in _acronyms_exceptions_vi.items()]

_ROMAN_NUMERALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_ABBRS = {"v.v": " vân vân", "v/v": " về việc", "ko": " không", "đ/c": "địa chỉ"}

_SYMBOLS_MAP = {
    "&": " và ",
    "+": " cộng ",
    "=": " bằng ",
    "#": " thăng ",
    ">": " lớn hơn ",
    "<": " nhỏ hơn ",
    "≥": " lớn hơn hoặc bằng ",
    "≤": " nhỏ hơn hoặc bằng ",
    "±": " cộng trừ ",
    "≈": " xấp xỉ ",
}


def _expand_number_with_sep(num_str):
    if not num_str:
        return ""
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
        num = m.group("num")
        mag = m.group("mag") if "mag" in m.groupdict() and m.group("mag") else ""
        expanded_num = _expand_number_with_sep(num)
        return f"{expanded_num} {mag} {full}".replace("  ", " ").strip()

    for pattern, standalone_pattern, full in _MEASUREMENT_PATTERNS:
        # Case with number
        text = pattern.sub(lambda m, f=full: _repl(m, f), text)

        # Standalone units
        if standalone_pattern:
            text = standalone_pattern.sub(f" {full} ", text)
    return text


def expand_currency(text):
    def _repl(m, full):
        num = m.group("num")
        mag = m.group("mag") if "mag" in m.groupdict() and m.group("mag") else ""
        expanded_num = _expand_number_with_sep(num)
        return f"{expanded_num} {mag} {full}".replace("  ", " ").strip()

    text = RE_CURRENCY_PREFIX_USD.sub(lambda m: _repl(m, "đô la Mỹ"), text)
    text = RE_CURRENCY_SUFFIX_USD.sub(lambda m: _repl(m, "đô la Mỹ"), text)
    text = RE_PERCENTAGE.sub(lambda m: f"{_expand_number_with_sep(m.group('num'))} phần trăm", text)

    for pattern, full in _CURRENCY_PATTERNS:
        text = pattern.sub(lambda m, f=full: _repl(m, f), text)
    return text


def expand_compound_units(text):
    def _repl_compound(m):
        num_str = m.group("num") if m.group("num") else ""
        num = _expand_number_with_sep(num_str)
        u1 = m.group("u1").lower()
        u2 = m.group("u2").lower()
        full1 = _measurement_key_vi.get(u1, u1)
        full2 = _measurement_key_vi.get(u2, u2)
        res = f" {full1} trên {full2} "
        if num:
            res = f"{num} {res}"
        return res

    text = RE_COMPOUND_UNIT.sub(_repl_compound, text)
    return text


def expand_roman(match):
    num = match.group(0).upper()
    if not num:
        return ""
    result = 0
    for i, c in enumerate(num):
        if (i + 1) == len(num) or _ROMAN_NUMERALS[c] >= _ROMAN_NUMERALS[num[i + 1]]:
            result += _ROMAN_NUMERALS[c]
        else:
            result -= _ROMAN_NUMERALS[c]
    return f" {n2w(str(result))} "


def expand_letter(match):
    prefix, q1, char, q2 = match.groups()
    if char.lower() in _letter_key_vi:
        return f"{prefix} {_letter_key_vi[char.lower()]} "
    return match.group(0)


def expand_abbreviations(text):
    for k, v in _ABBRS.items():
        text = text.replace(k, v)
    return text


def expand_standalone_letters(text):
    def _repl_letter(m):
        char = m.group(1).lower()
        dot = m.group(2) if m.group(2) else ""
        if char in _letter_key_vi:
            return f" {_letter_key_vi[char]}{dot} "
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
            elif char == ".":
                res.append("chấm")
            elif char == "/":
                res.append("xẹt")
            elif char == ":":
                res.append("hai chấm")
            elif char == "-":
                res.append("gạch ngang")
            elif char == "_":
                res.append("gạch dưới")
            elif char == "?":
                res.append("hỏi")
            elif char == "&":
                res.append("và")
            elif char == "=":
                res.append("bằng")
            else:
                res.append(char)
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
        parts = email.split("@")
        if len(parts) != 2:
            return email

        user_part, domain_part = parts

        # User part: spell out
        user_norm = []
        for char in user_part.lower():
            if char.isalnum():
                if char.isdigit():
                    user_norm.append(n2w_single(char))
                else:
                    user_norm.append(_vi_letter_names.get(char, char))
            elif char == ".":
                user_norm.append("chấm")
            elif char == "_":
                user_norm.append("gạch dưới")
            elif char == "-":
                user_norm.append("gạch ngang")
            else:
                user_norm.append(char)

        # Domain part
        domain_part_lower = domain_part.lower()
        if domain_part_lower in _common_email_domains:
            domain_norm = _common_email_domains[domain_part_lower]
        else:
            domain_parts = domain_part.split(".")
            norm_domain_parts = []
            for dp in domain_parts:
                dp_norm = []
                for char in dp.lower():
                    if char.isalnum():
                        if char.isdigit():
                            dp_norm.append(n2w_single(char))
                        else:
                            dp_norm.append(_vi_letter_names.get(char, char))
                    else:
                        dp_norm.append(char)
                norm_domain_parts.append(" ".join(dp_norm))
            domain_norm = " chấm ".join(norm_domain_parts)

        return " ".join(user_norm) + " a còng " + domain_norm

    return RE_EMAIL.sub(_repl_email, text)


WORD_LIKE_ACRONYMS = {
    "UNESCO",
    "NASA",
    "NATO",
    "ASEAN",
    "OPEC",
    "SARS",
    "FIFA",
    "UNIC",
    "RAM",
    "VRAM",
    "COVID",
    "IELTS",
    "STEM",
    "SWAT",
    "SEAL",
    "WASP",
    "COBOL",
    "BASIC",
    "OLED",
    "COVAX",
    "BRICS",
    "APEC",
    "VUCA",
    "PERMA",
    "DINK",
    "MENA",
    "EPIC",
    "OASIS",
    "BASE",
    "DART",
    "IDEA",
    "CHAOS",
    "SMART",
    "FANG",
}


# AT&T
def normalize_acronyms(text):
    sentences = RE_SENTENCE_SPLIT.split(text)
    processed = []
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        sep = sentences[i + 1] if i + 1 < len(sentences) else ""
        if not s:
            processed.append(sep)
            continue

        words = s.split()
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        is_all_caps = len(alpha_words) > 0 and all(w.isupper() for w in alpha_words)

        if not is_all_caps:

            def _repl_acronym(m):
                word = m.group(0)
                if word.isdigit():
                    return word
                if word in WORD_LIKE_ACRONYMS:
                    return f"__START_EN__{word.lower()}__END_EN__"
                if any(c.isdigit() for c in word):
                    if word.upper() == "B2B":
                        return "__START_EN__b two b__END_EN__"

                    res = []
                    for c in word.lower():
                        if c.isdigit():
                            res.append(n2w_single(c))
                        else:
                            res.append(_vi_letter_names.get(c, c))
                    return " ".join(res)

                spaced_word = " ".join(c.lower() for c in word if c.isalnum())
                if spaced_word:
                    return f"__START_EN__{spaced_word}__END_EN__"
                return word

            s = RE_ACRONYM.sub(_repl_acronym, s)

        processed.append(s + sep)
    return "".join(processed)


def expand_alphanumeric(text):
    def _repl(m):
        num = m.group(1)
        char = m.group(2).lower()
        if char in _letter_key_vi:
            pronunciation = _letter_key_vi[char]
            # Special case for roads (Quốc lộ 1D -> Quốc lộ 1 đê)
            if char == "d" and ("quốc lộ" in text.lower() or "ql" in text.lower()):
                pronunciation = "đê"
            return f"{num} {pronunciation}"
        return m.group(0)

    return RE_ALPHANUMERIC.sub(_repl, text)


def expand_symbols(text):
    for s, v in _SYMBOLS_MAP.items():
        text = text.replace(s, v)
    return text


def expand_temperatures(text):
    text = RE_TEMP_C_NEG.sub(r"âm \1 độ xê", text)
    text = RE_TEMP_F_NEG.sub(r"âm \1 độ ép", text)
    text = RE_TEMP_C.sub(r"\1 độ xê", text)
    text = RE_TEMP_F.sub(r"\1 độ ép", text)
    text = RE_DEGREE.sub(" độ ", text)
    return text


def normalize_others(text):
    """
    Apply various normalization rules that don't fit into specific categories.
    This function is called by clean_vietnamese_text.
    """
    # 1. Expand acronym exceptions and basic patterns
    for pattern, v in _ACRONYMS_EXCEPTIONS_RE:
        text = pattern.sub(v, text)

    text = normalize_urls(text)
    text = normalize_emails(text)
    text = normalize_slashes(text)

    # 2. Expand Roman numerals and special letter patterns
    text = RE_ROMAN_NUMBER.sub(expand_roman, text)
    text = RE_LETTER.sub(expand_letter, text)
    text = expand_alphanumeric(text)

    # 3. Clean quotes and expand general symbols
    text = RE_CLEAN_QUOTES.sub("", text)
    text = expand_symbols(text)

    # 4. Handle brackets and temperatures
    # Note: Measurement/Currency expansion is handled by clean_vietnamese_text caller
    text = RE_BRACKETS.sub(r", \1, ", text)
    text = RE_STRIP_BRACKETS.sub(" ", text)
    text = expand_temperatures(text)

    # 5. Normalize acronyms (spell out or tag with <en>)
    text = normalize_acronyms(text)

    # 6. Expand version numbers (e.g., 1.2.3 -> 1 chấm 2 chấm 3)
    text = RE_VERSION.sub(lambda m: " chấm ".join(m.group(1).split(".")), text)

    # 7. Final cleanup of any remaining unsupported characters
    text = RE_CLEAN_OTHERS.sub(" ", text)

    # Restore internal <en> tags
    text = text.replace("__START_EN__", "<en>").replace("__END_EN__", "</en>")

    return text
