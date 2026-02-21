import re
from .num2vi import n2w
from .symbols import vietnamese_re, vietnamese_for_date_re

day_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_date_seperator = r"(\/|-|\.)"

_full_date_pattern = r"\b(\d{1,2})" + _date_seperator + r"(\d{1,2})" + _date_seperator + r"(\d{4})\b"
_day_month_pattern = r"\b(\d{1,2})" + _date_seperator + r"(\d{1,2})\b"
_month_year_pattern = r"\b(\d{1,2})" + _date_seperator + r"(\d{4})\b"

_full_time_pattern = r"\b(\d{1,2})(g|:|h)(\d{1,2})(p|:|m)(\d{1,2})(s|g)?\b"
_time_pattern = r"\b(\d{1,2})(g|:|h)(\d{1,2})(p|m)?\b"

def _is_valid_date(day, month):
    try:
        day, month = int(day), int(month)
        return 1 <= month <= 12 and 1 <= day <= day_in_month[month - 1]
    except: return False

def _expand_full_date(match):
    day, sep1, month, sep2, year = match.groups()
    if _is_valid_date(day, month):
        day = str(int(day))
        month = str(int(month))
        return f"ngày {n2w(day)} tháng {n2w(month)} năm {n2w(year)}"
    return match.group(0)

def _expand_day_month(match):
    day, sep, month = match.groups()
    if _is_valid_date(day, month):
        day = str(int(day))
        month = str(int(month))
        return f"ngày {n2w(day)} tháng {n2w(month)}"
    return match.group(0)

def _norm_time_part(s):
    return '0' if s == '00' else s

def _expand_time(match):
    h, sep, m, suffix = match.groups()
    if 0 <= int(h) < 24 and 0 <= int(m) < 60:
        return f"{n2w(_norm_time_part(h))} giờ {n2w(_norm_time_part(m))} phút"
    return match.group(0)

def normalize_date(text):
    text = re.sub(_full_date_pattern, _expand_full_date, text, flags=re.IGNORECASE)
    text = re.sub(
        _month_year_pattern,
        lambda m: f"tháng {n2w(str(int(m.group(1))))} năm {n2w(m.group(3))}",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(_day_month_pattern, _expand_day_month, text, flags=re.IGNORECASE)
    text = re.sub(r'\bngày\s+ngày\b', 'ngày', text, flags=re.IGNORECASE)
    return text

def normalize_time(text):
    text = re.sub(
        _full_time_pattern,
        lambda m: f"{n2w(str(int(m.group(1))))} giờ {n2w(str(int(m.group(3))))} phút {n2w(str(int(m.group(5))))} giây",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(_time_pattern, _expand_time, text, flags=re.IGNORECASE)
    return text
