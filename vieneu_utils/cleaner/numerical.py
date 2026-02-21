import re
from .num2vi import n2w, n2w_single
from .symbols import vietnamese_set

_normal_number_re        = r"[\d]+"
_float_number_re         = r"[\d]+[,]{1}[\d]+"
_number_with_one_dot     = r"[\d]+[.]{1}[\d]{3}"
_number_with_two_dot     = r"[\d]+[.]{1}[\d]{3}[.]{1}[\d]{3}"
_number_with_three_dot   = r"[\d]+[.]{1}[\d]{3}[.]{1}[\d]{3}[.]{1}[\d]{3}"
_number_with_one_space   = r"[\d]+[\s]{1}[\d]{3}"
_number_with_two_space   = r"[\d]+[\s]{1}[\d]{3}[\s]{1}[\d]{3}"
_number_with_three_space = r"[\d]+[\s]{1}[\d]{3}[\s]{1}[\d]{3}[\s]{1}[\d]{3}"

_number_combined = (
    r"("
    + _float_number_re + "|"
    + _number_with_three_dot + "|"
    + _number_with_two_dot + "|"
    + _number_with_one_dot + "|"
    + _number_with_three_space + "|"
    + _number_with_two_space + "|"
    + _number_with_one_space + "|"
    + _normal_number_re
    + r")"
)

_number_re    = r"(.)(-{1})?" + _number_combined
_number_re_start = r"^(-{1})?" + _number_combined

_multiply_re     = r"(" + _normal_number_re + r")(x|\sx\s)(" + _normal_number_re + r")"
_ordinal_pattern = r"(thứ|hạng)(\s)(1|4)"
_phone_re        = r"((\+84|84|0|0084)(3|5|7|8|9)[0-9]{8})"

def _normalize_dot_sep(number: str) -> str:
    if re.fullmatch(r"\d+(\.\d{3})+", number):
        return number.replace(".", "")
    return number

def _num_to_words(number: str, negative: bool = False) -> str:
    number = _normalize_dot_sep(number).replace(" ", "")
    if "," in number:
        parts = number.split(",")
        return n2w(parts[0]) + " phẩy " + n2w(parts[1])
    elif negative:
        return "âm " + n2w(number)
    return n2w(number)

def _expand_number(match):
    prefix, negative_symbol, number = match.groups(0)
    negative = (negative_symbol == "-")
    word = _num_to_words(number, negative)
    prefix_str = "" if prefix in (0, None) else prefix
    return prefix_str + " " + word + " "

def _expand_number_start(match):
    negative_symbol, number = match.groups(0)
    negative = (negative_symbol == "-")
    return _num_to_words(number, negative) + " "

def _expand_phone(match):
    return n2w_single(match.group(0).strip())

def _expand_ordinal(match):
    prefix, space, number = match.groups(0)
    if number == "1": return prefix + space + "nhất"
    if number == "4": return prefix + space + "tư"
    return prefix + space + n2w(number)

def _expand_multiply_number(match):
    n1, _, n2 = match.groups(0)
    return n2w(n1) + " nhân " + n2w(n2)

def normalize_number_vi(text):
    text = re.sub(_ordinal_pattern, _expand_ordinal, text)
    text = re.sub(_multiply_re, _expand_multiply_number, text)
    text = re.sub(_phone_re, _expand_phone, text)
    text = re.sub(_number_re_start, _expand_number_start, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
