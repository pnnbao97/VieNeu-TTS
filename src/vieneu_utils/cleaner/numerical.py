import re
from .num2vi import n2w, n2w_single

_normal_number_re = r"[\d]+"
_float_number_re = r"[\d]+[,]{1}[\d]+"
_number_with_one_dot = r"[\d]+[.]{1}[\d]{3}"
_number_with_two_dot = r"[\d]+[.]{1}[\d]{3}[.]{1}[\d]{3}"
_number_with_three_dot = r"[\d]+[.]{1}[\d]{3}[.]{1}[\d]{3}[.]{1}[\d]{3}"
_number_with_one_space = r"[\d]+[\s]{1}[\d]{3}"
_number_with_two_space = r"[\d]+[\s]{1}[\d]{3}[\s]{1}[\d]{3}"
_number_with_three_space = r"[\d]+[\s]{1}[\d]{3}[\s]{1}[\d]{3}[\s]{1}[\d]{3}"

_number_combined = (
    r"("
    + _float_number_re
    + "|"
    + _number_with_three_dot
    + "|"
    + _number_with_two_dot
    + "|"
    + _number_with_one_dot
    + "|"
    + _number_with_three_space
    + "|"
    + _number_with_two_space
    + "|"
    + _number_with_one_space
    + "|"
    + _normal_number_re
    + r")"
)

# Compiled Regular Expressions
RE_NUMBER = re.compile(r"(\D)(-{1})?" + _number_combined)
RE_NUMBER_START = re.compile(r"^(-{1})?" + _number_combined, re.MULTILINE)
RE_MULTIPLY = re.compile(r"(" + _normal_number_re + r")(x|\sx\s)(" + _normal_number_re + r")")
RE_ORDINAL = re.compile(r"(thứ|hạng)(\s)(1|4)")
RE_PHONE = re.compile(r"((\+84|84|0|0084)(3|5|7|8|9)[0-9]{8})")
RE_DOT_SEP = re.compile(r"\d+(\.\d{3})+")


def _normalize_dot_sep(number: str) -> str:
    if RE_DOT_SEP.fullmatch(number):
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
    negative = negative_symbol == "-"
    word = _num_to_words(number, negative)
    prefix_str = "" if prefix in (0, None) else prefix
    return prefix_str + " " + word + " "


def _expand_number_start(match):
    negative_symbol, number = match.groups()
    negative = negative_symbol == "-"
    return _num_to_words(number, negative) + " "


def _expand_phone(match):
    return n2w_single(match.group(0).strip())


def _expand_ordinal(match):
    prefix, space, number = match.groups(0)
    if number == "1":
        return prefix + space + "nhất"
    if number == "4":
        return prefix + space + "tư"
    return prefix + space + n2w(number)


def _expand_multiply_number(match):
    n1, _, n2 = match.groups(0)
    return n2w(n1) + " nhân " + n2w(n2)


def normalize_number_vi(text):
    text = RE_ORDINAL.sub(_expand_ordinal, text)
    text = RE_MULTIPLY.sub(_expand_multiply_number, text)
    text = RE_PHONE.sub(_expand_phone, text)
    # 1. Start of string OR start of line (handling newlines preserved by normalization)
    text = RE_NUMBER_START.sub(_expand_number_start, text)
    # 2. Anywhere else (preceded by a non-digit)
    text = RE_NUMBER.sub(_expand_number, text)
    return text
