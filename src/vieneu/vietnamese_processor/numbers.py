"""Pure number-to-words and Roman-numeral helpers.

No regex, no I/O — these are reusable building blocks for the regex stages.
"""
from __future__ import annotations

DIGITS: dict[str, str] = {
    str(i): w
    for i, w in enumerate(
        ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    )
}

_TEENS: dict[int, str] = {
    10: "mười",
    11: "mười một",
    12: "mười hai",
    13: "mười ba",
    14: "mười bốn",
    15: "mười lăm",
    16: "mười sáu",
    17: "mười bảy",
    18: "mười tám",
    19: "mười chín",
}

ORDINAL_MAP: dict[str, str] = {
    "1": "nhất",
    "2": "hai",
    "3": "ba",
    "4": "tư",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
    "10": "mười",
}

ROMAN_VALUES: dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
}

_BIG_UNITS = [
    (1_000_000_000, "tỷ"),
    (1_000_000, "triệu"),
    (1_000, "nghìn"),
]


def number_to_words(num_str: str) -> str:
    """Convert a numeric string into spoken Vietnamese.

    Handles negatives, leading zeros, and values up to billions.
    Falls back to digit-by-digit reading on parse failure so the pipeline
    never raises on malformed input.
    """
    num_str = num_str.lstrip("0") or "0"
    if num_str.startswith("-"):
        return "âm " + number_to_words(num_str[1:])

    try:
        n = int(num_str)
    except ValueError:
        return num_str

    if n == 0:
        return "không"
    if n < 10:
        return DIGITS[str(n)]
    if n < 20:
        return _TEENS[n]

    if n < 100:
        t, u = divmod(n, 10)
        tens_w = DIGITS[str(t)] + " mươi"
        if u == 0:
            return tens_w
        if u == 1:
            return tens_w + " mốt"
        if u == 4:
            return tens_w + " tư"
        if u == 5:
            return tens_w + " lăm"
        return tens_w + " " + DIGITS[str(u)]

    if n < 1000:
        h, rem = divmod(n, 100)
        head = DIGITS[str(h)] + " trăm"
        if rem == 0:
            return head
        if rem < 10:
            return head + " lẻ " + DIGITS[str(rem)]
        return head + " " + number_to_words(str(rem))

    for div, name in _BIG_UNITS:
        if n >= div:
            q, rem = divmod(n, div)
            head = number_to_words(str(q)) + " " + name
            if rem == 0:
                return head
            if rem < 10:
                return head + " không trăm lẻ " + DIGITS[str(rem)]
            if rem < 100:
                return head + " không trăm " + number_to_words(str(rem))
            return head + " " + number_to_words(str(rem))

    # Fallback (shouldn't reach here for finite ints)
    return " ".join(DIGITS.get(d, d) for d in num_str)


def roman_to_arabic(s: str) -> int | None:
    """Return the Arabic value of a Roman numeral string, or None.

    Bounded to [1, 30] because the regex pipeline only uses Roman numerals
    for chapters/sections — wider values are typically false positives
    (e.g. acronyms like 'IV' meaning intravenous).
    """
    if not s or not all(c in ROMAN_VALUES for c in s):
        return None
    total = 0
    prev = 0
    for c in reversed(s):
        v = ROMAN_VALUES[c]
        total += v if v >= prev else -v
        prev = v
    if total < 1 or total > 30:
        return None
    return total


__all__ = [
    "DIGITS",
    "ORDINAL_MAP",
    "ROMAN_VALUES",
    "number_to_words",
    "roman_to_arabic",
]
