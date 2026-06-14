"""Vietnamese text preprocessor for TTS pipelines.

Public API:
    process_vietnamese_text(text)   — one-shot helper, drop-in replacement
                                      for the legacy module-level function.
    Pipeline                        — composable stage runner.
    Stage data exposed for tests:
        NON_VIETNAMESE_WORDS, ACRONYMS, ENGLISH_BLACKLIST, UNIT_MAP

Optimisation notes:
    Every regex is compiled exactly once at import time. The non-Vietnamese
    transliteration step (17k+ entries) is now a single precompiled
    alternation rather than a per-row `re.sub` loop, yielding large speedups
    on real workloads.
"""
from __future__ import annotations

from .data import (
    ACRONYMS,
    ACRONYMS_BUILTIN,
    ACRONYMS_CSV,
    ENGLISH_BLACKLIST,
    NON_VIETNAMESE_WORDS,
    UNIT_MAP,
)
from .numbers import (
    DIGITS,
    ORDINAL_MAP,
    ROMAN_VALUES,
    number_to_words,
    roman_to_arabic,
)
from .pipeline import (
    DEFAULT_STAGES,
    Pipeline,
    StageDef,
    process_vietnamese_text,
)

__version__ = "2.0.0"

__all__ = [
    "process_vietnamese_text",
    "Pipeline",
    "StageDef",
    "DEFAULT_STAGES",
    "number_to_words",
    "roman_to_arabic",
    "DIGITS",
    "ORDINAL_MAP",
    "ROMAN_VALUES",
    "NON_VIETNAMESE_WORDS",
    "ACRONYMS",
    "ACRONYMS_CSV",
    "ACRONYMS_BUILTIN",
    "UNIT_MAP",
    "ENGLISH_BLACKLIST",
]
