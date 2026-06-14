"""High-level pipeline orchestration for the Vietnamese preprocessor.

The default pipeline reproduces the legacy 12-step ordering exactly. The
`Pipeline` class lets advanced callers add, remove, or reorder stages for
benchmarking or domain-specific tweaks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from . import stages

Stage = Callable[[str], str]

logger = logging.getLogger(__name__)

_TRAILING_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_SYMBOL_RE = re.compile(r"[±×÷^+=@#%]")
_ROMAN_RE = re.compile(r"\b[IVXLCDM]+\b")


@dataclass(frozen=True)
class StageDef:
    """A named, callable pipeline step."""
    name: str
    fn: Stage


# Default pipeline (mirrors the legacy `process_vietnamese_text` order exactly).
DEFAULT_STAGES: tuple[StageDef, ...] = (
    StageDef("clean", stages.clean_text),
    StageDef("acronyms", stages.replace_acronyms),
    StageDef("non_vietnamese", stages.replace_non_vietnamese),
    StageDef("thousand_sep", stages.remove_thousand_separators),
    StageDef("vn_currency_abbrev", stages.convert_vn_currency_abbrev),
    StageDef("ranges_with_units", stages.convert_ranges_with_units),
    StageDef("negative", stages.convert_negative),
    StageDef("dot_decimal", stages.convert_dot_decimal),
    StageDef("pct_after_word", stages.convert_pct_after_word),
    StageDef("quarter", stages.convert_quarter),
    StageDef("percentage", stages.convert_percentage),
    StageDef("currency", stages.convert_currency),
    StageDef("date", stages.convert_date),
    StageDef("year_range", stages.convert_year_range),
    StageDef("time", stages.convert_time),
    StageDef("phone", stages.convert_phone),
    StageDef("measurement_units", stages.convert_measurement_units),
    StageDef("plusminus", stages.convert_plusminus),
    StageDef("symbols", stages.convert_symbols),
    StageDef("multiplier", stages.convert_multiplier),
    StageDef("ratio", stages.convert_ratio),
    StageDef("medical_ratio", stages.convert_medical_ratio),
    StageDef("alphanum_codes", stages.convert_alphanum_codes),
    StageDef("decimal", stages.convert_decimal),
    StageDef("ordinal", stages.convert_ordinal),
    StageDef("roman", stages.convert_roman),
    StageDef("standalone_numbers", stages.convert_standalone_numbers),
)


class Pipeline:
    """Composable text-preprocessing pipeline.

    Examples
    --------
    >>> Pipeline.default().process("Doanh thu quý ba đạt 1.5 tỷ đồng.")
    'Doanh thu quý ba đạt một phẩy năm tỷ đồng.'

    Drop a stage:

    >>> p = Pipeline.default().without("non_vietnamese")
    >>> "non_vietnamese" not in [s.name for s in p.stages]
    True
    """

    def __init__(self, steps: Iterable[StageDef]):
        self.stages: tuple[StageDef, ...] = tuple(steps)
        self._by_name = {s.name: s for s in self.stages}

    @classmethod
    def default(cls) -> "Pipeline":
        return cls(DEFAULT_STAGES)

    def without(self, *names: str) -> "Pipeline":
        return Pipeline(s for s in self.stages if s.name not in names)

    def stage(self, name: str) -> Stage:
        return self._by_name[name].fn

    def process(self, text: str) -> str:
        if not text or not text.strip():
            return text
        original = text

        has_roman = bool(_ROMAN_RE.search(text))
        has_digits = bool(_DIGIT_RE.search(text)) or has_roman
        has_symbols = bool(_SYMBOL_RE.search(text))

        for step in self.stages:
            name = step.name

            # Skip numeric stages if no digits present
            if not has_digits and name in (
                "thousand_sep", "vn_currency_abbrev", "ranges_with_units",
                "negative", "dot_decimal", "quarter", "percentage",
                "currency", "date", "year_range", "time", "phone",
                "measurement_units", "multiplier", "ratio", "medical_ratio",
                "alphanum_codes", "decimal", "ordinal", "standalone_numbers"
            ):
                continue

            # Skip symbol stages if no symbols present
            if not has_symbols and name in (
                "plusminus", "symbols", "pct_after_word", "percentage"
            ):
                continue

            # Skip Roman numeral conversion if no Roman letters present
            if not has_roman and name == "roman":
                continue

            text = step.fn(text)

        text = _TRAILING_WS_RE.sub(" ", text).strip()
        if text != original and logger.isEnabledFor(logging.DEBUG):
            logger.debug('[VN] "%s" → "%s"', original, text)
        return text


# Pre-built default instance — process_vietnamese_text() reuses it so callers
# don't pay re-construction cost on every request.
_DEFAULT = Pipeline.default()


def process_vietnamese_text(text: str) -> str:
    """Backward-compatible top-level entry point.

    Calls the default pipeline. For custom orderings, build a `Pipeline`
    yourself.
    """
    return _DEFAULT.process(text)


__all__ = ["Pipeline", "StageDef", "DEFAULT_STAGES", "process_vietnamese_text"]
