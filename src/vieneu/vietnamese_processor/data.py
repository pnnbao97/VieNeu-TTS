"""Data-file loaders for the Vietnamese preprocessor.

CSVs and blacklists live in `data/` next to (or one level above) the package.
The loaders are lenient — a missing file logs a warning and returns an empty
container, so a partial deploy never crashes the pipeline.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Override hook: respected first, then a list of conventional locations.
_ENV_VAR = "VIETNAMESE_PROCESSOR_DATA_DIR"

_PKG_DIR = Path(__file__).resolve().parent
_CANDIDATE_DIRS: list[Path] = [
    _PKG_DIR / "data",          # vietnamese_processor/data/
    _PKG_DIR.parent / "data",   # services/tts/vietnamese-processor/data/
    _PKG_DIR.parent,            # legacy: csv next to vietnamese_processor.py
]


def data_dir() -> Path:
    """Return the first existing data directory.

    Honours the `VIETNAMESE_PROCESSOR_DATA_DIR` env var for tests / deploys
    that ship data files separately from the package.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        logger.warning("%s=%s does not exist, falling back", _ENV_VAR, override)

    for d in _CANDIDATE_DIRS:
        if d.is_dir():
            return d
    # Last-ditch: return the package dir; loaders will warn on missing files.
    return _PKG_DIR


def _resolve(filename: str) -> Path:
    """Find `filename` under any candidate dir; return the first hit."""
    override = os.environ.get(_ENV_VAR)
    candidates: Iterable[Path] = (
        ([Path(override) / filename] if override else [])
        + [d / filename for d in _CANDIDATE_DIRS]
    )
    for p in candidates:
        if p.is_file():
            return p
    return _PKG_DIR / filename  # nominal path even if missing


def load_csv_kv(filename: str, *, lowercase_keys: bool = True) -> dict[str, str]:
    """Load a 2-column CSV (key,value) into a dict.

    Skips the header row by using `csv.DictReader`. The first column is the
    key, the second column is the value — column names are not required to
    match anything specific.

    Set ``lowercase_keys=False`` for data where case is significant (e.g. SI
    units like ``mmHg``, ``GB``, ``kHz``).
    """
    path = _resolve(filename)
    out: dict[str, str] = {}
    if not path.is_file():
        logger.warning("data file missing: %s", path)
        return out
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = list(row.values())
            if len(values) < 2:
                continue
            key, val = values[0], values[1]
            if key is None or val is None:
                continue
            key = key.strip()
            if lowercase_keys:
                key = key.lower()
            val = val.strip()
            if key:
                out[key] = val
    return out


def load_lines(filename: str) -> set[str]:
    """Load a newline-delimited file into a set of stripped, lowercased lines.

    Blank lines and `#` comments are ignored.
    """
    path = _resolve(filename)
    out: set[str] = set()
    if not path.is_file():
        logger.warning("data file missing: %s", path)
        return out
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.lower())
    return out


# ── Module-level singletons ─────────────────────────────────────────────
# CSV-backed
NON_VIETNAMESE_WORDS: dict[str, str] = load_csv_kv("non-vietnamese-words.csv")
ACRONYMS_CSV: dict[str, str] = load_csv_kv("acronyms.csv")
ACRONYMS_BUILTIN: dict[str, str] = load_csv_kv("acronyms_builtin.csv")
UNIT_MAP: dict[str, str] = load_csv_kv("units.csv", lowercase_keys=False)

# Plaintext-backed
ENGLISH_BLACKLIST: set[str] = load_lines("english_blacklist.txt")

# Compose acronym dict: builtin defaults are overridden by repo CSV.
ACRONYMS: dict[str, str] = dict(ACRONYMS_BUILTIN)
ACRONYMS.update(ACRONYMS_CSV)

# Drop blacklisted words from the transliteration dict so espeak handles
# them naturally (fixes "post"→"bờ-lốc" etc).
for _w in ENGLISH_BLACKLIST:
    NON_VIETNAMESE_WORDS.pop(_w, None)


__all__ = [
    "data_dir",
    "load_csv_kv",
    "load_lines",
    "NON_VIETNAMESE_WORDS",
    "ACRONYMS",
    "ACRONYMS_CSV",
    "ACRONYMS_BUILTIN",
    "UNIT_MAP",
    "ENGLISH_BLACKLIST",
]
