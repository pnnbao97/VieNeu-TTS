"""
Phonemization module for VieNeu-TTS.
Delegates all normalization and G2P logic to the sea-g2p library,
which provides a unified, tested, and maintained Vietnamese G2P pipeline.
"""

import functools
import logging
import re
from typing import Optional
from sea_g2p import SEAPipeline, G2P, Normalizer, punc_norm

logger = logging.getLogger("Vieneu.Phonemizer")

# Tách đoạn theo newline — dùng để normalize theo từng đoạn (ranh giới tự nhiên)
# trước khi chia chunk, giữ input cho regex backtracking ở mức an toàn.
RE_NEWLINE_SPLIT = re.compile(r"[\r\n]+")

# ---------------------------------------------------------------------------
# Inline non-verbal cues (emotion tokens) — v3 Turbo emotion checkpoint
# ---------------------------------------------------------------------------
# The emotion checkpoint was trained with three non-verbal cues embedded directly
# in the PHONEME stream as special tokens. In the *text* they appear as bracketed
# tags; phonemization must leave them as the matching <|emotion_k|> token instead
# of spelling the bracketed words out. The mapping + spacing reproduce the
# training data (cột `phones` của VieNeu-TTS-1000h-in-the-wild-coded) EXACTLY.
#
#   [chuckle]      / [cười]       -> <|emotion_1|>  (cười)
#   [sigh]         / [thở dài]    -> <|emotion_2|>  (thở dài)
#   [clear throat] / [hắng giọng] -> <|emotion_3|>  (hắng giọng)
_EMOTION_TAG_TO_K = {
    "chuckle": 1, "cười": 1, "cuoi": 1,
    "sigh": 2, "thở dài": 2, "tho dai": 2,
    "clear throat": 3, "hắng giọng": 3, "hang giong": 3,
}
# Split on a [bracketed tag] or an already-resolved <|emotion_k|> token.
_EMOTION_SPLIT_RE = re.compile(r"(\[[^\]]+\]|<\|emotion_\d+\|>)")
# Punctuation that stays attached to the preceding emotion token (no space),
# mirroring the training phones, e.g. "... <|emotion_2|>. ...".
_ATTACHING_PUNCT = set(".,!?;:…)]}\"'’”")


def _emotion_tag_token(tag: str) -> Optional[str]:
    """Map a raw ``[tag]`` / ``<|emotion_k|>`` string to its ``<|emotion_k|>`` form.

    Returns ``None`` for an unrecognized bracketed span (caller phonemizes it as
    ordinary text).
    """
    t = tag.strip()
    if t.startswith("<|"):
        return t  # already an explicit emotion token — pass through unchanged
    inner = t[1:-1].strip().lower()  # drop the surrounding [ ]
    k = _EMOTION_TAG_TO_K.get(inner)
    return f"<|emotion_{k}|>" if k is not None else None

# ---------------------------------------------------------------------------
# Always-punc_norm normalizer wrapper
# ---------------------------------------------------------------------------
# VieNeu-TTS LUÔN bật punc_norm (sea-g2p >= 0.7.6): câu ngắn (<5 từ) ép dấu cuối
# về ".", câu dài thiếu dấu kết thúc thì thêm ".". Dùng wrapper này thay cho
# sea_g2p.Normalizer ở mọi nơi để không phải truyền punc_norm=True rải rác và để
# bảo đảm hành vi "luôn luôn" kể cả ở các call-site tương lai.
class PuncNormalizer:
    """``sea_g2p.Normalizer`` với punc_norm mặc định True."""

    def __init__(self, lang: str = "vi") -> None:
        self._n = Normalizer(lang=lang)

    def normalize(self, text, punc_norm: bool = True):
        return self._n.normalize(text, punc_norm=punc_norm)

    def normalize_batch(self, texts, punc_norm: bool = True):
        return self._n.normalize_batch(texts, punc_norm=punc_norm)


# ---------------------------------------------------------------------------
# Shared singletons (instantiation is lazy-safe and thread-safe via GIL)
# ---------------------------------------------------------------------------
_pipeline: SEAPipeline = None
_g2p: G2P = None
_normalizer: PuncNormalizer = None

def _get_pipeline() -> SEAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SEAPipeline(lang="vi")
    return _pipeline

def _get_g2p() -> G2P:
    global _g2p
    if _g2p is None:
        _g2p = G2P(lang="vi")
    return _g2p

def _get_normalizer() -> PuncNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = PuncNormalizer()
    return _normalizer

# ---------------------------------------------------------------------------
# Public API  (same signatures as before — callers don't need to change)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _phonemize_cached(text: str, punc_norm: bool = True) -> str:
    """Cached single-text phonemization (normalize + G2P), punc_norm bật mặc định."""
    return _get_pipeline().run(text, punc_norm=punc_norm)


def phonemize_text(text: str) -> str:
    """Normalize and phonemize a single Vietnamese/bilingual text string."""
    return _phonemize_cached(text)


# Chốt dấu câu cuối cho MỘT chunk = đúng quy tắc punc_norm của sea-g2p (single
# source of truth, không tự viết lại): câu < 5 từ ép về một ".", câu dài thêm
# "." nếu chưa kết thúc bằng , . ! ?. Áp dụng đồng nhất cho chunk text & phoneme
# (kể cả chunk kết thúc bằng emotion token "<|emotion_k|>" -> "<|emotion_k|>.").


def phonemize_text_with_emotions(text: str) -> str:
    """Phonemize ``text`` while preserving inline non-verbal cues as emotion tokens.

    Same as :func:`phonemize_text`, but inline cues ``[cười]``/``[thở dài]``/
    ``[hắng giọng]`` (or the English ``[chuckle]``/``[sigh]``/``[clear throat]``,
    or an explicit ``<|emotion_k|>``) are kept as ``<|emotion_1|>``/``<|emotion_2|>``/
    ``<|emotion_3|>`` in the phoneme stream instead of being spelled out. Used by the
    v3 Turbo emotion checkpoint. Spacing matches the training data exactly: one
    space before the token, with following punctuation attached.
    """
    if "[" not in text and "<|emotion_" not in text:
        return _phonemize_cached(text)  # fast path: no cues → plain cached phonemize
    out = ""
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(text)):
        token = _emotion_tag_token(part) if i % 2 == 1 else None
        if token is not None:
            out = (out + " " + token) if out else token
            continue
        # Fragment giữa các emotion token: KHÔNG ép punc_norm để tránh chèn "."
        # vào giữa câu — giữ đúng spacing/format khớp dữ liệu train của checkpoint
        # emotion. (Toàn chunk đã được split sentence-aware trước đó.)
        ph = _phonemize_cached(part, punc_norm=False) if part and part.strip() else ""
        if not ph:
            continue
        if not out:
            out = ph
        elif ph[0] in _ATTACHING_PUNCT:
            out += ph          # punctuation attaches to the previous token/phones
        else:
            out += " " + ph
    # Chốt dấu cuối ở mức cả chunk (fragment bên trong đã punc_norm=False).
    return punc_norm(out)


def phonemize_batch(
    texts: list[str],
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
    **kwargs,
) -> list[str]:
    """
    Phonemize multiple texts with bilingual support.

    Args:
        texts:          List of input strings.
        skip_normalize: If True, assume the texts are already normalized
                        (i.e. only run G2P, not the normalizer).
        phoneme_dict:   Optional custom {word: phoneme} dict that overrides
                        the built-in dictionary for specific words.
    """
    if not texts:
        return []

    g2p = _get_g2p()

    # punc_norm LUÔN bật ở tầng G2P: kể cả text đã normalize sẵn (skip_normalize)
    # hay thiếu dấu câu, chuỗi phones vẫn kết thúc bằng "." hợp lệ.
    if skip_normalize:
        # Texts are pre-normalized — only run the G2P layer
        return g2p.phonemize_batch(texts, punc_norm=True, phoneme_dict=phoneme_dict)
    else:
        # Full pipeline: normalize (punc_norm) then G2P (punc_norm)
        normalizer = _get_normalizer()
        normalized = [normalizer.normalize(t) for t in texts]
        return g2p.phonemize_batch(normalized, punc_norm=True, phoneme_dict=phoneme_dict)


def phonemize_with_dict(
    text: str,
    phoneme_dict: dict = None,
    skip_normalize: bool = False,
) -> str:
    """
    Phonemize a single text, optionally with a custom word→phoneme mapping.

    When phoneme_dict is None and skip_normalize is False, the result is
    cached via lru_cache for performance.
    """
    if phoneme_dict is not None:
        # Custom dict supplied — skip cache to avoid cross-contamination
        return phonemize_batch(
            [text], skip_normalize=skip_normalize, phoneme_dict=phoneme_dict
        )[0]
    if skip_normalize:
        # punc_norm vẫn bật ở tầng G2P dù text đã normalize sẵn.
        return _get_g2p().phonemize_batch([text], punc_norm=True)[0]
    return _phonemize_cached(text)


def normalize_to_chunks(
    text: str,
    max_chars: int = 256,
    skip_normalize: bool = False,
) -> list[str]:
    """Normalize FIRST, then split the NORMALIZED text into <= max_chars chunks.

    Chia chunk SAU normalize. Normalizer mở rộng độ dài text (vd "100$" -> "một
    trăm u s d", "21/02/2025" -> "ngày hai mươi mốt tháng hai năm ..."), nên nếu
    cắt TRƯỚC khi norm thì chunk sẽ phình vượt ``max_chars`` sau khi chuẩn hóa.
    Ở đây normalize trước rồi mới cắt theo độ dài ĐÃ chuẩn hóa nên mỗi chunk thực
    sự <= ``max_chars``.

    Để không truyền nguyên một input cỡ DOCX vào bộ regex backtracking của
    normalizer, ta normalize theo từng ĐOẠN (tách theo newline) bằng
    ``normalize_batch`` — ranh giới tự nhiên, không ảnh hưởng độ dài chunk cuối —
    rồi mới gom lại và cắt. Mỗi chunk được chốt dấu câu cuối hợp lệ.
    """
    from vieneu_utils.core_utils import split_text_into_chunks

    if not text:
        return []

    if skip_normalize:
        normalized = text
    else:
        normalizer = _get_normalizer()
        paragraphs = [p for p in RE_NEWLINE_SPLIT.split(text) if p.strip()]
        normalized = (
            "\n".join(normalizer.normalize_batch(paragraphs, punc_norm=True))
            if paragraphs
            else ""
        )

    return [
        punc_norm(c)
        for c in split_text_into_chunks(normalized, max_chars=max_chars)
    ]


def chunk_phonemes(phonemes: str, max_chars: int = 256) -> list[str]:
    """Chia chuỗi phoneme (đã normalize + G2P, giữ inline cues) thành các chunk
    <= ``max_chars`` bằng CÁCH CHIA THƯỜNG (gộp câu tham lam, không vụn từng câu
    như ``split_into_chunks_v2``). Mỗi chunk được chốt dấu câu kết thúc hợp lệ
    (``.`` ``!`` ``?``) — kể cả khi một câu quá dài bị cắt theo cụm/từ.

    Dùng cho các đường phoneme-level (v3 Turbo, inline emotion) để chunk vừa
    không phình quá giới hạn vừa luôn kết thúc bằng dấu câu.
    """
    from vieneu_utils.core_utils import split_text_into_chunks

    if not phonemes:
        return []
    return [
        punc_norm(c)
        for c in split_text_into_chunks(phonemes, max_chars=max_chars)
    ]


def phonemize_to_chunks(
    text: str,
    max_chars: int = 256,
    min_chunk_size: int = 10,
    source_max_chars: Optional[int] = None,
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
):
    """
    Convert long raw text into bounded phoneme chunks.

    Thứ tự (chia chunk SAU normalize): normalize theo từng ĐOẠN (punc_norm=True)
    -> phonemize -> split ở tầng PHONEME (``split_into_chunks_v2``). Vì chunk
    được cắt sau khi đã normalize + phonemize nên độ dài chunk phản ánh đúng
    chuỗi phoneme thật và luôn <= ``max_chars`` (không bị phình như khi cắt trước
    norm). Mỗi chunk luôn có dấu câu kết thúc hợp lệ.

    Normalize theo đoạn (tách newline) giữ input cho bộ regex backtracking ở mức
    an toàn với văn bản cỡ lớn. ``source_max_chars`` được giữ cho tương thích chữ
    ký nhưng không còn dùng (việc cắt theo độ dài giờ làm ở tầng phoneme).
    """
    from vieneu_utils.core_utils import split_into_chunks_v2

    if not text:
        return []

    if skip_normalize:
        normalized_units = [text]
    else:
        normalizer = _get_normalizer()
        paragraphs = [p for p in RE_NEWLINE_SPLIT.split(text) if p.strip()] or [text]
        normalized_units = normalizer.normalize_batch(paragraphs, punc_norm=True)

    phonemes = phonemize_batch(
        normalized_units,
        skip_normalize=True,
        phoneme_dict=phoneme_dict,
    )

    phone_chunks = []
    for chunk_phonemes in phonemes:
        phone_chunks.extend(
            split_into_chunks_v2(
                chunk_phonemes,
                max_chunk_size=max_chars,
                min_chunk_size=min_chunk_size,
            )
        )
    return phone_chunks


# ---------------------------------------------------------------------------
# CLI helper (python -m vieneu_utils.phonemize_text "some text")
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    test_text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Giá SP500 hôm nay là 4.200,5 điểm."
    )
    print(f"Output: {phonemize_text(test_text)}")