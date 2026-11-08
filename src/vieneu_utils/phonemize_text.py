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


def _normalize_sentence_keep_cues(sentence: str, normalizer) -> str:
    """Normalize MỘT câu, giữ nguyên inline emotion cue thành ``<|emotion_k|>``."""
    if "[" not in sentence and "<|emotion_" not in sentence:
        return normalizer.normalize(sentence, punc_norm=False)
    parts = []
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(sentence)):
        if i % 2 == 1:                       # emotion tag
            tok = _emotion_tag_token(part)
            parts.append(tok if tok is not None else part)
        elif part.strip():
            parts.append(normalizer.normalize(part, punc_norm=False))
    return " ".join(p for p in parts if p)


def _normalized_sentences_by_para(
    text: str,
    skip_normalize: bool = False,
    keep_cues: bool = False,
) -> list[list[str]]:
    """Text thô -> list ĐOẠN, mỗi đoạn là list CÂU đã normalize.

    Thứ tự CỐ Ý là *tách câu trước, normalize sau*: normalizer xoá sạch dấu ngoặc
    (``"…"``/``(…)`` -> ``,``) nên nếu tách câu sau normalize thì dấu ``?`` trong
    câu trích dẫn bị nhầm là kết câu, đẻ ra mảnh vụn kiểu ``", đúng không anh?"``.
    Độ dài để đóng gói chunk vẫn được đo SAU normalize (ở
    :func:`~vieneu_utils.core_utils.pack_sentences_into_chunks`), nên chunk không
    phình khi normalizer nở text ("100$" -> "một trăm u s d").

    ``punc_norm`` KHÔNG bật ở tầng câu: luật "câu < 5 từ ép dấu cuối về ``.``" sẽ
    biến "Thế à?" thành "thế à." ngay giữa chunk, mất ngữ điệu hỏi. Dấu câu cuối
    chỉ được chốt một lần ở tầng CHUNK.
    """
    from vieneu_utils.core_utils import split_into_sentences

    paragraphs = [p for p in RE_NEWLINE_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    normalizer = None if skip_normalize else _get_normalizer()
    out: list[list[str]] = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        if not sentences:
            continue
        if skip_normalize:
            out.append(sentences)
        elif keep_cues:
            out.append([_normalize_sentence_keep_cues(s, normalizer) for s in sentences])
        else:
            out.append(normalizer.normalize_batch(sentences, punc_norm=False))
    return out


def normalize_to_chunks(
    text: str,
    max_chars: int = 256,
    skip_normalize: bool = False,
) -> list[str]:
    """Split text thành chunks <= max_chars, cắt ở ranh giới CÂU.

    Tách câu trên text THÔ (nhận biết ngoặc/trích dẫn) -> normalize từng câu ->
    đóng gói theo độ dài ĐÃ chuẩn hoá -> chốt dấu câu cuối mỗi chunk. Xem
    :func:`_normalized_sentences_by_para` để biết vì sao thứ tự phải là vậy.

    Normalize theo từng câu cũng giữ input cho bộ regex backtracking của
    normalizer ở mức an toàn với văn bản cỡ DOCX.
    """
    from vieneu_utils.core_utils import pack_sentences_into_chunks

    if not text:
        return []

    chunks: list[str] = []
    for sentences in _normalized_sentences_by_para(text, skip_normalize=skip_normalize):
        chunks.extend(pack_sentences_into_chunks(sentences, max_chars=max_chars))
    return [punc_norm(c) for c in chunks]


def normalize_to_chunks_v3(
    text: str, max_chars: int = 256, min_chunk_chars: int = 20
) -> list[str]:
    """Chia chunk cho đường v3 GIỐNG HỆT v2-gpu: cắt theo độ dài TEXT ĐÃ normalize.

    Đường v3 trước đây cắt ở tầng PHONEME (``chunk_phonemes``), mà phoneme dài hơn
    text ~1.2-1.4x nên cùng ``max_chars`` lại cắt vụn hơn v2-gpu. Hàm này cắt theo
    text-length như ``normalize_to_chunks`` (v2-gpu) để số chunk khớp nhau, ĐỒNG
    THỜI giữ inline emotion cue (``[cười]``/``<|emotion_k|>`` -> ``<|emotion_k|>``)
    mà ``normalize_to_chunks`` thường sẽ nuốt mất (dấu ``[...]`` bị xoá khi normalize).

    Chunk ngắn hơn ``min_chunk_chars`` được gộp vào chunk kề (xem
    :func:`_merge_short_chunks`) — chunk 1-2 từ đứng riêng làm model AR dễ ảo giác.

    Trả về list TEXT chunk (mỗi chunk có thể chứa ``<|emotion_k|>``); caller
    phonemize từng chunk bằng :func:`phonemize_text_with_emotions`.
    """
    return normalize_to_chunks_v3_with_gaps(
        text, max_chars=max_chars, min_chunk_chars=min_chunk_chars
    )[0]


_EMOTION_TOKEN_RE = re.compile(r"<\|emotion_\d+\|>")


def _effective_len(chunk: str) -> int:
    """Độ dài "đọc được" của chunk: emotion token chiếm ký tự nhưng không phải chữ."""
    return len(_EMOTION_TOKEN_RE.sub("", chunk).strip())


def _merge_short_chunks(
    chunks: list[str], gaps: list[str], min_chars: int
) -> tuple[list[str], list[str]]:
    """Gộp chunk ngắn hơn ``min_chars`` vào chunk kề, cập nhật ``gaps`` tương ứng.

    Chunk 1-2 từ (thường là dòng tiêu đề/thoại cụt sau khi tách theo newline) đứng
    một mình làm model AR thiếu điều kiện text, stop token bắn trượt -> ảo giác.
    Gộp ưu tiên: ranh giới KHÔNG phải ngắt đoạn trước, rồi hàng xóm ngắn hơn, rồi
    bên phải (tiêu đề dính vào nội dung theo sau nghe tự nhiên hơn). Khi buộc phải
    gộp xuyên ngắt đoạn, khoảng nghỉ ``para`` 0.35s của ranh giới đó nhường chỗ cho
    ngắt câu do dấu chấm quyết định — đánh đổi có chủ đích để tránh ảo giác.

    Chunk gộp có thể vượt ``max_chars`` tối đa ``min_chars + 1`` ký tự. Chunk đơn
    độc (toàn văn bản quá ngắn) giữ nguyên — trần frame theo phoneme lo phần đó.
    """
    while len(chunks) > 1:
        short = [i for i, c in enumerate(chunks) if _effective_len(c) < min_chars]
        if not short:
            break
        i = min(short, key=lambda k: _effective_len(chunks[k]))
        sides = []                                   # (ưu_tiên, side)
        if i < len(chunks) - 1:
            sides.append(((gaps[i] != "para", -len(chunks[i + 1])), "R"))
        if i > 0:
            sides.append(((gaps[i - 1] != "para", -len(chunks[i - 1])), "L"))
        side = max(sides, key=lambda t: t[0])[1]     # hoà -> "R" (đứng trước)
        if side == "R":
            chunks[i] = chunks[i] + " " + chunks.pop(i + 1)
            gaps.pop(i)
        else:
            chunks[i - 1] = chunks[i - 1] + " " + chunks.pop(i)
            gaps.pop(i - 1)
    return chunks, gaps


def normalize_to_chunks_v3_with_gaps(
    text: str, max_chars: int = 256, min_chunk_chars: int = 20
) -> tuple[list[str], list[str]]:
    """Như :func:`normalize_to_chunks_v3` nhưng trả kèm loại ranh giới GIỮA các
    chunk để ghép audio nghỉ dài/ngắn theo ngữ cảnh.

    Trả về ``(chunks, gaps)`` với ``gaps[i] in {"para","sentence","minor"}`` là
    ranh giới giữa ``chunks[i]`` và ``chunks[i+1]`` (``len(gaps) == len(chunks)-1``).
    Dùng với :func:`vieneu_utils.core_utils.gaps_to_silence` +
    ``join_audio_chunks(..., silence_ps=...)``.

    Ranh giới ``sentence``/``minor`` được phân loại LẠI trên chunk ĐÃ ``punc_norm``
    (punc_norm có thể ép dấu cuối cho câu ngắn) để khớp intonation audio thật;
    ``para`` (ngắt đoạn) giữ nguyên — kể cả trên đường có emotion cue.
    """
    from vieneu_utils.core_utils import pack_sentences_into_chunks, _classify_gap

    if not text:
        return [], []

    keep_cues = "[" in text or "<|emotion_" in text
    chunks: list[str] = []
    gaps: list[str] = []
    for sentences in _normalized_sentences_by_para(text, keep_cues=keep_cues):
        para_chunks = pack_sentences_into_chunks(sentences, max_chars=max_chars)
        if not para_chunks:
            continue
        if chunks:                   # ranh giới với đoạn TRƯỚC đó là ngắt đoạn
            gaps.append("para")
        for j, ch in enumerate(para_chunks):
            if j > 0:                # ranh giới trong CÙNG đoạn: phân loại lại sau
                gaps.append("sentence")
            chunks.append(ch)

    chunks = [punc_norm(c) for c in chunks]
    gaps = [g if g == "para" else _classify_gap(chunks[i]) for i, g in enumerate(gaps)]
    # Gộp SAU punc_norm: từng mảnh đã chốt dấu câu cuối nên text gộp giữ nguyên
    # ngữ điệu ("Chương một. Nội dung..."); gap của các ranh giới còn lại không đổi.
    return _merge_short_chunks(chunks, gaps, min_chunk_chars)


_SILENT_TAG_RE = re.compile(r"\[silent\b([^\[\]\r\n]*)\]", re.IGNORECASE)
_SILENT_DURATION_RE = re.compile(
    r"(?:([0-9]{1,5})\s*ms|([0-9]{1,2}(?:\.[0-9]{1,3})?)\s*s)", re.IGNORECASE
)
_MAX_SILENT_MS = 60_000


def split_silence_tags(text: str) -> tuple[list[str], list[int]]:
    """Split raw text around ``[silent …]`` tags and return their durations in ms."""
    parts = _SILENT_TAG_RE.split(text or "")
    durations = []
    total_ms = 0
    for raw in parts[1::2]:
        match = _SILENT_DURATION_RE.fullmatch(raw.strip())
        if match is None:
            raise ValueError("Invalid pause tag. Use [silent 200ms] or [silent 2s].")
        ms, seconds = match.groups()
        duration_ms = int(ms) if ms is not None else round(float(seconds) * 1000)
        total_ms += duration_ms
        if duration_ms == 0 or total_ms > _MAX_SILENT_MS:
            raise ValueError("Pause tags must be at least 1ms each and at most 60s in total.")
        durations.append(duration_ms)
    return parts[::2], durations


def plan_v3_chunks(text: str, max_chars: int = 256) -> tuple[list[str], list[str], dict[int, int]]:
    """Plan speech chunks, default gaps and explicit silence before each chunk index.

    An explicit pause at index ``len(chunks)`` goes after the last chunk. Splitting
    before normalization keeps durations out of the phonemes and prevents short
    chunks from being merged across an explicit pause.
    """
    parts, durations = split_silence_tags(text)
    if not durations:
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        return chunks, gaps, {}

    chunks, gaps, pauses = [], [], {}
    for i, part in enumerate(parts):
        if i > 0:
            index = len(chunks)
            pauses[index] = pauses.get(index, 0) + durations[i - 1]
        # Ignore empty and punctuation-only fragments between or after tags.
        if not re.search(r"\w", part):
            continue
        section, section_gaps = normalize_to_chunks_v3_with_gaps(part, max_chars=max_chars)
        if section:
            if chunks:
                gaps.append("sentence")  # replaced by the explicit pause at this boundary
            chunks.extend(section)
            gaps.extend(section_gaps)
    return chunks, gaps, pauses


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
