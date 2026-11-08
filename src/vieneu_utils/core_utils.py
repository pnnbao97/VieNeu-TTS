import re
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ─── Regex ───────────────────────────────────────────────────────────────────

RE_NEWLINE          = re.compile(r'[\r\n]+')  # dùng chung cho cả v1 và v2
RE_SENTENCE_FINDALL = re.compile(r'[^.!?]+[.!?]*|[.!?]+')

# Một "từ" để đóng gói chunk: coi NGUYÊN một thẻ <en>...</en> là một token không
# thể tách (thẻ chứa khoảng trắng như "<en>u s d</en>" sẽ vỡ nếu .split() theo
# space). Dùng khi chia text ĐÃ normalize (có chèn <en>) thành chunk.
RE_TOKEN_KEEP_EN = re.compile(r'<en>.*?</en>|\S+', re.IGNORECASE | re.DOTALL)


def _tokenize_keep_en(s: str) -> List[str]:
    """Tách ``s`` thành token, giữ NGUYÊN mỗi cụm ``<en>...</en>``."""
    return RE_TOKEN_KEEP_EN.findall(s)

# v1 only
RE_SENTENCE_END = re.compile(r'(?<=[\.\!\?\…])\s+')
RE_MINOR_PUNCT  = re.compile(r'(?<=[\,\;\:\-\–\—])\s+')

# ─── Tách câu nhận biết ngoặc/trích dẫn ──────────────────────────────────────
# Dấu kết câu nằm BÊN TRONG một cặp ngoặc/trích dẫn KHÔNG phải ranh giới câu:
#   Có phải ... kiểu như: "Rồi sao nữa? Mình phải làm đến bao giờ?", đúng không anh?
# là MỘT câu, không phải ba. Cắt theo regex thuần (RE_SENTENCE_END) sẽ vỡ câu này
# thành mảnh, mảnh cuối ", đúng không anh?" mở đầu bằng dấu phẩy — không phải câu.
#
# Cố tình BỎ nháy đơn ' và ’ khỏi danh sách: chúng trùng với dấu lược trong
# "don't" / "l’ordre", sẽ mở ngoặc mà không bao giờ đóng và nuốt phần còn lại.
_OPEN_TO_CLOSE = {
    '(': ')', '[': ']', '{': '}',
    '“': '”', '‘': '’', '«': '»', '‹': '›', '「': '」', '『': '』',
}
_OPENERS = frozenset(_OPEN_TO_CLOSE)
_CLOSERS = frozenset(_OPEN_TO_CLOSE.values())
_SYMMETRIC_QUOTE = '"'   # cùng một ký tự vừa mở vừa đóng -> dùng cờ bật/tắt
_SENT_END_CHARS  = frozenset('.!?…')
# Dấu đóng bám NGAY SAU dấu kết câu vẫn thuộc về câu đó: `bao giờ?"` , `(thế à!)`
_TRAILING_CLOSE = _CLOSERS | frozenset('"\'’”')

# v2 noise cleanup
_NOISE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'([.!?])[.,;:]+'), r'\1'),
    (re.compile(r'[.,;:]+([.!?])'), r'\1'),
    (re.compile(r'\s+[,;]\s+'),     ' '),
    (re.compile(r' {2,}'),          ' '),
]
_MULTI_PUNCT = re.compile(r'([.!?])\s*[.!?]+')

# ─── Data class ──────────────────────────────────────────────────────────────

@dataclass
class PhoneChunk:
    text: str
    is_sentence_end: bool  # True = kết thúc câu thật | False = cắt nhân tạo

# ─── Audio utils ─────────────────────────────────────────────────────────────

# Khoảng nghỉ TỐI THIỂU (giây) giữa hai chunk tuỳ RANH GIỚI đã cắt: ngắt đoạn
# (xuống dòng) nghỉ dài nhất, hết câu (.!?) nghỉ vừa, ngắt trong câu (,;: hoặc cắt cưỡng
# bức) nghỉ ngắn. Đây là khoảng nghỉ THẬT nghe được (im lặng đuôi chunk trước +
# zeros chèn + im lặng đầu chunk sau): ``join_audio_chunks`` giữ nguyên audio
# model sinh và chỉ chèn zeros khi tổng im lặng ở khe chưa đủ con số này; đuôi
# tự nhiên dài hơn thì giữ nguyên. Lịch sử: bảng gốc (0.35/0.18/0.04) là phần
# chèn THÊM không đo đuôi, nên khoảng nghỉ thật phụ thuộc giọng (preset v3 Turbo
# tự phát ~300 ms im lặng trước EOS, giọng clone hầu như không -> chunk "đè"
# nhau); v3.5.1 cắt mép rồi bù cho đúng bằng bảng (0.55/0.32/0.14) — cắt mép bị
# bỏ 09/2026 vì làm mất đuôi tự nhiên mà không cần thiết (phần cắt < -45 dB).
# Giá trị hiện tại theo khoảng nghỉ model TỰ sinh khi cả câu nằm trong một chunk
# (đo trên 23 preset v3 Turbo, 09/2026): phẩy ~0.30–0.45 s, hết câu ~0.50–0.65 s.
V3_GAP_SILENCE = {"para": 0.70, "sentence": 0.50, "minor": 0.30}

# Cắt/fade mép chunk (``trim_and_fade``): còn dùng ở engine v3 Nano cho đầu ra flow
# model; join_audio_chunks đường v3 Turbo KHÔNG cắt mép nữa (09/2026).
EDGE_THRESH_DB = -45.0   # ngưỡng "có tiếng" trên envelope mean|x| cửa sổ 10 ms
EDGE_KEEP_S = 0.04       # im lặng giữ lại mỗi đầu sau khi cắt
EDGE_FADE_S = 0.015      # fade cosine ở hai mép để không click

# Trần số frame audio hợp lý cho MỘT chunk theo độ dài phoneme — chặn-trên đối
# xứng với guard chặn-dưới MIN_FRAMES_PER_PHONE=0.25 bên v3_turbo_serve.engine.
# Đo trên dataset pnnbao-ump/vi-tts-v3-finetune-mix (129,790 rows, 2026-08):
# frames/ký-tự-phoneme p50=0.53, p99=1.10, p99.9=1.50; trần 24 + 2.0*len phủ
# 99.9915% rows (11/129,790 vượt). Row rất ngắn có ratio cao vì chi phí cố định
# (bin len<15: max 25 frames) — phần đó nằm trong slack. Chunk ngắn (1-2 từ) hay
# bắn trượt stop token rồi "nói thêm" — trần này cắt cụt phần bịa thay vì để
# chạy hết max_new_frames (300); chunk dài bình thường có trần > 300 nên không
# bị ảnh hưởng. Markup (<en>…</en>, <|emotion_k|>) chiếm ký tự nhưng không tốn
# frame nên bị loại trước khi đo.
MAX_FRAMES_PER_PHONE = 2.0
_FRAME_CAP_SLACK = 24            # frame trừ hao cho lead-in / chi phí cố định
_FRAME_MARKUP_RE = re.compile(r"<\|emotion_\d+\|>|</?en>")

# ── Mã hoá clip tham chiếu: đuôi lẻ (issue #198) ────────────────────────────
# MOSS codec: 1 frame = 3840 mẫu @ 48 kHz. Clip không tròn frame thì encoder tự đệm
# frame cuối và frame ấy LUÔN ra codebook-0 = 455 — không phải im lặng (482) mà là
# một tiếng ngắn nghe rõ (RMS ~0.03 so với đuôi thật ~0.001). Nó dính vào ref_codes
# của mọi giọng nhân bản và vào nhãn của 100% hàng fine-tune, mô hình học thuộc và
# nhả ra ở cuối câu. Sửa: đệm zero cho tròn frame TRƯỚC khi mã hoá (frame cuối thành
# mã thật của phần đuôi + im lặng) và ép số frame = số mẫu / 3840 (đường ONNX luôn
# trả thêm một frame). Mã đã lưu từ trước thì bỏ frame 455 ở đuôi khi nạp.
CODEC_SAMPLES_PER_FRAME = 3840
ENCODER_PAD_CODE = 455


def pad_to_codec_frame(wav, sr: int = 48_000):
    """Đệm zero ở đuôi để độ dài là bội số của một frame codec (chỉ áp @ 48 kHz)."""
    n = len(wav)
    if sr != 48_000 or n % CODEC_SAMPLES_PER_FRAME == 0:
        return wav
    return np.concatenate([np.asarray(wav, dtype=np.float32),
                           np.zeros(CODEC_SAMPLES_PER_FRAME - n % CODEC_SAMPLES_PER_FRAME, np.float32)])


def strip_encoder_pad_frame(codes):
    """Bỏ frame đuôi do encoder đệm (codebook-0 == 455) khỏi mã ``(T, n_vq)`` đã lưu
    bằng bản cũ. Frame thật hiếm khi mang mã này, và mất 80 ms ngữ cảnh tham chiếu
    thì vô hại — còn giữ lại thì mọi lượt sinh đều thấy một tiếng lạ ở cuối ngữ cảnh."""
    if codes is None:
        return None
    arr = np.asarray(codes)
    if arr.ndim == 2 and arr.shape[0] > 1 and int(arr[-1, 0]) == ENCODER_PAD_CODE:
        return arr[:-1]
    return arr


# Codec chạy 12.5 frame/giây. Chunk rất ngắn thì công thức tuyến tính theo phoneme
# vẫn quá hào phóng ("chào" -> 40 frame = 3.2s toàn phần bịa), nên chặn thêm một
# trần theo SỐ TIẾNG (âm tiết): 13 frame (~1s) cho 1 tiếng, +5 frame mỗi tiếng
# thêm, áp cho chunk <= SYLLABLE_CAP_MAX_SYL tiếng. Đếm theo âm tiết chứ KHÔNG
# theo số từ: "notification" là một từ nhưng 5 âm tiết, đọc gần 1s — trần 13
# frame cố định cho "một từ" sẽ cắt cụt nó (xem syllable_count). Đo chunk ngắn
# (2026-09): 1 tiếng 6-9 frame, 2 tiếng 6-10, 3-4 tiếng 12-15 — trần này còn dư
# >= 1.5x. Không áp khi có emotion cue (tiếng cười/thở dài tốn frame thật).
SINGLE_WORD_MAX_FRAMES = 13      # trần cho chunk 1 tiếng (~1s @ 12.5 frame/s)
SYLLABLE_CAP_PER_EXTRA = 5       # +frame cho mỗi tiếng thêm
SYLLABLE_CAP_MAX_SYL = 4         # chunk dài hơn dùng công thức theo phoneme
_SINGLE_WORD_MAX_PHONES = 24     # phoneme tối đa hợp lý cho MỘT tiếng


def is_cue_only(phonemes: str) -> bool:
    """Chunk chỉ gồm emotion cue ("[cười]", "[thở dài]"...), không có tiếng nào."""
    ph = phonemes or ""
    return "<|emotion_" in ph and not any(ch.isalpha() for ch in _FRAME_MARKUP_RE.sub("", ph))


def max_expected_frames(phonemes: str) -> int:
    """Số frame TỐI ĐA hợp lý cho chunk có chuỗi ``phonemes`` này."""
    stripped = _FRAME_MARKUP_RE.sub("", phonemes or "")
    eff_len = len(stripped)
    cap = _FRAME_CAP_SLACK + int(np.ceil(MAX_FRAMES_PER_PHONE * eff_len))
    if is_cue_only(phonemes):
        # Cue đứng một mình: đo 48 lần sinh (2026-09) tiếng cười / thở dài / hắng
        # giọng tự nhiên dài 5-12 frame; ca trượt EOS nhảy thẳng lên 19-60 frame.
        # Xử lý như chunk 1 tiếng: trần ~1s, chạm trần => sinh lại (babble_suspect).
        return min(cap, SINGLE_WORD_MAX_FRAMES)
    if "<|emotion_" not in (phonemes or ""):
        syl = max(1, syllable_count(phonemes))      # chuỗi rỗng / không nguyên âm -> coi như 1 tiếng
        # Một "tiếng" dài bất thường (> _SINGLE_WORD_MAX_PHONES phoneme mỗi tiếng) là
        # do normalize dính chữ, không phải tiếng thật -> để công thức thường lo.
        if syl <= SYLLABLE_CAP_MAX_SYL and eff_len <= _SINGLE_WORD_MAX_PHONES * syl:
            cap = min(cap, SINGLE_WORD_MAX_FRAMES + SYLLABLE_CAP_PER_EXTRA * (syl - 1))
    return cap


def gaps_to_silence(gaps: List[str]) -> List[float]:
    """Map list loại-ranh-giới -> list TỔNG khoảng nghỉ (giây) cho ``join_audio_chunks``."""
    return [V3_GAP_SILENCE.get(g, V3_GAP_SILENCE["sentence"]) for g in gaps]


def edge_silence(
    wav: np.ndarray, sr: int, thresh_db: float = EDGE_THRESH_DB, win_s: float = 0.01
) -> Tuple[int, int]:
    """``(lead, tail)``: số mẫu im lặng ở đầu và cuối ``wav`` (envelope mean|x| theo
    cửa sổ ``win_s``, dưới ``thresh_db`` là im lặng). Wav toàn im lặng -> ``(len, 0)``."""
    n_samp = int(wav.size)
    win = max(1, int(win_s * sr))
    n_win = n_samp // win
    if n_win == 0:
        return n_samp, 0
    env = np.abs(wav[: n_win * win]).reshape(n_win, win).mean(1)
    above = np.flatnonzero(env > 10 ** (thresh_db / 20))
    if not above.size:
        return n_samp, 0
    return int(above[0]) * win, n_samp - (int(above[-1]) + 1) * win


def trim_and_fade(
    wav: np.ndarray,
    sr: int,
    thresh_db: float = EDGE_THRESH_DB,
    keep_s: float = EDGE_KEEP_S,
    fade_s: float = EDGE_FADE_S,
) -> np.ndarray:
    """Cắt im lặng model tự sinh ở hai đầu (giữ lại ``keep_s`` mỗi đầu) rồi fade
    cosine ``fade_s`` ở hai mép, để khe nối không click và khoảng nghỉ chỉ do
    ``join_audio_chunks`` quyết định. Trả về bản sao; wav rỗng trả nguyên."""
    if wav.size == 0:
        return wav
    lead, tail = edge_silence(wav, sr, thresh_db)
    keep = int(keep_s * sr)
    a = max(0, lead - keep)
    b = wav.size - max(0, tail - keep)
    out = np.array(wav[a:b], dtype=np.float32, copy=True)
    n = min(int(fade_s * sr), out.size // 2)
    if n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out


def pause_pad_samples(prev_wav: np.ndarray, next_wav: np.ndarray, sr: int, pause_s: float) -> int:
    """Số mẫu zeros cần chèn giữa ``prev_wav`` và ``next_wav`` để khoảng nghỉ THẬT
    (im lặng đuôi trước + zeros + im lặng đầu sau) đạt ``pause_s``; 0 nếu đã đủ.
    Dùng cho streaming, nơi chunk trước đã phát đi nên không trim được nữa (mẩu
    cuối toàn im lặng thì tính cả mẩu là đuôi)."""
    lead_prev, tail = edge_silence(prev_wav, sr)
    if lead_prev == prev_wav.size:          # prev toàn im lặng
        tail = prev_wav.size
    lead, _ = edge_silence(next_wav, sr)
    return max(0, int(pause_s * sr) - tail - lead)


def join_audio_chunks(
    chunks: List[np.ndarray],
    sr: int,
    silence_p: float = 0.0,
    crossfade_p: float = 0.0,
    silence_ps: Optional[List[float]] = None,
    *,
    explicit_pauses_ms: Optional[dict[int, int]] = None,
) -> np.ndarray:
    """Ghép các chunk audio.

    ``silence_ps`` (tuỳ chọn, đường v3): ``silence_ps[i]`` là khoảng nghỉ TỐI
    THIỂU (giây) giữa chunk ``i`` và ``i+1`` theo loại ranh giới. Audio từng chunk
    giữ NGUYÊN (không cắt mép, không fade); :func:`pause_pad_samples` đo im lặng
    đuôi chunk trước + đầu chunk sau và chỉ chèn zeros cho phần còn thiếu — giọng
    clone đuôi ngắn được bù cho đủ, preset đuôi dài giữ nhịp tự nhiên. Khi truyền
    ``silence_ps`` thì ``silence_p``/``crossfade_p`` bị bỏ qua; khe thiếu giá trị
    nghỉ 0 (nối thẳng).

    ``explicit_pauses_ms[i]`` chèn đúng số mẫu im lặng trước chunk ``i``, thay
    cho khoảng nghỉ tự động. Index ``len(chunks)`` là khoảng nghỉ cuối audio.

    Không có ``silence_ps`` (đường v1/v2): chèn ``silence_p`` giây zeros, hoặc
    crossfade ``crossfade_p`` giây, hoặc nối thẳng — giữ nguyên như cũ.
    """
    if silence_ps is not None or explicit_pauses_ms:
        pauses = explicit_pauses_ms or {}
        parts: List[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            pad = pauses.get(i, 0) * sr // 1000
            if i not in pauses and silence_ps is not None and 0 < i <= len(silence_ps):
                pad = pause_pad_samples(chunks[i - 1], chunk, sr, silence_ps[i - 1])
            if pad > 0:
                parts.append(np.zeros(pad, dtype=np.float32))
            parts.append(chunk)
        tail = pauses.get(len(chunks), 0) * sr // 1000
        if tail:
            parts.append(np.zeros(tail, dtype=np.float32))
        if parts:
            return np.concatenate(parts) if len(parts) > 1 else parts[0]
        return np.array([], dtype=np.float32)

    if not chunks:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return chunks[0]

    silence_samples   = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    final_wav = chunks[0]

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        if silence_samples > 0:
            silence   = np.zeros(silence_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif crossfade_samples > 0:
            overlap = min(len(final_wav), len(next_chunk), crossfade_samples)
            if overlap > 0:
                fade_out  = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in   = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                blended   = final_wav[-overlap:] * fade_out + next_chunk[:overlap] * fade_in
                final_wav = np.concatenate([final_wav[:-overlap], blended, next_chunk[overlap:]])
            else:
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            final_wav = np.concatenate([final_wav, next_chunk])

    return final_wav

# ─── v1: split raw text ──────────────────────────────────────────────────────

def _scan_sentences(text: str, quote_aware: bool = True) -> Tuple[List[str], bool]:
    """Quét ``text`` một lượt, cắt ở dấu ``.!?…`` KHÔNG nằm trong ngoặc/trích dẫn.

    Trả ``(sentences, balanced)``; ``balanced=False`` nghĩa là văn bản có ngoặc/
    nháy lệch (thiếu dấu đóng) — caller nên quét lại với ``quote_aware=False``.
    """
    sentences: List[str] = []
    n = len(text)
    start = i = 0
    depth = 0          # độ sâu ngoặc ( [ { “ « …
    in_quote = False   # đang trong "…" (nháy kép thẳng, đối xứng)

    while i < n:
        ch = text[i]
        if quote_aware and ch == _SYMMETRIC_QUOTE:
            in_quote = not in_quote
        elif quote_aware and ch in _OPENERS:
            depth += 1
        elif quote_aware and ch in _CLOSERS:
            if depth:
                depth -= 1
        elif ch in _SENT_END_CHARS and depth == 0 and not in_quote:
            j = i + 1
            while j < n and text[j] in _SENT_END_CHARS:   # nuốt "?!", "..."
                j += 1
            while j < n and text[j] in _TRAILING_CLOSE:   # nuốt dấu đóng bám sau
                j += 1
            # Chỉ là ranh giới câu khi theo sau là khoảng trắng hoặc hết văn bản;
            # nhờ vậy "3.5 triệu" / "8.30 sáng" không bị cắt.
            if j >= n or text[j].isspace():
                sentences.append(text[start:j])
                start = i = j
                continue
            i = j
            continue
        i += 1

    if start < n:
        sentences.append(text[start:])

    return [s.strip() for s in sentences if s.strip()], (depth == 0 and not in_quote)


def split_into_sentences(text: str) -> List[str]:
    """Tách ``text`` thành câu, KHÔNG cắt bên trong ngoặc/trích dẫn.

    Dùng trên text THÔ (trước normalize): normalizer của sea-g2p xoá sạch mọi dấu
    ngoặc (``"…"`` -> ``,``), nên sau normalize thì không còn cách nào phân biệt
    dấu ``?`` kết câu với dấu ``?`` trong câu trích dẫn.

    Nếu văn bản có ngoặc lệch (thiếu dấu đóng) thì quét lại bỏ qua ngoặc, để một
    dấu nháy lạc không nuốt toàn bộ phần còn lại thành một câu khổng lồ.
    """
    if not text:
        return []
    sentences, balanced = _scan_sentences(text, quote_aware=True)
    if not balanced:
        sentences, _ = _scan_sentences(text, quote_aware=False)
    return sentences


# ─── Cắt mảnh dài không còn dấu ngắt: ưu tiên từ nối ─────────────────────────
_CONN_WORDS = frozenset(
    "và nhưng hoặc song rồi nên vì nếu khi để do bởi".split()
)
# Cặp hai từ là MỘT từ nối: cắt trước cả cặp. Cặp cũng là luật CHẶN — không cắt
# lọt vào giữa cặp ("sau | khi") hay ngay sau từ đầu cặp ("cho | đến khi").
_CONN_PAIRS = frozenset([
    ("sau", "khi"), ("trước", "khi"), ("trong", "khi"), ("mỗi", "khi"),
    ("đến", "khi"), ("tới", "khi"), ("cho", "nên"), ("cho", "đến"),
    ("bởi", "vì"), ("nếu", "như"), ("tuy", "nhiên"), ("thế", "nhưng"),
    ("vì", "vậy"), ("vì", "thế"), ("do", "đó"), ("sau", "đó"),
])
_CONN_STRIP = "\"'“”‘’()[]«»…"


def _conn_key(token: str) -> str:
    return token.strip(_CONN_STRIP).lower()


# Trần ký tự của chunk là TƯƠNG ĐỐI, không cứng: phần dư sau điểm cắt mà quá
# ngắn (<= slack ký tự, tính cả dấu câu liền) thì gộp luôn vào chunk trước dù
# vượt trần. Ví dụ trần 128: "...tiêu diệt sinh lực đối" | "phương." -> mảnh
# "phương." 7 ký tự gộp về trước thành chunk 135 ký tự, thay vì đứng riêng rồi
# bị dán vào câu sau ("phương. Tuy nhiên, ..."). Slack 15 với trần thông thường
# (>= 120); trần bé thì thu theo tỉ lệ (max_chars // 8) để không nới quá tay.
CHUNK_TAIL_SLACK = 15


def _tail_slack(max_chars: int) -> int:
    return min(CHUNK_TAIL_SLACK, max_chars // 8)


def _fits(cur_len: int, add_len: int, max_chars: int) -> bool:
    """``add_len`` ký tự nối thêm (cách 1 dấu cách) vào chunk dài ``cur_len`` có
    vừa không: vừa trần, hoặc phần thêm đủ ngắn để hưởng slack."""
    total = cur_len + 1 + add_len if cur_len else add_len
    slack = _tail_slack(max_chars)
    return total <= max_chars or (add_len <= slack and total <= max_chars + slack)


# Từ mà normalizer dùng để đọc số. Cắt giữa hai từ này là cắt ngang một con số
# ("…đến hai nghìn | không trăm ba mươi mốt"). "năm"/"ba"/"tư" cũng là từ thường —
# chặn thừa một chút còn hơn xẻ đôi năm 2031.
_NUMBER_WORDS = frozenset(
    "không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi trăm nghìn ngàn "
    "triệu tỷ tỉ linh lẻ phẩy chấm".split()
)


def _is_number_word(token: str) -> bool:
    key = _conn_key(token)
    return key in _NUMBER_WORDS or key.isdigit()


def _span_len(words: List[str], start: int, end: int) -> int:
    """Độ dài ``" ".join(words[start:end])``."""
    if end <= start:
        return 0
    return sum(len(w) for w in words[start:end]) + (end - start - 1)


def _balanced_cut(words: List[str], start: int, target: float, max_chars: int, min_left: int) -> int:
    """Chọn ``end`` sao cho ``words[start:end]`` <= ``max_chars`` và dài GẦN
    ``target`` nhất. Ưu tiên cắt trước từ nối (``_CONN_WORDS``/``_CONN_PAIRS``,
    mảnh trái >= ``min_left``); không có thì ranh giới từ gần đích nhất, miễn
    không lọt giữa cặp từ nối và không xẻ đôi một con số. Không còn chỗ hợp lệ
    (một token khổng lồ, chuỗi cặp chồng lấn) thì cắt sát trần, lùi khỏi cặp."""
    best_nat = best_plain = None
    end_cap = start + 1
    for j in range(start + 1, len(words)):
        left = _span_len(words, start, j)
        if left > max_chars:
            break
        end_cap = j
        key, prev = _conn_key(words[j]), _conn_key(words[j - 1])
        nxt = _conn_key(words[j + 1]) if j + 1 < len(words) else ""
        d = abs(left - target)
        inside_pair = (prev, key) in _CONN_PAIRS
        natural = (key in _CONN_WORDS or (key, nxt) in _CONN_PAIRS) and prev not in _CONN_WORDS
        if natural and not inside_pair and left >= min_left and (best_nat is None or d < best_nat[0]):
            best_nat = (d, j)
        plain_ok = not inside_pair and not (_is_number_word(words[j - 1]) and _is_number_word(words[j]))
        if plain_ok and (best_plain is None or d < best_plain[0]):
            best_plain = (d, j)
    if best_nat is not None:
        return best_nat[1]
    if best_plain is not None:
        return best_plain[1]
    end = end_cap
    while end > start + 1 and (_conn_key(words[end - 1]), _conn_key(words[end])) in _CONN_PAIRS:
        end -= 1
    return end


def _split_long_part(part: str, max_chars: int) -> List[str]:
    """Cắt một mảnh dài quá ``max_chars`` (không còn dấu ngắt nào để bám) thành
    các mảnh <= ``max_chars`` theo TỪ, chia ĐỀU: biết trước cần
    ``k = ceil(len / max_chars)`` mảnh thì mỗi mảnh nhắm ``len / k`` ký tự và
    cắt ở điểm hợp lệ gần đích nhất (xem :func:`_balanced_cut`). Trước đây là
    greedy — đổ đầy sát trần rồi phần dư đứng riêng — nên 304 ký tự thành
    251 + 53, và điểm cắt "gần trần" thường trúng chỗ tệ: từ nối duy nhất bị
    từ chối vì mảnh trái chưa đủ nửa trần, rồi cắt cưỡng bức xẻ đôi
    "hai nghìn | không trăm ba mươi mốt".

    Từ nối chỉ được nhận khi mảnh trái >= ``max_chars // 3`` (từng là ``// 2``;
    một mệnh đề 85 ký tự vẫn là mệnh đề, còn con số bị cắt đôi thì không phải
    tiếng nói). Phần dư cuối ngắn hơn slack (``_tail_slack``) vẫn gộp vào mảnh
    trước dù vượt trần (trần tương đối). Token ``<en>...</en>`` luôn nguyên vẹn."""
    words = _tokenize_keep_en(part)
    min_left = max_chars // 3
    pieces: List[str] = []
    start = 0
    while start < len(words):
        rest = _span_len(words, start, len(words))
        if rest <= max_chars:
            pieces.append(" ".join(words[start:]))
            break
        # Trần tương đối: phần dư sau chỗ đầy trần mà chỉ là một mẩu ("phương.")
        # thì gộp luôn, không để mảnh vụn đứng riêng (xem ``_fits``).
        full = start + 1
        while full < len(words) and _span_len(words, start, full + 1) <= max_chars:
            full += 1
        if _fits(_span_len(words, start, full), _span_len(words, full, len(words)), max_chars):
            pieces.append(" ".join(words[start:]))
            break
        n_pieces = -(-rest // max_chars)          # ceil
        end = _balanced_cut(words, start, rest / n_pieces, max_chars, min_left)
        pieces.append(" ".join(words[start:end]))
        start = end
    return pieces


def pack_sentences_into_chunks(sentences: List[str], max_chars: int = 256) -> List[str]:
    """Đóng gói các CÂU đã cho thành chunk ~<= ``max_chars`` (greedy, giữ thứ tự).

    Câu dài hơn ``max_chars`` mới bị cắt phụ — trước theo dấu ngắt trong câu
    (``,;:``), sau cùng mới theo từ (ưu tiên cắt trước từ nối, xem
    :func:`_split_long_part`).

    Trần là TƯƠNG ĐỐI: câu/mảnh nối thêm ngắn hơn ``_tail_slack(max_chars)`` (15
    ký tự với trần thường) thì vẫn gộp vào chunk đang mở dù vượt trần bấy nhiêu —
    tránh mảnh vụn kiểu "phương." đứng riêng rồi bị dán sang câu sau.
    """
    final_chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if buffer:
                final_chunks.append(buffer)
                buffer = ""

            sub_parts = RE_MINOR_PUNCT.split(sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                if _fits(len(buffer), len(part), max_chars):
                    buffer = (buffer + ' ' + part) if buffer else part
                else:
                    if buffer:
                        final_chunks.append(buffer)
                    buffer = part
                    if len(buffer) > max_chars:
                        pieces = _split_long_part(buffer, max_chars)
                        final_chunks.extend(pieces[:-1])
                        buffer = pieces[-1] if pieces else ""
        else:
            if buffer and not _fits(len(buffer), len(sentence), max_chars):
                final_chunks.append(buffer)
                buffer = sentence
            else:
                buffer = (buffer + ' ' + sentence) if buffer else sentence

    if buffer:
        final_chunks.append(buffer)

    return [c.strip() for c in final_chunks if c.strip()]


def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """Split raw text (chưa phonemize) thành chunks <= max_chars."""
    if not text:
        return []

    final_chunks: List[str] = []
    for para in RE_NEWLINE.split(text.strip()):
        para = para.strip()
        if para:
            final_chunks.extend(
                pack_sentences_into_chunks(split_into_sentences(para), max_chars)
            )
    return final_chunks


def _classify_gap(chunk: str) -> str:
    """Phân loại ranh giới NGAY SAU ``chunk`` dựa trên dấu câu cuối: hết câu
    (``.!?``) -> ``"sentence"``; còn lại (``,;:`` hoặc cắt cưỡng bức giữa câu)
    -> ``"minor"``. Ranh giới ``"para"`` (ngắt đoạn) do caller gán riêng."""
    c = chunk.rstrip()
    return "sentence" if c and c[-1] in ".!?" else "minor"


def split_text_into_chunks_with_gaps(
    text: str, max_chars: int = 256
) -> Tuple[List[str], List[str]]:
    """Như :func:`split_text_into_chunks` nhưng trả kèm loại ranh giới GIỮA các
    chunk để ghép audio nghỉ dài/ngắn khác nhau.

    Trả về ``(chunks, gaps)`` với ``gaps[i] in {"para","sentence","minor"}`` là
    ranh giới giữa ``chunks[i]`` và ``chunks[i+1]`` (``len(gaps) == len(chunks)-1``):
      * ``"para"``     — hai chunk khác ĐOẠN (cách nhau bởi ``\\n``) -> nghỉ dài
      * ``"sentence"`` — hết câu (chunk trái tận cùng ``.!?``)       -> nghỉ vừa
      * ``"minor"``    — ngắt trong câu (``,;:`` / cắt cưỡng bức)     -> gần như liền
    """
    if not text:
        return [], []

    paragraphs = RE_NEWLINE.split(text.strip())
    chunks: List[str] = []
    gaps: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_chunks = split_text_into_chunks(para, max_chars=max_chars)
        if not para_chunks:
            continue
        if chunks:                       # ranh giới với đoạn TRƯỚC đó là ngắt đoạn
            gaps.append("para")
        for j, ch in enumerate(para_chunks):
            if j > 0:                    # ranh giới trong CÙNG đoạn: theo dấu câu
                gaps.append(_classify_gap(para_chunks[j - 1]))
            chunks.append(ch)

    return chunks, gaps

# ─── v2 helpers ──────────────────────────────────────────────────────────────

def _pick_strongest(m: re.Match) -> str:
    s = m.group(0)
    return '!' if '!' in s else '?' if '?' in s else '.'


def _clean_phoneme_noise(text: str) -> str:
    for pattern, repl in _NOISE_RULES:
        text = pattern.sub(repl, text)
    return _MULTI_PUNCT.sub(_pick_strongest, text).strip()


def _find_best_split(text: str, max_size: int) -> Tuple[int, bool]:
    mid = max_size // 2
    best_comma_pos, best_comma_dist = -1, max_size
    best_space_pos, best_space_dist = -1, max_size

    for i in range(min(max_size, len(text))):
        ch = text[i]
        if ch == ',':
            d = abs(i - mid)
            if d < best_comma_dist:
                best_comma_dist, best_comma_pos = d, i
        elif ch == ' ':
            d = abs(i - mid)
            if d < best_space_dist:
                best_space_dist, best_space_pos = d, i

    if best_comma_pos != -1:
        return best_comma_pos, True
    if best_space_pos != -1:
        return best_space_pos, False
    return -1, False


def _smart_split_body(text: str, max_chunk_size: int) -> List[str]:
    result: List[str] = []
    stack = [text.strip()]

    while stack:
        seg = stack.pop()
        if not seg:
            continue
        if len(seg) <= max_chunk_size:
            result.append(seg)
            continue

        pos, _ = _find_best_split(seg, max_chunk_size)
        if pos != -1:
            left  = seg[:pos].rstrip()
            right = seg[pos + 1:].lstrip()
        else:
            cut = max_chunk_size
            while cut > 0 and seg[cut - 1] != ' ':
                cut -= 1
            if cut == 0:
                cut = max_chunk_size
            left  = seg[:cut].rstrip()
            right = seg[cut:].lstrip()

        if right:
            stack.append(right)
        if left:
            stack.append(left)

    return result


def _split_sentence(sent: str, max_chunk_size: int) -> List[PhoneChunk]:
    sent = sent.strip()
    if not sent:
        return []

    if sent[-1] in '.!?':
        body, punct = sent[:-1].rstrip(), sent[-1]
    else:
        body, punct = sent, '.'

    if not body:
        return []

    if len(sent) <= max_chunk_size:
        return [PhoneChunk(text=body + punct, is_sentence_end=True)]

    sub_chunks = _smart_split_body(body, max_chunk_size)
    if not sub_chunks:
        return [PhoneChunk(text=punct, is_sentence_end=True)]

    last_idx = len(sub_chunks) - 1
    return [
        PhoneChunk(
            text=chunk + (punct if i == last_idx else '.'),
            is_sentence_end=(i == last_idx),
        )
        for i, chunk in enumerate(sub_chunks)
        if chunk
    ]

# ─── v2: split phoneme string ────────────────────────────────────────────────

def split_into_chunks_v2(
    full_phones: str,
    max_chunk_size: int = 256,
    min_chunk_size: int = 10,
) -> List[PhoneChunk]:
    """
    Phân đoạn chuỗi phoneme thành các PhoneChunk.
      is_sentence_end=True  → kết thúc câu thật → cần silence
      is_sentence_end=False → cắt nhân tạo → không cần silence
    """
    if not full_phones:
        return []

    full_phones = _clean_phoneme_noise(full_phones)

    raw_parts: List[PhoneChunk] = []
    for para in RE_NEWLINE.split(full_phones):
        para = para.strip()
        if not para:
            continue
        for sent in RE_SENTENCE_FINDALL.findall(para):
            sent = sent.strip()
            if sent:
                raw_parts.extend(_split_sentence(sent, max_chunk_size))

    if not raw_parts:
        return []

    merged: List[PhoneChunk] = []
    i, n = 0, len(raw_parts)
    while i < n:
        cur = raw_parts[i]
        while len(cur.text) < min_chunk_size and i + 1 < n:
            nxt       = raw_parts[i + 1]
            candidate = cur.text.rstrip('.!?').rstrip() + ' ' + nxt.text
            if len(candidate) <= max_chunk_size:
                cur = PhoneChunk(text=candidate, is_sentence_end=nxt.is_sentence_end)
                i += 1
            else:
                break
        merged.append(cur)
        i += 1

    if len(merged) >= 2 and len(merged[-1].text) < min_chunk_size:
        last      = merged.pop()
        candidate = merged[-1].text.rstrip('.!?').rstrip() + ' ' + last.text
        if len(candidate) <= max_chunk_size:
            merged[-1] = PhoneChunk(text=candidate, is_sentence_end=last.is_sentence_end)
        else:
            merged.append(last)

    return merged


def get_silence_duration_v2(chunk: PhoneChunk) -> float:
    """
    Silence sau chunk (giây).
      is_sentence_end=False → 0.0s
      kết thúc '!'/'?' → 0.4s
      kết thúc '.' → 0.3s
    """
    if not chunk.is_sentence_end:
        return 0.0
    return 0.4 if chunk.text.strip()[-1] in '!?' else 0.3

# ─── Misc ────────────────────────────────────────────────────────────────────

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def trust_remote_code_enabled() -> bool:
    """``trust_remote_code`` for user-supplied model repos (``VIENEU_TRUST_REMOTE_CODE=1``).

    Off by default: the v1/v2 checkpoints are stock Qwen2/Qwen3 models that need no
    custom code, while a repo id typed into the Web UI can point anywhere — with
    remote code on, loading it runs that repo's Python on this machine.
    """
    return env_bool("VIENEU_TRUST_REMOTE_CODE", default=False)


def check_sampling(temperature, top_k, top_p, repetition_penalty) -> Tuple[float, int, float, float]:
    """Reject sampling settings that would break a shared GPU graph.

    A NaN temperature or a zero penalty turns the logits into NaN/inf, and
    ``torch.multinomial`` on CUDA then fails with a device-side assert that
    poisons the CUDA context for every request in the process. Returns the
    values as ``(float, int, float, float)``; raises ``ValueError`` otherwise.
    ``temperature`` 0 is greedy and ``top_k`` <= 0 disables top-k.
    """
    try:
        t, k, p, rp = float(temperature), int(top_k), float(top_p), float(repetition_penalty)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"invalid sampling settings: {e}") from None
    if not (math.isfinite(t) and t >= 0):
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
    if not math.isfinite(p):
        raise ValueError(f"top_p must be a finite number, got {top_p!r}")
    if not (math.isfinite(rp) and 1e-3 <= rp <= 1e3):
        raise ValueError(f"repetition_penalty must be between 0.001 and 1000, got {repetition_penalty!r}")
    return t, k, p, rp

# ── Babble guard (chunk rất ngắn "nói thêm") ─────────────────────────────────
# Chunk 1-3 tiếng thỉnh thoảng bắn trượt stop token rồi bịa thêm một từ cho
# "trọn câu" ("Được." -> "Được không?", "Vâng." -> "Vâng khi tại."). Đo 160 chunk
# ngắn trên GPU (2026-09): ~3-6% chunk bị, gần như chỉ ở chunk <= 2 tiếng. Xác suất
# EOS không báo trước và mã codec "im lặng" cũng xuất hiện ở khoảng nghỉ giữa tiếng,
# nên không chặn được TRONG vòng lặp sinh; thay vào đó sinh xong thì đếm số cụm
# năng lượng: một tiếng = một cụm, nhiều cụm hơn số tiếng => bịa => sinh lại.
BABBLE_MAX_SYLLABLES = 3       # chỉ kiểm chunk có <= N tiếng (dài hơn thì cụm dính nhau, không đếm được)
BABBLE_MAX_RETRIES = 2
_BURST_THRESH_DB = -18.0       # cụm = RMS trên ngưỡng này so với RMS đỉnh (hơi thở ~ -25 dB không tính)
_BURST_MIN_GAP_MS = 60         # hai cụm cách nhau dưới mức này là một cụm (phụ âm đầu bật hơi)
_BURST_MIN_MS = 30
# ɚ ɝ  nguyên âm r-colored (computer, fire, hour, bird — giọng Mỹ)
# ᵻ ᵿ  nguyên âm giảm eSpeak dùng cho âm tiết không nhấn (director -> dᵻɹˈɛktɚ)
# Thiếu bốn ký hiệu này thì cả một âm tiết biến mất khỏi phép đếm. Tiếng Việt
# không dùng ký hiệu nào trong số đó nên bổ sung không đụng tới tiếng Việt.
_IPA_VOWELS = set("aeiouyæɐɑɒɔəɘɛɜɤɯɵøœʉʊʌɪɨɚɝᵻᵿ")


def syllable_count(phonemes: str) -> int:
    """Số tiếng (âm tiết) ước lượng từ chuỗi phoneme SEA-G2P.

    Tiếng Việt: mỗi từ là một tiếng. Từ tiếng Anh (<en>) có thể nhiều âm tiết
    nên đếm theo số cụm nguyên âm ('sˈækaɪ' -> 2). Markup và dấu câu bị loại.
    """
    stripped = _FRAME_MARKUP_RE.sub("", phonemes or "")
    total = 0
    for tok in stripped.split():
        # Nhóm nguyên âm mới chỉ bắt đầu sau một PHỤ ÂM thật; dấu dài (ː), dấu nhấn
        # (ˈ ˌ) và số thanh điệu không tách nhóm — 'kwˈaːɜ' (quá) là MỘT tiếng dù
        # 'ɜ' ở đây là ký hiệu thanh sắc chứ không phải nguyên âm.
        groups, in_v, consonant_seen = 0, False, True
        for ch in tok:
            if ch in _IPA_VOWELS:
                if not in_v and consonant_seen:
                    groups += 1
                in_v, consonant_seen = True, False
            elif ch in "ːˈˌ" or ch.isdigit():
                # Dấu nhấn đứng TRƯỚC âm tiết mà nó nhấn, nên một dấu nhấn xuất
                # hiện SAU khi token đã có cụm nguyên âm là ranh giới âm tiết —
                # kể cả khi không có phụ âm nào chen giữa:
                #   kɹiːˈeɪt (create)   iː | ˈeɪ        -> 2, trước đây đếm 1
                #   kəmpjˈuːɾɚ          ə | ˈuː | ɾɚ    -> 3, trước đây đếm 2
                #   ɹˈeɪdɪˌoʊ (radio)   ˈeɪ | dɪ | ˌoʊ  -> 3, trước đây đếm 2
                # Điều kiện groups > 0 giữ tiếng Việt nguyên vẹn: sea-g2p đặt
                # đúng MỘT dấu nhấn cho mỗi tiếng và luôn TRƯỚC cụm nguyên âm
                # đầu tiên (bˈaːɜw, kwˈaːɜ), lúc đó groups vẫn bằng 0.
                if ch in "ˈˌ" and groups > 0:
                    in_v, consonant_seen = False, True
                else:
                    in_v = False
            else:
                in_v, consonant_seen = False, True
        if any(ch.isalpha() for ch in tok):
            total += max(1, groups)
    return total


def count_speech_bursts(wav: np.ndarray, sr: int) -> int:
    """Số cụm năng lượng (xấp xỉ số tiếng) trong một waveform mono."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    hop = max(1, int(sr * 0.010))
    n = len(wav) // hop
    if n == 0:
        return 0
    env = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    peak = float(env.max())
    if peak <= 1e-6:
        return 0
    on = env > peak * (10 ** (_BURST_THRESH_DB / 20))
    min_gap = max(1, _BURST_MIN_GAP_MS // 10)
    min_len = max(1, _BURST_MIN_MS // 10)
    bursts, start, last_on = [], None, None
    for i, o in enumerate(on):
        if o:
            if start is None:
                start = i
            elif i - last_on > min_gap:          # gap dài hơn ngưỡng -> cụm mới
                bursts.append((start, last_on)); start = i
            last_on = i
    if start is not None:
        bursts.append((start, last_on))
    return sum(1 for a, b in bursts if (b - a + 1) >= min_len)


def babble_suspect(wav: np.ndarray, sr: int, phonemes: str, cap_frames: int,
                   n_frames: Optional[int] = None, frames_per_sec: float = 12.5):
    """-> (suspect, syllables, bursts, n_frames) cho MỘT chunk vừa sinh.

    Chỉ xét chunk <= BABBLE_MAX_SYLLABLES tiếng, không có emotion cue. Nghi "nói
    thêm" khi: số cụm âm > số tiếng, HOẶC chunk <= 2 tiếng chạy tới sát trần frame
    (đo A/B 720 chunk: mọi ca bịa thêm từ đều là chunk 1 tiếng ở 12-13/13 frame,
    chunk 1 tiếng bình thường EOS ở 6-9 frame).
    """
    if n_frames is None:
        n_frames = int(round(len(wav) / (sr / frames_per_sec)))
    if is_cue_only(phonemes):
        # Không đếm cụm được (một tràng cười là nhiều cụm) — chỉ dùng luật chạm trần.
        return n_frames >= cap_frames - 1, 0, 0, n_frames
    syl = syllable_count(phonemes)
    if syl == 0 or syl > BABBLE_MAX_SYLLABLES or "<|emotion_" in (phonemes or ""):
        return False, syl, 0, 0
    bursts = count_speech_bursts(wav, sr)
    hit_cap = syl <= 2 and n_frames >= cap_frames - 1
    return (bursts > syl) or hit_cap, syl, bursts, n_frames


def babble_prefer(new, old) -> bool:
    """Bản sinh lại ``new`` có đáng thay ``old`` không (tuple từ babble_suspect)."""
    (n_bad, _, n_b, n_len), (o_bad, _, o_b, o_len) = new, old
    if n_bad != o_bad:
        return not n_bad
    return n_b < o_b or (n_b == o_b and n_len < o_len)


def babble_log_line(best, tries: int, cap: int) -> str:
    what = f"chunk {best[1]} tiếng: {best[2]} cụm âm" if best[1] else "cue đứng một mình"
    return (f"babble guard: {what}, {best[3]}/{cap} frame sau {tries} lần sinh lại"
            + (" — vẫn nghi ngờ" if best[0] else ""))
