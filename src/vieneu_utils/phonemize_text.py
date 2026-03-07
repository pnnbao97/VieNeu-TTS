import functools
import glob
import json
import logging
import os
import platform
import re
import sqlite3
import threading

from phonemizer import phonemize
from phonemizer.backend.espeak.espeak import EspeakWrapper

from vieneu_utils.normalize_text import VietnameseTTSNormalizer

# Configuration
DICT_DIR = os.getenv(
    'PHONEME_DICT_DIR',
    os.path.join(os.path.dirname(__file__), "phone_dict")
)

DB_PATH = os.path.join(DICT_DIR, "phone_dict.db")

# Configure logging
logger = logging.getLogger("Vieneu.Phonemizer")

class PhonemeDB:
    """SQLite-based dictionary for fast lookup and low memory usage."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()

    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn

    def lookup_batch(self, words: list[str]) -> tuple[dict, dict]:
        """Fetch multiple words from DB in two logical groups: merged and common."""
        if not words: return {}, {}
        conn = self._get_conn()
        cursor = conn.cursor()

        merged_map = {}
        common_map = {}

        # SQLite has a limit on the number of host parameters (typically 999)
        chunk_size = 950
        for i in range(0, len(words), chunk_size):
            chunk = words[i : i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))

            # Query merged table
            cursor.execute(f"SELECT word, phone FROM merged WHERE word IN ({placeholders})", chunk)
            merged_map.update(dict(cursor.fetchall()))

            # Query common table
            cursor.execute(f"SELECT word, vi_phone, en_phone FROM common WHERE word IN ({placeholders})", chunk)
            for row in cursor.fetchall():
                common_map[row[0]] = {"vi": row[1], "en": row[2]}

        return merged_map, common_map

def setup_espeak_library() -> None:
    """Configure eSpeak library path based on operating system."""
    system = platform.system()

    if system == "Windows":
        _setup_windows_espeak()
    elif system == "Linux":
        _setup_linux_espeak()
    elif system == "Darwin":
        _setup_macos_espeak()
    else:
        logger.warning(f"Warning: Unsupported OS: {system}")
        return

def _setup_windows_espeak() -> None:
    """Setup eSpeak for Windows."""
    default_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    if os.path.exists(default_path):
        EspeakWrapper.set_library(default_path)
    else:
        logger.warning("\033[91;1m⚠️ eSpeak-NG is not installed. The system will use the built-in dictionary, but it is recommended to install eSpeak-NG for maximum performance and accuracy.\033[0m")

def _setup_linux_espeak() -> None:
    """Setup eSpeak for Linux."""
    search_patterns = [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so*",
        "/usr/lib/x86_64-linux-gnu/libespeak.so*",
        "/usr/lib/libespeak-ng.so*",
        "/usr/lib64/libespeak-ng.so*",
        "/usr/local/lib/libespeak-ng.so*",
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            EspeakWrapper.set_library(sorted(matches, key=len)[0])
            return

    logger.warning("\033[91;1m⚠️ eSpeak-NG is not installed on Linux. The system will use the built-in dictionary, but it is recommended to install eSpeak-NG (sudo apt install espeak-ng) for maximum performance.\033[0m")

def _setup_macos_espeak() -> None:
    """Setup eSpeak for macOS."""
    espeak_lib = os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')

    paths_to_check = [
        espeak_lib,
        "/opt/homebrew/lib/libespeak-ng.dylib",  # Apple Silicon
        "/usr/local/lib/libespeak-ng.dylib",     # Intel
        "/opt/local/lib/libespeak-ng.dylib",     # MacPorts
    ]

    for path in paths_to_check:
        if path and os.path.exists(path):
            EspeakWrapper.set_library(path)
            return

    logger.warning("\033[91;1m⚠️ eSpeak-NG is not installed on macOS. The system will use the built-in dictionary, but it is recommended to install eSpeak-NG (brew install espeak-ng) for maximum performance.\033[0m")

# Initialize
setup_espeak_library()
phone_db = PhonemeDB(DB_PATH)
normalizer = VietnameseTTSNormalizer()

# Compiled Regular Expressions for tokenization
RE_PHONEMIZE_MATCH = re.compile(r'(<en>.*?</en>)|(\w+)|([^\w\s])', re.I | re.U)
RE_PHONEMIZE_TAG_CONTENT = re.compile(r'(\w+)|([^\w\s])', re.U)
RE_PHONEMIZE_TAG_STRIP = re.compile(r'</?en>', flags=re.I)
RE_PHONEMIZE_PUNCT_CLEANUP = re.compile(r'\s+([.,!?;:])')

_VI_ACCENTS = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"

def espeak_fallback_batch(texts: list[str], language: str = 'en-us') -> list[str]:
    """Batch fallback to espeak-ng for unknown segments."""
    if not texts: return []
    try:
        ph = phonemize(
            texts,
            language=language,
            backend='espeak',
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags"
        )
        if isinstance(ph, str): ph = [ph]
        return [p.strip() for p in ph]
    except Exception as e:
        logger.warning(f"eSpeak fallback ({language}) failed: {e}")
        return texts

_STOP_PUNCT = {'.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}'}

def propagate_language(tokens):
    """
    Propagate language labels for 'common' words based on the closest anchor.
    Sentence boundaries (strong punctuation) block propagation.
    Optimized to find anchors and update islands in a more streamlined way.
    """
    if not tokens:
        return

    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i]['lang'] == 'common':
            # Found start of a 'common' island
            start = i
            while i < n and tokens[i]['lang'] == 'common':
                i += 1
            end = i - 1

            # Now find anchors for this island [start, end]
            left_anchor, left_dist = None, 999
            right_anchor, right_dist = None, 999

            # Search left
            for l in range(start - 1, -1, -1):
                if tokens[l]['content'] in _STOP_PUNCT: break
                if tokens[l]['lang'] in ('vi', 'en'):
                    left_anchor = tokens[l]['lang']
                    left_dist = start - l
                    break

            # Search right
            for r in range(end + 1, n):
                if tokens[r]['content'] in _STOP_PUNCT: break
                if tokens[r]['lang'] in ('vi', 'en'):
                    right_anchor = tokens[r]['lang']
                    right_dist = r - end
                    break

            # Decision logic
            final_lang = 'vi' # Default fallback
            if left_anchor and right_anchor:
                final_lang = right_anchor if right_dist <= left_dist else left_anchor
            elif left_anchor:
                final_lang = left_anchor
            elif right_anchor:
                final_lang = right_anchor

            for idx in range(start, end + 1):
                tokens[idx]['lang'] = final_lang
        else:
            i += 1

@functools.lru_cache(maxsize=1024)
def _phonemize_with_dict_cached(text: str, skip_normalize: bool = False) -> str:
    return phonemize_batch([text], skip_normalize=skip_normalize, phoneme_dict=None)[0]

def phonemize_batch(texts: list[str], skip_normalize: bool = False, phoneme_dict: dict = None, **kwargs) -> list[str]:
    """Phonemize multiple texts with bilingual support and batch deduplication."""
    if not texts: return []
    if not skip_normalize:
        texts = [normalizer.normalize(t) for t in texts]

    use_system = (phoneme_dict is None)
    custom = phoneme_dict or {}

    batch_token_lists = []
    all_words = set()
    global_unknown = set()
    force_espeak_words = set()

    # 1. Tokenize and identify words
    for text in texts:
        sent_tokens = []
        for m in RE_PHONEMIZE_MATCH.finditer(text):
            en_tag, word, punct = m.groups()
            if en_tag:
                content = RE_PHONEMIZE_TAG_STRIP.sub('', en_tag).strip()
                for st in RE_PHONEMIZE_TAG_CONTENT.finditer(content):
                    sw, sp = st.groups()
                    if sp:
                        sent_tokens.append({'lang': 'punct', 'content': sp, 'phone': sp})
                    else:
                        sent_tokens.append({'lang': 'en', 'content': sw, 'phone': None, 'force_espeak': True})
                        force_espeak_words.add(sw)
            elif punct:
                sent_tokens.append({'lang': 'punct', 'content': punct, 'phone': punct})
            elif word:
                sent_tokens.append({'lang': 'unknown', 'content': word, 'phone': None})
                all_words.add(word.lower())
        batch_token_lists.append(sent_tokens)

    # 2. Bulk resolve from DB
    db_merged, db_common = phone_db.lookup_batch(list(all_words)) if use_system else ({}, {})

    # 3. Primary lookup (Custom dict -> Merged DB -> Common DB)
    for sent in batch_token_lists:
        for t in sent:
            if t['lang'] == 'punct' or t.get('force_espeak'): continue
            lw = t['content'].lower()

            if lw in custom:
                t['phone'], t['lang'] = custom[lw], 'en'
            elif lw in db_merged:
                val = db_merged[lw]
                t['phone'] = val
                t['lang'] = 'en' if val.startswith('<en>') else 'vi'
            elif lw in db_common:
                t['phone'], t['lang'] = db_common[lw], 'common'
            else:
                global_unknown.add(t['content'])
                t['lang'] = 'en' # Placeholder for espeak

    # 4. Batch espeak for forced and unknown words
    lut = {}
    if force_espeak_words:
        fe_words = sorted(list(force_espeak_words))
        fe_phones = espeak_fallback_batch(fe_words, 'en-us')
        # Check if espeak is actually working by seeing if phonemes differ from original words
        if fe_phones and fe_phones != fe_words:
             lut.update({w: f"<en>{p}" for w, p in zip(fe_words, fe_phones)})
        else:
             # Fallback: if espeak is not working, don't wrap in <en> to avoid mismatching with tagged tests
             lut.update({w: p for w, p in zip(fe_words, fe_phones)})

    if global_unknown:
        u_words = sorted(list(global_unknown))
        def has_accent(w): return any(c in _VI_ACCENTS for c in w.lower())
        vi_words = [w for w in u_words if has_accent(w)]
        en_words = [w for w in u_words if not has_accent(w)]

        if vi_words:
            res_vi = espeak_fallback_batch(vi_words, 'vi')
            lut.update(dict(zip(vi_words, res_vi)))
        if en_words:
            res_en = espeak_fallback_batch(en_words, 'en-us')
            lut.update({w: f"<en>{p}" for w, p in zip(en_words, res_en)})

    # 5. Apply espeak results and finalize
    results = []
    for sent in batch_token_lists:
        for t in sent:
            if t['phone'] is None and t['content'] in lut:
                t['phone'] = lut[t['content']]
                if not t.get('force_espeak') and any(c in _VI_ACCENTS for c in t['content'].lower()):
                    t['lang'] = 'vi'

        propagate_language(sent)

        sent_phones = []
        for t in sent:
            if t['lang'] == 'punct':
                sent_phones.append(t['phone'])
            else:
                p = t['phone']
                if isinstance(p, dict):
                    p = p['en'] if t['lang'] == 'en' else p['vi']

                if p is None:
                    p = t['content']

                sent_phones.append(p.replace('<en>', ''))

        txt = RE_PHONEMIZE_PUNCT_CLEANUP.sub(r'\1', " ".join(sent_phones)).strip()
        results.append(txt)
    return results

def phonemize_text(text: str) -> str:
    return phonemize_batch([text])[0]

def phonemize_with_dict(text: str, phoneme_dict: dict = None, skip_normalize: bool = False) -> str:
    if phoneme_dict is not None:
        return phonemize_batch([text], skip_normalize=skip_normalize, phoneme_dict=phoneme_dict)[0]
    return _phonemize_with_dict_cached(text, skip_normalize=skip_normalize)

if __name__ == "__main__":
    import sys
    test_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Đọc kết quả sẽ giúp hiểu rõ hơn sở thích và thói quen của thế hệ Z."
    print(f"Output: {phonemize_text(test_text)}")
