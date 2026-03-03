import os
import json
import platform
import glob
import re
import logging
import functools
from phonemizer import phonemize
from phonemizer.backend.espeak.espeak import EspeakWrapper
from vieneu_utils.normalize_text import VietnameseTTSNormalizer

# Configuration
DICT_DIR = os.getenv(
    'PHONEME_DICT_DIR',
    os.path.join(os.path.dirname(__file__), "phone_dict")
)

MERGED_DICT_PATH = os.path.join(DICT_DIR, "phone_dict_merged.json")
COMMON_DICT_PATH = os.path.join(DICT_DIR, "phone_dict_common.json")

# Configure logging
logger = logging.getLogger("Vieneu.Phonemizer")

def load_json(path: str) -> dict:
    """Load JSON file safely."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Warning: Dictionary not found at {path}")
        return {}

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
phone_dict_merged = load_json(MERGED_DICT_PATH)
phone_dict_common = load_json(COMMON_DICT_PATH)
normalizer = VietnameseTTSNormalizer()

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

def propagate_language(tokens):
    """
    Propagate language labels for 'common' words based on the closest anchor.
    Sentence boundaries (strong punctuation) block propagation.
    """
    STOP_PUNCT = {'.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}'}
    
    # 1. Identify islands of common words
    islands = []
    current_island = []
    for i, token in enumerate(tokens):
        if token['lang'] == 'common':
            current_island.append(i)
        else:
            # Any non-common token (vi, en, or punct) breaks the island
            if current_island:
                islands.append(current_island)
                current_island = []
    if current_island:
        islands.append(current_island)

    # 2. For each island, find the closest valid anchor
    for island in islands:
        left_anchor, left_dist = None, 999
        right_anchor, right_dist = None, 999
        
        # Search left from the start of island
        for l in range(island[0] - 1, -1, -1):
            if tokens[l]['content'] in STOP_PUNCT: break
            if tokens[l]['lang'] in ('vi', 'en'):
                left_anchor = tokens[l]['lang']
                left_dist = island[0] - l
                break
        
        # Search right from the end of island
        for r in range(island[-1] + 1, len(tokens)):
            if tokens[r]['content'] in STOP_PUNCT: break
            if tokens[r]['lang'] in ('vi', 'en'):
                right_anchor = tokens[r]['lang']
                right_dist = r - island[-1]
                break
        
        # Decision logic: closest wins, tie-break to RIGHT for better switching
        final_lang = 'vi' # Default
        if left_anchor and right_anchor:
            # If distance is equal, we often prefer the language that follows (the target phrase)
            final_lang = right_anchor if right_dist <= left_dist else left_anchor
        elif left_anchor:
            final_lang = left_anchor
        elif right_anchor:
            final_lang = right_anchor
            
        for idx in island:
            tokens[idx]['lang'] = final_lang

@functools.lru_cache(maxsize=1024)
def _phonemize_with_dict_cached(text: str, skip_normalize: bool = False) -> str:
    return phonemize_batch([text], skip_normalize=skip_normalize, phoneme_dict=None)[0]

def phonemize_batch(texts: list[str], skip_normalize: bool = False, phoneme_dict: dict = None, **kwargs) -> list[str]:
    """Phonemize multiple texts with bilingual support and batch deduplication."""
    if not texts: return []
    if not skip_normalize: texts = [normalizer.normalize(t) for t in texts]

    use_system = (phoneme_dict is None)
    custom = phoneme_dict if phoneme_dict else {}

    batch_token_lists = []
    global_unknown = set()

    for text in texts:
        matches = re.finditer(r'(<en>.*?</en>)|(\w+)|([^\w\s])', text, re.I | re.U)
        sent_tokens = []
        for m in matches:
            en_tag, word, punct = m.groups()
            if en_tag:
                content = re.sub(r'</?en>', '', en_tag, flags=re.I).strip()
                for st in re.finditer(r'(\w+)|([^\w\s])', content, re.U):
                    sw, sp = st.groups()
                    if sp: sent_tokens.append({'lang': 'punct', 'content': sp, 'phone': sp})
                    else:
                        lw = sw.lower()
                        if lw in custom: sent_tokens.append({'lang': 'en', 'content': sw, 'phone': custom[lw]})
                        elif use_system and lw in phone_dict_common: sent_tokens.append({'lang': 'en', 'content': sw, 'phone': phone_dict_common[lw]})
                        elif use_system and lw in phone_dict_merged and phone_dict_merged[lw].startswith('<en>'):
                            sent_tokens.append({'lang': 'en', 'content': sw, 'phone': phone_dict_merged[lw]})
                        else:
                            sent_tokens.append({'lang': 'en', 'content': sw, 'phone': None})
                            global_unknown.add(sw)
            elif punct: sent_tokens.append({'lang': 'punct', 'content': punct, 'phone': punct})
            elif word:
                lw = word.lower()
                if lw in custom: sent_tokens.append({'lang': 'en', 'content': word, 'phone': custom[lw]})
                elif use_system and lw in phone_dict_merged:
                    val = phone_dict_merged[lw]
                    sent_tokens.append({'lang': 'en' if val.startswith('<en>') else 'vi', 'content': word, 'phone': val})
                elif use_system and lw in phone_dict_common:
                    sent_tokens.append({'lang': 'common', 'content': word, 'phone': phone_dict_common[lw]})
                else:
                    sent_tokens.append({'lang': 'en', 'content': word, 'phone': None})
                    global_unknown.add(word)
        batch_token_lists.append(sent_tokens)

    if global_unknown:
        u_words = sorted(list(global_unknown))
        res_phones = espeak_fallback_batch(u_words, 'en-us')
        lut = {w: f"<en>{p}" for w, p in zip(u_words, res_phones)}
        for sent in batch_token_lists:
            for t in sent:
                if t['phone'] is None and t['content'] in lut:
                    t['phone'] = lut[t['content']]

    results = []
    for sent in batch_token_lists:
        propagate_language(sent)
        sent_phones = []
        for t in sent:
            if t['lang'] == 'punct':
                sent_phones.append(t['phone'])
            else:
                p = t['phone']
                if isinstance(p, dict):
                    p = p['en'] if t['lang'] == 'en' else p['vi']
                if p is None: p = t['content']
                p = p.replace('<en>', '')
                # North VI 'r' adjustment removed as requested
                # if t['lang'] == 'vi' and t['content'].lower().startswith('r') and not p.startswith('ɹ'):
                #     p = 'ɹ' + p[1:]
                sent_phones.append(p)
        txt = " ".join(sent_phones)
        txt = re.sub(r'\s+([.,!?;:])', r'\1', txt)
        results.append(txt.strip())
    return results

def phonemize_text(text: str) -> str:
    return phonemize_batch([text])[0]

def phonemize_with_dict(text: str, phoneme_dict: dict = None, skip_normalize: bool = False) -> str:
    if phoneme_dict is not None:
        return phonemize_batch([text], skip_normalize=skip_normalize, phoneme_dict=phoneme_dict)[0]
    return _phonemize_with_dict_cached(text, skip_normalize=skip_normalize)

if __name__ == "__main__":
    import sys
    test_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "tôi muốn to long go to the market"
    print(f"Output: {phonemize_text(test_text)}")