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
PHONEME_DICT_PATH = os.getenv(
    'PHONEME_DICT_PATH',
    os.path.join(os.path.dirname(__file__), "phoneme_dict.json")
)

def load_phoneme_dict(path: str = PHONEME_DICT_PATH) -> dict:
    """Load phoneme dictionary from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phoneme dictionary not found at {path}. "
            "Please create it or set PHONEME_DICT_PATH environment variable."
        )

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

# Configure logging
logger = logging.getLogger("Vieneu.Phonemizer")

# Initialize
setup_espeak_library()

try:
    phoneme_dict = load_phoneme_dict()
    normalizer = VietnameseTTSNormalizer()
except Exception as e:
    logger.error(f"Initialization error: {e}")
    # We still need normalizer to function
    normalizer = VietnameseTTSNormalizer()
    phoneme_dict = {}

def phonemize_text(text: str) -> str:
    """
    Convert text to phonemes (simple version without dict, without EN tag).
    Kept for backward compatibility.
    """
    text = normalizer.normalize(text)
    return phonemize(
        text,
        language="vi",
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags"
    )


def phonemize_with_dict(text: str, phoneme_dict: dict = None, skip_normalize: bool = False) -> str:
    """
    Phonemize single text with dictionary lookup and EN tag support.
    Uses LRU cache when the default dictionary is used.
    """
    if phoneme_dict is None or phoneme_dict is globals().get('phoneme_dict'):
        return _phonemize_with_dict_cached(text, skip_normalize)

    return phonemize_batch([text], phoneme_dict=phoneme_dict, skip_normalize=skip_normalize)[0]


@functools.lru_cache(maxsize=1024)
def _phonemize_with_dict_cached(text: str, skip_normalize: bool = False) -> str:
    """Internal cached version of phonemization using the global dictionary."""
    return phonemize_batch([text], skip_normalize=skip_normalize)[0]


def phonemize_batch(texts: list[str], phoneme_dict: dict = phoneme_dict, skip_normalize: bool = False) -> list[str]:
    """
    Phonemize multiple texts with optimal batching and deduplication.
    
    Args:
        texts: List of text strings to phonemize
        phoneme_dict: Phoneme dictionary for lookup
        skip_normalize: If True, skip normalization (use when text is pre-normalized)
    
    Returns:
        List of phonemized texts
    """
    if not texts:
        return []

    if skip_normalize:
        normalized_texts = texts
    else:
        normalized_texts = [normalizer.normalize(text) for text in texts]
    
    # Using dict for deduplication while preserving order (in Python 3.7+)
    unique_en_segments = {}
    unique_vi_cores = {}
    
    # Structure to hold intermediate results
    # Each entry: [ [parts], [parts] ... ] where parts can be a string (EN) or list (VI words)
    results = []
    
    for text_idx, text in enumerate(normalized_texts):
        # Split by <en> tags
        parts = re.split(r'(<en>.*?</en>)', text, flags=re.IGNORECASE)
        processed_parts = []
        
        for part_idx, part in enumerate(parts):
            if not part:
                processed_parts.append("")
                continue

            if re.match(r'<en>.*</en>', part, re.IGNORECASE):
                en_content = re.sub(r'</?en>', '', part, flags=re.IGNORECASE).strip()
                if en_content not in unique_en_segments:
                    unique_en_segments[en_content] = None
                processed_parts.append({'type': 'en', 'content': en_content})
            else:
                words = part.split()
                processed_words = []
                
                for word in words:
                    match = re.match(r'^(\W*)(.*?)(\W*)$', word)
                    pre, core, suf = match.groups() if match else ("", word, "")
                    
                    if not core:
                        processed_words.append({'type': 'fixed', 'content': word})
                    elif core in phoneme_dict:
                        processed_words.append({'type': 'fixed', 'content': f"{pre}{phoneme_dict[core]}{suf}"})
                    else:
                        if core not in unique_vi_cores:
                            unique_vi_cores[core] = None
                        processed_words.append({'type': 'vi_core', 'pre': pre, 'core': core, 'suf': suf})
                
                processed_parts.append({'type': 'vi_words', 'content': processed_words})
        
        results.append(processed_parts)
    
    # 1. Phonemize unique EN segments in one batch
    if unique_en_segments:
        en_list = list(unique_en_segments.keys())
        try:
            en_phonemes = phonemize(
                en_list,
                language='en-us',
                backend='espeak',
                preserve_punctuation=True,
                with_stress=True,
                language_switch="remove-flags"
            )
            if isinstance(en_phonemes, str):
                en_phonemes = [en_phonemes]
            
            for original, phoneme in zip(en_list, en_phonemes):
                unique_en_segments[original] = phoneme.strip()
        except Exception as e:
            logger.warning(f"Warning: Batch EN phonemization failed: {e}")
            for original in en_list:
                unique_en_segments[original] = original

    # 2. Phonemize unique VI cores in one batch
    if unique_vi_cores:
        vi_list = list(unique_vi_cores.keys())
        try:
            vi_phonemes = phonemize(
                vi_list,
                language='vi',
                backend='espeak',
                preserve_punctuation=True,
                with_stress=True,
                language_switch='remove-flags'
            )
            if isinstance(vi_phonemes, str):
                vi_phonemes = [vi_phonemes]
            
            for original, phoneme in zip(vi_list, vi_phonemes):
                ph = phoneme.strip()
                # Special rule for 'r' starting words
                if original.lower().startswith('r') and ph:
                    ph = 'ɹ' + ph[1:]
                
                unique_vi_cores[original] = ph
                phoneme_dict[original] = ph # Cache for future
        except Exception as e:
            logger.warning(f"Warning: Batch VI phonemization failed: {e}")
            for original in vi_list:
                unique_vi_cores[original] = original

    # 3. Assemble final results
    final_results = []
    for processed_parts in results:
        text_parts = []
        for part in processed_parts:
            if not part: continue

            if part['type'] == 'en':
                text_parts.append(unique_en_segments.get(part['content'], part['content']))
            elif part['type'] == 'vi_words':
                word_list = []
                for w in part['content']:
                    if w['type'] == 'fixed':
                        word_list.append(w['content'])
                    else:
                        ph = unique_vi_cores.get(w['core'], w['core'])
                        word_list.append(f"{w['pre']}{ph}{w['suf']}")
                text_parts.append(" ".join(word_list))
        
        full_text = " ".join(text_parts)
        # Cleanup spaces before punctuation
        full_text = re.sub(r'\s+([.,!?;:])', r'\1', full_text)
        final_results.append(full_text)
    
    return final_results