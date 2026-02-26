import os
import json
import platform
import glob
import re
from phonemizer import phonemize
from phonemizer.backend.espeak.espeak import EspeakWrapper
from vieneu_utils.normalize_text import VietnameseTTSNormalizer

# Configuration
PHONEME_DICT_PATH = os.getenv(
    'PHONEME_DICT_PATH',
    os.path.join(os.path.dirname(__file__), "phoneme_dict.json")
)

def load_phoneme_dict(path=PHONEME_DICT_PATH):
    """Load phoneme dictionary from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phoneme dictionary not found at {path}. "
            "Please create it or set PHONEME_DICT_PATH environment variable."
        )

def setup_espeak_library():
    """Configure eSpeak library path based on operating system."""
    system = platform.system()
    
    if system == "Windows":
        _setup_windows_espeak()
    elif system == "Linux":
        _setup_linux_espeak()
    elif system == "Darwin":
        _setup_macos_espeak()
    else:
        print(f"Warning: Unsupported OS: {system}")
        return

def _setup_windows_espeak():
    """Setup eSpeak for Windows."""
    default_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    if os.path.exists(default_path):
        EspeakWrapper.set_library(default_path)
    else:
        print("⚠️ eSpeak-NG is not installed. The system will use the built-in dictionary, but it is recommended to install eSpeak-NG for maximum performance and accuracy.")

def _setup_linux_espeak():
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
    
    print("⚠️ eSpeak-NG is not installed on Linux. The system will use the built-in dictionary, but it is recommended to install eSpeak-NG (sudo apt install espeak-ng) for maximum performance.")

def _setup_macos_espeak():
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
    
    print("⚠️ eSpeak-NG is not installed on macOS. The system will use the built-in dictionary, but it is recommended to install eSpeak-NG (brew install espeak-ng) for maximum performance.")

# Pre-compiled regex patterns
EN_PART_RE = re.compile(r'(<en>.*?</en>)', re.IGNORECASE)
EN_CONTENT_RE = re.compile(r'<en>(.*?)</en>', re.IGNORECASE)
WORD_BOUND_RE = re.compile(r'^(\W*)(.*?)(\W*)$')
CLEANUP_RE = re.compile(r'\s+([.,!?;:])')

# Initialize
setup_espeak_library()

try:
    phoneme_dict = load_phoneme_dict()
    normalizer = VietnameseTTSNormalizer()
except Exception as e:
    print(f"Initialization error: {e}")
    normalizer = VietnameseTTSNormalizer()
    phoneme_dict = {}

def phonemize_text(text: str) -> str:
    """Convert text to phonemes (backward compatibility)."""
    text = normalizer.normalize(text)
    return phonemize(
        text,
        language="vi",
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags"
    )

def phonemize_with_dict(text: str, phoneme_dict=phoneme_dict, skip_normalize: bool = False) -> str:
    """Phonemize single text with dictionary lookup and unique word optimization."""
    if not skip_normalize:
        text = normalizer.normalize(text)
    
    parts = EN_PART_RE.split(text)
    processed_parts = []
    
    en_texts, en_indices = [], []
    vi_word_info = [] # list of (part_idx, word_idx, pre, core, suf)
    unknown_cores = set()

    for part_idx, part in enumerate(parts):
        en_match = EN_CONTENT_RE.match(part)
        if en_match:
            en_content = en_match.group(1).strip()
            en_texts.append(en_content)
            en_indices.append(part_idx)
            processed_parts.append(None)
        else:
            words = part.split()
            processed_words = []
            for word_idx, word in enumerate(words):
                match = WORD_BOUND_RE.match(word)
                pre, core, suf = match.groups() if match else ("", word, "")
                
                if not core:
                    processed_words.append(word)
                elif core in phoneme_dict:
                    processed_words.append(f"{pre}{phoneme_dict[core]}{suf}")
                else:
                    unknown_cores.add(core)
                    vi_word_info.append((part_idx, word_idx, pre, core, suf))
                    processed_words.append(None)
            processed_parts.append(processed_words)
    
    # Handle English
    if en_texts:
        try:
            en_phonemes = phonemize(en_texts, language='en-us', backend='espeak', preserve_punctuation=True, with_stress=True, language_switch="remove-flags")
            if isinstance(en_phonemes, str): en_phonemes = [en_phonemes]
            for part_idx, ph in zip(en_indices, en_phonemes):
                processed_parts[part_idx] = ph.strip()
        except Exception as e:
            print(f"Warning: EN phonemization failed: {e}")
            for part_idx, content in zip(en_indices, en_texts): processed_parts[part_idx] = content

    # Handle unknown Vietnamese cores
    if unknown_cores:
        unique_cores = sorted(list(unknown_cores))
        try:
            vi_phonemes = phonemize(unique_cores, language='vi', backend='espeak', preserve_punctuation=True, with_stress=True, language_switch='remove-flags')
            if isinstance(vi_phonemes, str): vi_phonemes = [vi_phonemes]
            
            # Update dictionary with unique results
            for core, ph in zip(unique_cores, vi_phonemes):
                ph = ph.strip()
                if core.lower().startswith('r') and ph:
                    ph = 'ɹ' + ph[1:]
                phoneme_dict[core] = ph
            
            # Fill in processed_parts
            for p_idx, w_idx, pre, core, suf in vi_word_info:
                processed_parts[p_idx][w_idx] = f"{pre}{phoneme_dict[core]}{suf}"
        except Exception as e:
            print(f"Warning: VI phonemization failed: {e}")
            for p_idx, w_idx, pre, core, suf in vi_word_info:
                processed_parts[p_idx][w_idx] = f"{pre}{core}{suf}"

    final_results = []
    for part in processed_parts:
        if isinstance(part, list): final_results.append(' '.join(str(w) for w in part if w is not None))
        elif part is not None: final_results.append(part)
    
    res = ' '.join(final_results)
    return CLEANUP_RE.sub(r'\1', res)

def phonemize_batch(texts: list, phoneme_dict=phoneme_dict, skip_normalize: bool = False) -> list:
    """Phonemize multiple texts with optimal batching and unique word filtering."""
    if not skip_normalize:
        normalized_texts = [normalizer.normalize(text) for text in texts]
    else:
        normalized_texts = texts
    
    all_en_texts, all_en_maps = [], []
    all_vi_word_info = [] # (text_idx, part_idx, word_idx, pre, core, suf)
    unknown_cores = set()
    results = []
    
    for text_idx, text in enumerate(normalized_texts):
        parts = EN_PART_RE.split(text)
        processed_parts = []
        for part_idx, part in enumerate(parts):
            en_match = EN_CONTENT_RE.match(part)
            if en_match:
                en_content = en_match.group(1).strip()
                all_en_texts.append(en_content)
                all_en_maps.append((text_idx, part_idx))
                processed_parts.append(None)
            else:
                words = part.split()
                p_words = []
                for word_idx, word in enumerate(words):
                    match = WORD_BOUND_RE.match(word)
                    pre, core, suf = match.groups() if match else ("", word, "")
                    if not core: p_words.append(word)
                    elif core in phoneme_dict: p_words.append(f"{pre}{phoneme_dict[core]}{suf}")
                    else:
                        unknown_cores.add(core)
                        all_vi_word_info.append((text_idx, part_idx, word_idx, pre, core, suf))
                        p_words.append(None)
                processed_parts.append(p_words)
        results.append(processed_parts)
    
    if all_en_texts:
        try:
            en_ph = phonemize(all_en_texts, language='en-us', backend='espeak', preserve_punctuation=True, with_stress=True, language_switch="remove-flags")
            if isinstance(en_ph, str): en_ph = [en_ph]
            for (t_idx, p_idx), ph in zip(all_en_maps, en_ph): results[t_idx][p_idx] = ph.strip()
        except Exception: pass
    
    if unknown_cores:
        unique_cores = sorted(list(unknown_cores))
        try:
            vi_ph = phonemize(unique_cores, language='vi', backend='espeak', preserve_punctuation=True, with_stress=True, language_switch='remove-flags')
            if isinstance(vi_ph, str): vi_ph = [vi_ph]
            for core, ph in zip(unique_cores, vi_ph):
                ph = ph.strip()
                if core.lower().startswith('r') and ph: ph = 'ɹ' + ph[1:]
                phoneme_dict[core] = ph
            for t_idx, p_idx, w_idx, pre, core, suf in all_vi_word_info:
                results[t_idx][p_idx][w_idx] = f"{pre}{phoneme_dict[core]}{suf}"
        except Exception:
            for t_idx, p_idx, w_idx, pre, core, suf in all_vi_word_info:
                results[t_idx][p_idx][w_idx] = f"{pre}{core}{suf}"
                
    final_results = []
    for pr in results:
        fps = []
        for p in pr:
            if isinstance(p, list): fps.append(' '.join(str(w) for w in p if w is not None))
            elif p is not None: fps.append(p)
        res = ' '.join(fps)
        final_results.append(CLEANUP_RE.sub(r'\1', res))
    return final_results
