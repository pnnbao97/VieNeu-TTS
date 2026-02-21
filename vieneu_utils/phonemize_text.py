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

# Initialize
setup_espeak_library()

try:
    phoneme_dict = load_phoneme_dict()
    normalizer = VietnameseTTSNormalizer()
except Exception as e:
    print(f"Initialization error: {e}")
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


def phonemize_with_dict(text: str, phoneme_dict=phoneme_dict, skip_normalize: bool = False) -> str:
    """
    Phonemize single text with dictionary lookup and EN tag support.
    """
    if not skip_normalize:
        text = normalizer.normalize(text)
    
    # Split by EN tags
    parts = re.split(r'(<en>.*?</en>)', text, flags=re.IGNORECASE)
    
    en_texts = []
    en_indices = []
    vi_texts = []
    vi_indices = []
    vi_word_maps = []
    
    processed_parts = []
    
    for part_idx, part in enumerate(parts):
        if re.match(r'<en>.*</en>', part, re.IGNORECASE):
            # English part
            en_content = re.sub(r'</?en>', '', part, flags=re.IGNORECASE).strip()
            en_texts.append(en_content)
            en_indices.append(part_idx)
            processed_parts.append(None)
        else:
            # Vietnamese part
            words = part.split()
            processed_words = []
            
            for word_idx, word in enumerate(words):
                match = re.match(r'^(\W*)(.*?)(\W*)$', word)
                pre, core, suf = match.groups() if match else ("", word, "")
                
                if not core:
                    processed_words.append(word)
                elif core in phoneme_dict:
                    processed_words.append(f"{pre}{phoneme_dict[core]}{suf}")
                else:
                    vi_texts.append(word)
                    vi_indices.append(part_idx)
                    vi_word_maps.append((part_idx, len(processed_words)))
                    processed_words.append(None)
            
            processed_parts.append(processed_words)
    
    if en_texts:
        try:
            en_phonemes = phonemize(
                en_texts,
                language='en-us',
                backend='espeak',
                preserve_punctuation=True,
                with_stress=True,
                language_switch="remove-flags"
            )
            
            if isinstance(en_phonemes, str):
                en_phonemes = [en_phonemes]
            
            for idx, (part_idx, phoneme) in enumerate(zip(en_indices, en_phonemes)):
                processed_parts[part_idx] = phoneme.strip()
        except Exception as e:
            print(f"Warning: Could not phonemize EN texts: {e}")
            for part_idx in en_indices:
                processed_parts[part_idx] = en_texts[en_indices.index(part_idx)]
    
    if vi_texts:
        try:
            vi_phonemes = phonemize(
                vi_texts,
                language='vi',
                backend='espeak',
                preserve_punctuation=True,
                with_stress=True,
                language_switch='remove-flags'
            )
            
            if isinstance(vi_phonemes, str):
                vi_phonemes = [vi_phonemes]
            
            for idx, (part_idx, word_idx) in enumerate(vi_word_maps):
                phoneme = vi_phonemes[idx].strip()
                
                original_word = vi_texts[idx]
                if original_word.lower().startswith('r'):
                    phoneme = 'ɹ' + phoneme[1:] if len(phoneme) > 0 else phoneme
                
                phoneme_dict[original_word] = phoneme
                
                if processed_parts[part_idx] is not None:
                    processed_parts[part_idx][word_idx] = phoneme
        except Exception as e:
            print(f"Warning: Could not phonemize VI texts: {e}")
            for idx, (part_idx, word_idx) in enumerate(vi_word_maps):
                if processed_parts[part_idx] is not None:
                    processed_parts[part_idx][word_idx] = vi_texts[idx]
    
    final_parts = []
    for part in processed_parts:
        if isinstance(part, list):
            final_parts.append(' '.join(str(w) for w in part if w is not None))
        elif part is not None:
            final_parts.append(part)
    
    result = ' '.join(final_parts)
    
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)
    
    return result


def phonemize_batch(texts: list, phoneme_dict=phoneme_dict, skip_normalize: bool = False) -> list:
    """
    Phonemize multiple texts with optimal batching.
    
    Args:
        texts: List of text strings to phonemize
        phoneme_dict: Phoneme dictionary for lookup
        skip_normalize: If True, skip normalization (use when text is pre-normalized)
    
    Returns:
        List of phonemized texts
    """
    if skip_normalize:
        normalized_texts = texts
    else:
        normalized_texts = [normalizer.normalize(text) for text in texts]
    
    all_en_texts = []
    all_en_maps = []
    
    all_vi_texts = []
    all_vi_maps = []
    
    results = []
    
    for text_idx, text in enumerate(normalized_texts):
        parts = re.split(r'(<en>.*?</en>)', text, flags=re.IGNORECASE)
        processed_parts = []
        
        for part_idx, part in enumerate(parts):
            if re.match(r'<en>.*</en>', part, re.IGNORECASE):
                en_content = re.sub(r'</?en>', '', part, flags=re.IGNORECASE).strip()
                all_en_texts.append(en_content)
                all_en_maps.append((text_idx, part_idx))
                processed_parts.append(None)
            else:
                words = part.split()
                processed_words = []
                
                for word in words:
                    match = re.match(r'^(\W*)(.*?)(\W*)$', word)
                    pre, core, suf = match.groups() if match else ("", word, "")
                    
                    if not core:
                        processed_words.append(word)
                    elif core in phoneme_dict:
                        processed_words.append(f"{pre}{phoneme_dict[core]}{suf}")
                    else:
                        all_vi_texts.append(word)
                        all_vi_maps.append((text_idx, part_idx, len(processed_words)))
                        processed_words.append(None)
                
                processed_parts.append(processed_words)
        
        results.append(processed_parts)
    
    if all_en_texts:
        try:
            en_phonemes = phonemize(
                all_en_texts,
                language='en-us',
                backend='espeak',
                preserve_punctuation=True,
                with_stress=True,
                language_switch="remove-flags"
            )
            
            if isinstance(en_phonemes, str):
                en_phonemes = [en_phonemes]
            
            for (text_idx, part_idx), phoneme in zip(all_en_maps, en_phonemes):
                results[text_idx][part_idx] = phoneme.strip()
        except Exception as e:
            print(f"Warning: Batch EN phonemization failed: {e}")
    
    if all_vi_texts:
        try:
            vi_phonemes = phonemize(
                all_vi_texts,
                language='vi',
                backend='espeak',
                preserve_punctuation=True,
                with_stress=True,
                language_switch='remove-flags'
            )
            
            if isinstance(vi_phonemes, str):
                vi_phonemes = [vi_phonemes]
            
            for idx, (text_idx, part_idx, word_idx) in enumerate(all_vi_maps):
                phoneme = vi_phonemes[idx].strip()
                
                original_word = all_vi_texts[idx]
                if original_word.lower().startswith('r'):
                    phoneme = 'ɹ' + phoneme[1:] if len(phoneme) > 0 else phoneme
                
                phoneme_dict[original_word] = phoneme
                results[text_idx][part_idx][word_idx] = phoneme
        except Exception as e:
            print(f"Warning: Batch VI phonemization failed: {e}")
    
    final_results = []
    for processed_parts in results:
        final_parts = []
        for part in processed_parts:
            if isinstance(part, list):
                final_parts.append(' '.join(str(w) for w in part if w is not None))
            elif part is not None:
                final_parts.append(part)
        
        result = ' '.join(final_parts)
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        final_results.append(result)
    
    return final_results