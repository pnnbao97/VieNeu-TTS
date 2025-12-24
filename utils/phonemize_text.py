import os
import json
import platform
import glob
from phonemizer import phonemize
from phonemizer.backend.espeak.espeak import EspeakWrapper
from utils.normalize_text import VietnameseTTSNormalizer

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
        raise OSError(
            f"Unsupported OS: {system}. "
            "Only Windows, Linux, and macOS are supported."
        )

def _setup_windows_espeak():
    """Setup eSpeak for Windows."""
    default_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    if os.path.exists(default_path):
        EspeakWrapper.set_library(default_path)
    else:
        raise FileNotFoundError(
            f"eSpeak library not found at {default_path}. "
            "Please install eSpeak NG from: https://github.com/espeak-ng/espeak-ng/releases"
        )

def _setup_linux_espeak():
    """Setup eSpeak for Linux."""
    # First check environment variable
    espeak_lib = os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')
    if espeak_lib and os.path.exists(espeak_lib):
        EspeakWrapper.set_library(espeak_lib)
        return
    
    # Search common library paths
    search_patterns = [
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so*",  # ARM64
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so*",    # x86_64
        "/usr/lib/arm-linux-gnueabihf/libespeak-ng.so*", # ARM32
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
    
    raise RuntimeError(
        "eSpeak NG library not found. Install with:\n"
        "  Ubuntu/Debian: sudo apt-get install espeak-ng\n"
        "  Fedora: sudo dnf install espeak-ng\n"
        "  Arch: sudo pacman -S espeak-ng\n"
        "See: https://github.com/pnnbao97/VieNeu-TTS/issues/5"
    )

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
    
    raise FileNotFoundError(
        "eSpeak library not found. Install with:\n"
        "  brew install espeak-ng\n"
        "Or set: export PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.dylib"
    )

# Initialize
try:
    setup_espeak_library()
    phoneme_dict = load_phoneme_dict()
    normalizer = VietnameseTTSNormalizer()
except Exception as e:
    print(f"Initialization error: {e}")
    raise

def phonemize_text(text: str) -> str:
    """Convert text to phonemes using phonemizer."""
    text = normalizer.normalize(text)
    return phonemize(
        text,
        language="vi",
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags"
    )

def phonemize_with_dict(text: str, phoneme_dict=phoneme_dict) -> str:
    """Phonemize text with dictionary lookup."""
    text = normalizer.normalize(text)
    words = text.split()
    result = []
    
    for word in words:
        if word in phoneme_dict:
            phone_word = phoneme_dict[word]
        else:
            try:
                phone_word = phonemize(
                    word,
                    language='vi',
                    backend='espeak',
                    preserve_punctuation=True,
                    with_stress=True,
                    language_switch='remove-flags'
                )
                
                if word.lower().startswith('r'):
                    phone_word = 'ɹ' + phone_word[1:]
                
                phoneme_dict[word] = phone_word
            except Exception as e:
                print(f"Warning: Could not phonemize '{word}': {e}")
                phone_word = word
        
        result.append(phone_word)
    
    return ' '.join(result)