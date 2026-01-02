from pathlib import Path


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"
MULTI_VOICE_JS_PATH = ASSETS_DIR / "multi_voice.js"
HEADER_HTML_PATH = ASSETS_DIR / "header.html"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_css() -> str:
    return _read_text(CSS_PATH)


def load_multi_voice_head() -> str:
    multi_voice_js = _read_text(MULTI_VOICE_JS_PATH)
    return f"<script>({multi_voice_js})();</script>"


def load_header_html() -> str:
    return _read_text(HEADER_HTML_PATH)
