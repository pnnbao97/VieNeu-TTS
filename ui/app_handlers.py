import gradio as gr

from app_config import BACKBONE_CONFIGS


def update_info(backbone: str) -> str:
    return f"Streaming: {'?' if BACKBONE_CONFIGS[backbone]['supports_streaming'] else '?'}"


def set_mode_preset():
    return "preset_mode", gr.update(visible=True)


def set_mode_custom():
    return "custom_mode", gr.update(visible=True)


def set_mode_multi():
    return "multi_mode", gr.update(visible=False)
