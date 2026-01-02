import gradio as gr

from app_config import GGUF_ALLOWED_VOICES, SPEAKER_MODE_LABELS, VOICE_SAMPLES


def get_voice_options(backbone_choice: str):
    """Filter voice options: GGUF only shows the 4 allowed voices."""
    if "gguf" in backbone_choice.lower():
        return [v for v in GGUF_ALLOWED_VOICES if v in VOICE_SAMPLES]
    return list(VOICE_SAMPLES.keys())


def update_voice_dropdown(backbone_choice: str, current_voice: str):
    options = get_voice_options(backbone_choice)
    new_value = current_voice if current_voice in options else (options[0] if options else None)
    return gr.update(choices=options, value=new_value)


def update_multi_mode_visibility(mode_label):
    is_preset = mode_label == "Giọng mẫu"
    return (
        gr.update(visible=is_preset),
        gr.update(visible=not is_preset),
        gr.update(visible=not is_preset),
    )


def update_multi_voice_dropdowns(backbone_choice, *current_values):
    options = get_voice_options(backbone_choice)
    updates = []
    for value in current_values:
        new_value = value if value in options else (options[0] if options else None)
        updates.append(gr.update(choices=options, value=new_value))
    return updates


def update_multi_line_visibility(line_count, max_lines):
    try:
        line_count = int(line_count)
    except (TypeError, ValueError):
        line_count = 1
    line_count = max(1, min(line_count, max_lines))
    return [gr.update(visible=i < line_count) for i in range(max_lines)]


def _order_string(line_count):
    return ",".join(str(i) for i in range(1, line_count + 1))


def add_multi_line_with_order(line_count, max_lines):
    try:
        line_count = int(line_count)
    except (TypeError, ValueError):
        line_count = 1
    new_count = min(line_count + 1, max_lines)
    return [new_count] + update_multi_line_visibility(new_count, max_lines) + [_order_string(new_count)]


def remove_multi_line_with_order(line_count, max_lines):
    try:
        line_count = int(line_count)
    except (TypeError, ValueError):
        line_count = 1
    new_count = max(line_count - 1, 1)
    return [new_count] + update_multi_line_visibility(new_count, max_lines) + [_order_string(new_count)]


def clear_multi_lines_with_text(max_lines):
    new_count = 1
    updates = [new_count] + update_multi_line_visibility(new_count, max_lines)
    updates.extend([gr.update(value="") for _ in range(max_lines)])
    updates.append(_order_string(new_count))
    return updates


def remove_multi_line_at(line_index, line_count, backbone_choice, max_lines, *line_inputs):
    try:
        line_count = int(line_count)
    except (TypeError, ValueError):
        line_count = 1

    line_fields = 5
    options = get_voice_options(backbone_choice)
    default_voice = options[0] if options else None

    lines = []
    for i in range(max_lines):
        base = i * line_fields
        if base + line_fields <= len(line_inputs):
            lines.append(list(line_inputs[base:base + line_fields]))
        else:
            lines.append([SPEAKER_MODE_LABELS[0], default_voice, None, "", ""])

    if line_count <= 1:
        lines[0] = [SPEAKER_MODE_LABELS[0], default_voice, None, "", ""]
        new_count = 1
    else:
        if 0 <= line_index < line_count:
            for i in range(line_index, line_count - 1):
                lines[i] = lines[i + 1]
            lines[line_count - 1] = [SPEAKER_MODE_LABELS[0], default_voice, None, "", ""]
            new_count = line_count - 1
        else:
            new_count = line_count

    updates = [new_count] + update_multi_line_visibility(new_count, max_lines)
    for i in range(max_lines):
        mode_label, voice_choice, custom_audio, custom_text, line_text = lines[i]
        updates.extend([mode_label, voice_choice, custom_audio, custom_text, line_text])

    updates.append(_order_string(new_count))
    return updates
