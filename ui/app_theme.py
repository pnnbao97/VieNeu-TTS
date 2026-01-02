import gradio as gr


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
    )
