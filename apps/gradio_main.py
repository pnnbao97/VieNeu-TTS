import os
from .ui_shared import build_common_ui, env_bool

demo = build_common_ui(mode='standard')

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=env_bool("GRADIO_SHARE", False)
    )
