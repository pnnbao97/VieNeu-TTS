import os
from utils.core_utils import env_bool

from ui.app_layout import create_demo


demo = create_demo()

if __name__ == "__main__":
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    #
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
