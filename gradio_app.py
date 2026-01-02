import os

from ui.app_layout import create_demo


demo = create_demo()

if __name__ == "__main__":
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.queue().launch(server_name=server_name, server_port=server_port)
