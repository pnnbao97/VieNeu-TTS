import os
import sys
import subprocess
import torch
from .ui_shared import build_common_ui, env_bool

intel_dll_path = os.path.join(sys.prefix, "Library", "bin")
if os.path.exists(intel_dll_path): os.environ["PATH"] = intel_dll_path + os.pathsep + os.environ["PATH"]

try:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "gradio_main.py")] + sys.argv[1:]); sys.exit(0)
except ImportError: pass

demo = build_common_ui(mode='xpu')

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=env_bool("GRADIO_SHARE", False)
    )
