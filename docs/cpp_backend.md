# ⚡ High-Performance C++ Backend Integration (audio.cpp)

VieNeu-TTS supports an ultra-fast, lightweight, and torch-free C++ backend mode using the `audio.cpp` model engine. 

By leveraging the GGUF model quantization and C++ execution, this backend achieves:
- **Instant model load & warmup times** (<0.08s compared to ~11s in PyTorch/ONNX).
- **Sub-second audio generation** (RTF ~ 0.24, which is **4x faster than real-time** on commodity CPU).
- **Minimal memory footprints** suitable for offline or low-powered environments.

## 📊 Performance Benchmarks

The following benchmark compares the native CPU backends of VieNeu-TTS v3 Turbo (PyTorch CPU, ONNX CPU int8, and ONNX CPU fp32) against the compiled **C++ CPU** backend (using 4 threads, generating **~3.5 seconds of high-fidelity 48 kHz audio**):

| Metric | Native PyTorch CPU | Upstream ONNX CPU (`onnx_int8`) | Upstream ONNX CPU (`onnx_update` / FP32) | C++ CPU (`audio.cpp`) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Load Time** | `9.30 seconds` | `8.93 seconds` | `17.04 seconds` | **`Instant (<0.01s)`** |
| **First Run (Warmup)** | `1.70 seconds` | `1.57 seconds` | `1.40 seconds` | **`0.07 seconds`** |
| **Inference Time (Avg)** | `1.58 seconds` | `1.51 seconds` | `2.80 seconds` | **`0.89 seconds`** |
| **Real-Time Factor (RTF)** | `0.44` | `0.43` | `0.80` | **`0.24`** *(4x faster)* |
| **Combined Load+Warmup** | `11.00 seconds` | `10.50 seconds` | `18.44 seconds` | **`0.08 seconds`** |

> 💡 *Note: The C++ backend achieves sub-second overall latency and instant startup out-of-the-box by avoiding Python interpreter and engine runtime initialization times, making it perfect for real-time edge integration.*

---

## 🛠️ Setup Instructions

### 1. Clone and Build the C++ Engine

Clone the C++ model runner repository and build the target binary:

```bash
# Clone the repository
git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp

# Compile for CPU backend (use --backend cuda for GPU builds)
./scripts/build_linux.sh --backend cpu --target audiocpp_cli
```
This produces the compiled CLI executable at `build/linux-cpu-release/bin/audiocpp_cli`.

### 2. Download the GGUF Model Weights

Download the GGUF model weights for VieNeu-TTS v3 Turbo and place them inside the `audio.cpp` model folder:

```bash
# Create target weights directory
mkdir -p models/VieNeu-TTS-v3-Turbo

# Download model.gguf
wget -O models/VieNeu-TTS-v3-Turbo/model.gguf \
  https://huggingface.co/phuocnguyen90/VieNeu-TTS-v3-Turbo-GGUF/resolve/main/model.gguf
```

---

## 📦 Using the Python SDK Wrapper

Once compiled, you can initialize the C++ backend wrapper directly using `Vieneu(mode="cpp")`:

```python
from vieneu import Vieneu

# Initialize C++ engine
# (Auto-discovers the C++ binary and GGUF model if cloned to standard paths)
tts = Vieneu(mode="cpp")

# Run inference
text = "Xin chào thế giới. Đây là bản thử nghiệm."
audio = tts.infer(text)

# Save output audio
tts.save(audio, "output_cpp.wav")
```

### Advanced Options (Manual Path Configuration)

If you have cloned or built the engine/model at non-default paths, configure them explicitly during initialization or via environment variables:

```python
tts = Vieneu(
    mode="cpp",
    binary_path="/path/to/your/audiocpp_cli",
    model_path="/path/to/your/model.gguf",
    threads=4
)
```

Alternatively, configure the variables in your environment:
```bash
export VIENEU_CPP_BINARY_PATH="/path/to/audiocpp_cli"
export VIENEU_CPP_MODEL_PATH="/path/to/model.gguf"
export VIENEU_CPP_REF_AUDIO="/path/to/sample.wav"
```
