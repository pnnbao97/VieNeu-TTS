# 🦜 VieNeu-TTS

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) with **instant voice cloning** and **English–Vietnamese bilingual** support. The SDK **defaults to VieNeu-TTS v3 Turbo (48 kHz)** and the minimal install is **torch-free** — on CPU it runs entirely on ONNX Runtime.

[![Hugging Face v3 Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3%20Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> **Looking for VieNeu-TTS v4?** v4 is not in this package. It is a hosted API: docs at [docs.vieneu.io](https://docs.vieneu.io) ([llms.txt](https://docs.vieneu.io/llms.txt), [OpenAPI](https://docs.vieneu.io/openapi.json)), base URL `https://api.vieneu.io/api/v1`, OpenAI-compatible `POST /v1/audio/speech`.

## ✨ Key Features
- **v3 Turbo, 48 kHz** — high-fidelity, natural Vietnamese speech (default).
- **Torch-free on CPU** — minimal install runs on ONNX Runtime; PyTorch is never imported.
- **fp32 backbone by default on CPU** — maximum fidelity. Use `Vieneu(precision="int8")` for ~1.6× speed & ~4× smaller download (needs a VNNI-capable CPU). Need it much faster, or on a phone / ARM board? See [v3 Nano (preview)](#v3-nano) — `Vieneu(mode="v3nano")`, lower quality but ~3× faster than Turbo fp32.
- **Built-in default voices** — call them by name, no reference clip needed.
- **Instant voice cloning** — clone any voice from 3–5s of audio.
- **Emotion cues** *(experimental)* — drop `[cười]`, `[thở dài]`, `[hắng giọng]` into the text.
- **Bilingual (En–Vi) code-switching**, fully offline.

---

## 📦 Install

**CPU (default)** — torch-free, runs v3 Turbo via ONNX Runtime. Most users want this:

```bash
pip install vieneu
```

**GPU (CUDA)** — only if you have an NVIDIA GPU. Install a CUDA build of PyTorch
**yourself first**. Batching then turns on automatically on CUDA — same API, no code change:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6"   # Qwen3 backbone + MOSS codec (pinned — most stable for the GPU SDK)
pip install vieneu
```

> ℹ️ **When is GPU actually worth it?** The GPU win comes from **batching**, so it
> only pays off on **long text** (many chunks generated together in one forward —
> long-form or bulk synthesis). For **short text** the torch-free **CPU/ONNX** path
> is usually *faster* (there's no batch to fill). Use CPU for short, interactive
> calls; reach for GPU for long-form or high-throughput work.

---

## 🚀 Quick Start (Python SDK)

```python
from vieneu import Vieneu

# Default = v3 Turbo (48 kHz). GPU → PyTorch (auto-detected).
# On CPU the backbone runs fp32 by default (max quality); pass precision="int8" for speed (VNNI CPU).
vieneu = Vieneu()                    # fp32 backbone (default, max quality)
# vieneu = Vieneu(precision="int8")  # int8 backbone (faster on CPU with VNNI, ~4x smaller)
# vieneu = Vieneu(mode="v3nano")     # v3 Nano (preview): fastest, lower quality — see "v3 Nano" below
# 💡 On a GPU machine you can still switch to ONNX/CPU if you prefer: Vieneu(backend="onnx")

# 1. Built-in voice by name — no reference needed
print("🔊 Generating speech...")
audio = vieneu.infer("Xin chào, đây là VieNeu-TTS.", voice="Hải Đăng")
vieneu.save(audio, "output.wav")
print("✅ Saved to output.wav")

# List the built-in voices
voices = vieneu.list_preset_voices()
print(f"\n🎙️  {len(voices)} built-in voices available:")
for label, voice_id in voices:
    print(f"  - {label} ({voice_id})")

# 2. Reading style: DEPRECATED on v3 Turbo — `style` is accepted but IGNORED. The style
#    is already implied by the reference (preset voice / cloned clip), so output is
#    always the natural reading style. Just pick the voice:
audio = vieneu.infer("Bản tin sáng nay.", voice="Hải Đăng")

# 3. Emotion / non-verbal cues — EXPERIMENTAL: [cười] [thở dài] [hắng giọng]
audio = vieneu.infer("Nghe hay quá đi [cười].", voice="Hải Đăng")

# 4. ⚡ Batch on GPU: infer_batch() runs many texts in ONE batched forward — same API.
#    On a CUDA GPU the chunks from every text share each forward step (big throughput
#    win); on CPU it still works (no error), just sequentially. Batch caps at
#    max_batch_size (default 32; or infer_batch(..., batch_size=64); batch_size=1
#    disables). A single long infer() also auto-batches its own chunks. Uncomment to try:
#
# import time
# texts = [
#     "Chào cả nhà, hôm nay mình sẽ hướng dẫn các bạn cách cài đặt và sử dụng bộ giọng đọc mới.",
#     "Giọng nghe cực kỳ tự nhiên và truyền cảm, lại có thể chuyển đổi biểu cảm một cách linh hoạt.",
#     "Nếu thấy hữu ích, các bạn nhớ để lại một lượt thích và chia sẻ video này cho mọi người nhé!",
# ] * 10   # 30 texts — enough to fill the batch and really show the GPU throughput win
# t0 = time.time()
# audios = vieneu.infer_batch(texts, voice="Hải Đăng")
# elapsed = time.time() - t0
# total_audio = sum(len(a) for a in audios) / 48_000
# print(f"⚡ {len(texts)} texts | audio {total_audio:.1f}s | wall {elapsed:.1f}s | RTF {elapsed/total_audio:.3f}")
# for i, a in enumerate(audios):
#     vieneu.save(a, f"batch_{i}.wav")
```

### 🔊 Real-time streaming

v3 Turbo streams frame-by-frame (first audio ~300 ms, RTF < 1 on CPU). Streaming runs on the **ONNX/CPU** engine — the GPU/PyTorch engine is for **batch throughput**, not streaming, so pin `backend="onnx"` for realtime. Iterate `infer_stream`:

```python
vieneu = Vieneu(backend="onnx")   # force ONNX/CPU — the streaming path (int8)
for chunk in vieneu.infer_stream("Xin chào các bạn!", voice="Hải Đăng"):
    play(chunk)   # np.float32 @ 48 kHz, play/write as it arrives
```

A full FastAPI web demo is in [`apps/web_stream.py`](apps/web_stream.py) (`uv run python -m apps.web_stream` → http://localhost:8001).

### 🦜 Zero-shot Voice Cloning

Clone from a short clip; the reference is auto-denoised and trimmed to ≤ 8s.

```python
from vieneu import Vieneu
vieneu = Vieneu()

# Clone straight from a 3–8s clip
audio = vieneu.infer("Chào bạn, đây là giọng của tôi.", ref_audio="path/to/voice.wav", denoise=True)
vieneu.save(audio, "cloned.wav")

# Save a cloned voice and reuse it by name
vieneu.add_voice("Giọng của tôi", "path/to/voice.wav")
audio = vieneu.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Just clean up a clip (no synthesis)
wav, sr = vieneu.denoise("noisy.wav", out_path="clean.wav")
```

> `denoise`, `add_voice`, and cloning work on every backend, including the torch-free CPU/ONNX install — except **v3 Nano** (preset voices only).

<a id="v3-nano"></a>
### v3 Nano (preview) — edge devices / weak CPUs only

v3 Turbo stays the default. `Vieneu(mode="v3nano")` loads a 48M-parameter flow model (ONNX, CPU, 24 kHz) for machines where Turbo is too slow: on the same desktop CPU it runs at **RTF 0.22** (16 steps) or **0.11** (`steps=8, sway=-1`) vs 0.37 for Turbo int8 and 0.62 for Turbo fp32. It is **lower quality than Turbo, especially on English and code-switched text**, ships **6 preset voices only (no cloning)**, and has no frame-level streaming.

```python
tts = Vieneu(mode="v3nano")
audio = tts.infer("Xin chào, mình là giọng đọc của VieNeu Nano.", voice="Minh Quân")   # 24 kHz
```

---

## 🔬 Model Overview

| Model | Engine | Device | Sample Rate | Features |
|---|---|---|---|---|
| **VieNeu-TTS v3 Turbo** *(default)* | ONNX (CPU) / PyTorch (GPU) | CPU/GPU | 48 kHz | Default voices, cloning, emotion cues |
| **VieNeu-TTS v3 Nano** *(preview, weak CPUs)* | ONNX (CPU) | weak CPU / edge | 24 kHz | 6 preset voices, emotion cues — no cloning, weaker English / En-Vi |

---

## 🤝 Support & Links
- **GitHub:** [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)

**Made with ❤️ for the Vietnamese TTS community**
