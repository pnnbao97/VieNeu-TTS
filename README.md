# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face VieNeu-TTS-v3-Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3--Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![Hugging Face VieNeu-TTS-v2](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS** is the next generation of on-device Vietnamese TTS: **10,000+ hours** of bilingual training, **instant voice cloning**, a dedicated **Podcast/Conversation** mode, and **extremely fast** — real-time streaming with first audio in ~115 ms, **RTF ≈ 0.01–0.02** batched on a consumer GPU (RTX 3060), ~2× real-time on a plain CPU ([benchmarks](#benchmarks)).

> [!IMPORTANT]
> **🦜 VieNeu-TTS v4 — available on [vieneu.io](https://www.vieneu.io)**
>
> VieNeu-TTS v4 delivers **near-original voice cloning fidelity**, allowing a short reference clip to be reproduced with very high speaker similarity.
>
> Due to the strength of its voice-cloning capabilities and the potential for misuse, **v4 is proprietary and will not be open-sourced**. It is available exclusively through the **VieNeu API / vieneu.io**.
>
> **VieNeu-TTS v3 Turbo remains the latest open-source version available in this repository.** Future open-source releases, including potential v3.x updates, may also be published here.

> [!NOTE]
> **🦜 VieNeu-TTS v3 Turbo is officially released!**
> A brand-new architecture **designed and trained from scratch by Phạm Nguyễn Ngọc Bảo** (codec: [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano); phonemizer: [sea-g2p](https://github.com/pnnbao97/sea-g2p)):
> - **48 kHz** high-fidelity audio (up from 24 kHz).
> - **25 built-in preset voices** (10 editors' picks) — stable and consistent, no reference clip needed.
> - **Natural reading style** everywhere — the style follows the reference voice (the `style` argument is deprecated and ignored).
> - **Emotion / non-verbal cues** *(experimental)*: drop `[cười]`, `[thở dài]`, `[hắng giọng]` straight into the text.
> - **Batched generation** (batch size up to 32), including a multi-speaker **Conversation** mode that batches the whole script regardless of speaker.
> - **Instant voice cloning** from a 3–8s clip, with automatic reference denoising.
> - **Real-time streaming + OpenAI-compatible API + Docker** — `POST /v1/audio/speech` drop-in for the OpenAI SDK / Pipecat / LiveKit; first audio in **~115 ms** and **16 concurrent streams under 200 ms** on a single RTX 3060 (32 max), CPU-only streaming too. See [§3](#docker-remote), [§4 Benchmarks](#benchmarks) and [docs/streaming.md](docs/streaming.md).
> - **LoRA fine-tuning** — train your own voice or reading style on one consumer GPU ([§5](#finetune)).
>
> Try it in the Web UI (backbone **"VieNeu-TTS-v3-Turbo"**) or the SDK (`Vieneu(mode="v3turbo")`, the default).

<h3>🎬 Demos</h3>

> [!TIP]
> **Dubbing**, **Lecture**, **Tiktok Reel** and **Audiobook** features are exclusively available in the [official VieNeu App](https://www.vieneu.io/#/download). This GitHub repository only provides a simplified Gradio demo interface and the core SDK for developers.

<table>
  <!-- Hàng 1 -->
  <tr>
    <td align="center" width="35%">
      <b>Voice Cloning</b><br><br>
      <video src="https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd" controls width="100%"></video>
    </td>
    <td align="center" width="35%">
      <b>Dubbing</b><br><br>
      <video src="https://github.com/user-attachments/assets/5888aea1-4f32-4397-9dd9-9c7b743d31bd" controls width="100%"></video>
    </td>
    <!-- Cột phải: TikTok Reel chiếm 2 hàng -->
    <td align="center" width="30%" rowspan="2" style="vertical-align: middle;">
      <b>Tiktok Reel</b><br><br>
      <video src="https://github.com/user-attachments/assets/561a1c55-a298-4a78-a501-c778c2640013" controls width="100%"></video>
    </td>
  </tr>
  <!-- Hàng 2 -->
  <tr>
    <td align="center" width="35%">
      <b>Dubbing / Conversation</b><br><br>
      <video src="https://github.com/user-attachments/assets/28104b78-2d55-4914-85b7-5f425a7e99da" controls width="100%"></video>
    </td>
    <td align="center" width="35%">
      <b>Lecture</b><br><br>
      <video src="https://github.com/user-attachments/assets/de3b2d4e-4c50-4164-acdc-e3de90840e26" controls width="100%"></video>
    </td>
  </tr>
</table>

## 📌 Table of Contents

1. [🦜 Installation & Web UI](#installation)
2. [📦 Using the Python SDK](#sdk)
3. [🐳 API Server & Docker](#docker-remote) — OpenAI-compatible streaming API (v3 Turbo) · legacy v2 server
4. [📊 Benchmarks](#benchmarks) — every speed / latency number in one place (CPU vs GPU, batch, streaming, Nano)
5. [🎓 Fine-tuning (LoRA)](#finetune)
6. [🔬 Model Overview](#backbones)
7. [🚀 Roadmap](#roadmap)
8. [🤝 Support & Contact](#support)
9. [📑 Citation](#citation)

---

## 🦜 1. Installation & Web UI <a name="installation"></a>
> [!TIP]
> **On Windows?** The fastest way to get started is the standalone installer at **[vieneu.io/#/download](https://www.vieneu.io/#/download)** — no need to install `uv` or clone the repo manually.
> **macOS**: a similar installer is coming in an upcoming release; for now, please use the `uv sync` steps below.
> **Docker?** Skip the steps below: `--profile api-gpu` / `api-cpu` (OpenAI-compatible streaming API) — see [§3 API Server & Docker](#docker-remote).

### Setup with `uv` (Recommended)
`uv` is the fastest way to manage dependencies. 
```bash
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Install Dependencies:**
   > 📊 **Which one?** CPU ≈ **RTF 0.5** (2× real-time, one stream) · GPU ≈ **RTF 0.02** batched (~50× real-time) and **16 real-time streams** — all measurements in [§4 Benchmarks](#benchmarks).

   - **Option 1: CPU & macOS (minimal, torch-free)** — RTF ≈ 0.5, no GPU needed

     ```bash
     uv sync
     ```
   - **Option 2: GPU** — **v3 Turbo on GPU (PyTorch)** — RTF ≈ 0.02 batched, 16 real-time streams

     ```bash
     uv sync --extra cuda
     ```

3. **Start the Web UI:**
   ```bash
   uv run vieneu-web
   ```
   Access the UI at `http://127.0.0.1:7860`.

---


## 📦 2. Using the Python SDK (vieneu) <a name="sdk"></a>

The `vieneu` SDK **defaults to VieNeu-TTS v3 Turbo (48 kHz)**. The minimal install is **torch-free**: on CPU everything runs on **ONNX Runtime** (PyTorch is never imported), and on a CUDA machine it auto-switches to the PyTorch engine — where inference is **batched automatically** (same API, no code change).

### Quick Start

**CPU (default)** — torch-free, runs v3 Turbo via ONNX Runtime. Most users want this — **RTF ≈ 0.5** on a 12th-gen Core i5 (numbers in [§4 Benchmarks](#benchmarks)):
> ⚡**On CPU the backbone runs `fp32` by default** (maximum fidelity). Need more speed? Pass `Vieneu(precision="int8")` — ~1.6× faster (RTF ≈ 0.35) and ~4× smaller, but it requires a CPU with VNNI (AVX-512 VNNI / AVX-VNNI); on older CPUs int8 can produce garbled audio. `precision` only affects the CPU/ONNX path; on GPU it's ignored (PyTorch).
>
> 🪶 **Still too slow, or deploying on a phone / ARM board?** Try **[VieNeu-TTS v3 Nano (preview)](#v3-nano)** — `Vieneu(mode="v3nano")`, ~3× faster than Turbo fp32 on CPU (RTF 0.11–0.22 on a desktop CPU), but **noticeably lower quality** (especially English / bilingual), 24 kHz, 11 preset voices + voice cloning. Details and caveats in the [v3 Nano section](#v3-nano) below.

```bash
pip install vieneu
```

**GPU (CUDA)** — only if you have an NVIDIA GPU; **~25× faster than the CPU path** (RTF ≈ 0.02 batched, 16 real-time streams — [§4 Benchmarks](#benchmarks)). On Linux `pip install "vieneu[cuda]"` is enough (PyPI torch ships CUDA there); on Windows install the CUDA torch **first** as below.
> ⚡ Every audio frame on GPU is **one CUDA graph** (acoustic decoder + sampling +
> repetition penalty + backbone step in a single replay — no `torch.compile`, no
> C++ toolchain needed). The first call for each batch size pays ~0.5 s to capture
> the graph (kept afterwards; servers can call `warm_fused()` at start-up).
> `VIENEU_FUSED_FRAME=0` restores the plain loop.

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6"   # pinned — most stable transformers for the GPU SDK
pip install vieneu
```

```python
import time
from vieneu import Vieneu

# Default = v3 Turbo (48 kHz). GPU → PyTorch (auto-detected).
vieneu = Vieneu() # On a GPU machine you can still switch to ONNX/CPU if you prefer: Vieneu(backend="onnx")

# 1. Built-in voice by name — no reference clip needed
print("🔊 Generating speech...")

start_time = time.time()
audio = vieneu.infer("[cười] Trời ơi, cái giọng nó tự nhiên mà nó mượt mà dã man, nghe không khác gì người thật luôn. Giờ thì tha hồ mà quẩy content với cả kho giọng nói đa dạng, đủ mọi sắc thái biểu cảm. Mọi người bật loa lên rồi cùng trải nghiệm thử với mình nhé!", voice="Phạm Tuyên")
elapsed_time = time.time() - start_time

vieneu.save(audio, "output.wav")
print("✅ Saved to output.wav")

# Tính RTF (Real-Time Factor)
sample_rate = 48000
audio_duration = len(audio) / sample_rate
rtf = elapsed_time / audio_duration

print(f"\n⏱️  Thời gian xử lý: {elapsed_time:.3f}s")
print(f"🎵 Thời lượng audio: {audio_duration:.3f}s")
print(f"📊 RTF: {rtf:.4f}  ({'nhanh hơn' if rtf < 1 else 'chậm hơn'} real-time {1/rtf:.2f}x)" if rtf > 0 else "")

# List the built-in voices
voices = vieneu.list_preset_voices()
print(f"\n🎙️  {len(voices)} built-in voices available:")
for label, voice_id in voices:
    print(f"  - {label} ({voice_id})")

# 2. ⚡ Batch on GPU: infer_batch() runs many texts in ONE batched forward — same API.
#    On a CUDA GPU the chunks from every text share each forward step (big throughput
#    win). On CPU it still WORKS (no error) — just sequentially, so there's no batch
#    gain. Batch caps at max_batch_size (default 32; tune via Vieneu(max_batch_size=64)
#    or infer_batch(..., batch_size=64), or batch_size=1 to disable). A single long
#    infer() also auto-batches its own chunks. For real-time use, infer_stream() is the
#    streaming twin (GPU: 16 concurrent streams — see "Streaming" below). Uncomment to
#    try (GPU recommended):
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

#### Streaming (real-time) 🔊

> v3 Turbo streams **frame by frame** on both backends. **GPU** (PyTorch): first audio in **~115 ms** and **16 concurrent streams** on one RTX 3060 (continuous batching — one CUDA graph serves every `infer_stream` call, each keeping RTF ≈ 0.5–0.6). **CPU** (ONNX): first audio in ~140 ms (int8) / ~300 ms (fp32), one stream (two with int8). Just iterate `infer_stream`:

```python
from vieneu import Vieneu
vieneu = Vieneu()                                  # GPU → PyTorch + stream scheduler; no GPU → ONNX/CPU
for chunk in vieneu.infer_stream("Xin chào các bạn!", voice="Mai Anh"):
    play(chunk)                                   # np.float32 @ 48 kHz — play/write as it arrives
```

Calling `infer_stream` from many threads at once is the intended way to serve many listeners on a GPU (`Vieneu(max_streams=16)` sets the ceiling).

An **OpenAI-compatible streaming API** (`POST /v1/audio/speech`, `pcm`/`wav`, chunked or SSE — works with the OpenAI SDK, Pipecat, LiveKit, …) is in [`apps/openai_speech.py`](apps/openai_speech.py):

```bash
# Pick ONE of these — all serve http://localhost:8000/v1/audio/speech
uv run python -m apps.openai_speech                                  # from the repo (auto-detects GPU/CPU)
docker compose -f docker/docker-compose.yml --profile api-gpu up     # or: Docker, GPU
docker compose -f docker/docker-compose.yml --profile api-cpu up     # or: Docker, CPU only
```

📊 **[docs/streaming.md](docs/streaming.md)** — every measurement on an RTX 3060 (TTFA / RTF / streams vs `max_streams`), estimates for smaller GPUs, and the CPU numbers. The older browser demo is still at [`apps/web_stream.py`](apps/web_stream.py).

#### Available Voices

The v3 Turbo engine includes **25 preset voices** covering **3 regions** (North, Central, South) with diverse genders and speaking characters. `list_preset_voices()` (and the Web UI / API voice lists) show them in this order:

- ⭐ **Editors' picks** — the 10 we recommend starting with, hand-selected for naturalness and stability: **Adam bựa, Trúc Ly, Thiện Minh, Mai Anh, Hải Đăng** *(default)*, **Thùy Dung, Thiền Tâm Đức, Ngọc Huyền, Quang Sơn, Ngọc Trân**
- **Northern (Bắc)**: Minh Đức, Phạm Tuyên, Xuân Vĩnh, Thanh Bình, Ngọc Linh, Đoan Trang, Quỳnh Anh, Quốc Tuấn (+ picks above)
- **Central (Trung)**: Quang Sơn, Ngọc Trân
- **Southern (Nam)**: Adam, Thái Sơn, Thục Đoan, Minh Triết, Mỹ Duyên, Đức Trí, Kim Thanh (+ Thùy Dung)

### Reading style — **deprecated** ⚠️

> [!WARNING]
> **`style` is deprecated on v3 Turbo and has no effect.** The reading style is already
> baked into the reference itself (the speaker embedding + reference codes of the preset
> voice or of your cloned clip), so every generation follows the reference and comes out
> in its natural reading style.
>
> The `style` argument is **still accepted** by `infer`, `infer_stream`, `infer_batch`
> and `add_voice` so existing code keeps running — whatever you pass (`"tin_tuc"`,
> `"doc_truyen"`, …) is simply ignored. New code should just omit it.

```python
# Old code — still runs, but `style` is ignored
audio = vieneu.infer("Bản tin sáng nay.", voice="Hải Đăng", style="tin_tuc")

# New code — pick the reading character through the voice / reference clip instead
audio = vieneu.infer("Bản tin sáng nay.", voice="Hải Đăng")
```

### Emotion cues (experimental)

Inline tags are supported anywhere in the text: `[cười]` (chuckle), `[thở dài]` (sigh), `[hắng giọng]` (clear throat).

```python
audio = vieneu.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Hải Đăng")
```

### Pause tags

Use `[silent 200ms]`, `[silent 2s]` or `[silent 0.5s]` in text on v3 Turbo and v3 Nano.
Each tag inserts that duration of silence in place of the automatic gap at that
boundary. Silence already present in the generated speech is kept. Adjacent tags
add together, up to 60 seconds of tagged silence per input text.

### Voice cloning

Clone any voice from a short reference clip. The clip is cleaned up automatically
(background noise removed, and trimmed to ≤ 8 seconds) before cloning — keep
`denoise=True` unless your clip is already clean.

```python
audio = vieneu.infer(
    "Đây là giọng được nhân bản tức thì.",
    ref_audio="my_voice.wav",   # a 3–8s reference clip
    denoise=True,               # default; set False if the clip is already clean
)
vieneu.save(audio, "cloned.wav")
```

### Save & reuse a cloned voice

Register a reference once with `add_voice`, then use it by name like a built-in voice.

```python
# Enroll a voice (denoises + extracts the speaker profile once)
vieneu.add_voice("Giọng của tôi", "my_voice.wav")

# Now reuse it anywhere, including the conversation mode
audio = vieneu.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Persist your voices so they load next session
vieneu.save_voices()                 # writes to the default voices file
# vieneu.remove_voice("Giọng của tôi")

# Add a voice you already cleaned yourself → skip denoising
vieneu.add_voice("Giọng sạch", "already_clean.wav", denoise=False)
```

### Clean up a clip on its own

Get the denoised audio without synthesizing anything (e.g. to inspect or store it):

```python
wav, sr = vieneu.denoise("noisy.wav", out_path="clean.wav")   # 44.1 kHz mono
```

> **Note:** `denoise`, `add_voice`, and voice cloning work on every backend — the
> torch-free CPU/ONNX install included (the whole cloning pipeline runs on
> onnxruntime + soxr + kaldi-native-fbank). **v3 Nano** below clones the same way (its cloning graphs are fetched on first use).

<a id="v3-nano"></a>
### v3 Nano (preview) — for edge devices / weak CPUs only 🪶

> [!WARNING]
> **v3 Turbo remains the default and the recommended model.** Use v3 Nano only when Turbo is
> too slow on your hardware (old laptops, mini PCs, ARM boards, CPUs without AVX-512/VNNI where
> the int8 Turbo build produces garbled audio). Nano is a 48M-parameter flow-matching model
> (ONNX, CPU, torch-free) and it **trades quality for speed**:
> - **Lower quality than v3 Turbo — most noticeably on English and code-switched (En-Vi) text.**
>   Vietnamese is close; English words come out with a Vietnamese accent and are less stable.
> - **24 kHz** output (Turbo: 48 kHz).
> - **11 preset voices + voice cloning** (`ref_audio`, `add_voice`, `encode_reference` work like Turbo; the three cloning graphs, ~110 MB, download on first use).
> - **No frame-level streaming** — `infer_stream` yields one finished chunk at a time.

On the same 12th-gen Core i5, Nano runs at **RTF 0.22** (16 steps) or **0.11** (8 steps) against Turbo's 0.62 fp32 / 0.37 int8 — about **3× faster than Turbo fp32**, with a 282 MB download and ~3 s load time. Full table in [§4 Benchmarks](#benchmarks).

```python
from vieneu import Vieneu

tts = Vieneu(mode="v3nano")                      # ONNX, CPU, torch-free
audio = tts.infer("Xin chào, mình là giọng đọc của VieNeu Nano.", voice="Minh Quân")
tts.save(audio, "nano.wav")                      # 24 kHz

tts.list_preset_voices()                         # Adam, Ái Hân, Mỹ Duyên, Đức Trí, Hữu Quân, Xuân Tiên, Mai Anh, Trúc Ly, Anh Khôi, Minh Quân, Mạnh Dũng
audio = tts.infer("Bản nhanh cho máy rất yếu.", voice="Ái Hân", steps=8, sway=-1)   # ~2× faster
```

Knobs: `steps` (Euler steps, 16 default; 8 ≈ 2× faster, slightly rougher — pair with `sway=-1`),
`cfg` (classifier-free guidance, 3.0 default; `cfg=0` halves compute but hurts intelligibility),
`speed`, `seed`, `threads`. Emotion cues `[cười]` `[thở dài]` `[hắng giọng]` work as on Turbo.

---

## 🐳 3. API Server & Docker <a name="docker-remote"></a>

### Streaming API — OpenAI-compatible (v3 Turbo, CPU or GPU)

`apps/openai_speech.py` serves `POST /v1/audio/speech` exactly like OpenAI's TTS endpoint (`pcm`/`wav`, chunked body or SSE), so the **OpenAI SDK, Pipecat, LiveKit Agents, Vercel AI SDK, …** work by changing `base_url`. Audio streams as it is generated: first chunk in **~115 ms** with **16 concurrent streams** on an RTX 3060 (continuous batching), ~140–300 ms and 1–2 streams on a CPU.

```bash
# Start the server — pick ONE of these three (all listen on http://localhost:8000):
uv run python -m apps.openai_speech                                  # from the repo (auto-detects GPU/CPU)
docker compose -f docker/docker-compose.yml --profile api-gpu up     # or: Docker, GPU container
docker compose -f docker/docker-compose.yml --profile api-cpu up     # or: Docker, CPU container (torch-free)

# Then, in another terminal: measure TTFA / RTF on your machine
uv run python examples/openai_speech_client.py --bench 8
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
with client.audio.speech.with_streaming_response.create(
    model="vieneu-v3-turbo", voice="Mai Anh", input="Xin chào! Đây là chế độ streaming của VieNeu, phát tới đâu nghe tới đó.", response_format="pcm",
) as r:
    for chunk in r.iter_bytes(4096):   # s16le 48 kHz mono, as it is generated
        play(chunk)
```

Endpoints: `POST /v1/audio/speech`, `GET /v1/models`, `GET /v1/voices`, `POST /v1/voices` (clone from an uploaded clip), `GET /health`. Concurrency is capped per backend (`VIENEU_MAX_STREAMS`, default 16 on GPU / 1 on CPU) with a small queue and `429` beyond it.

📊 **[docs/streaming.md](docs/streaming.md)** — every measurement on an RTX 3060 (TTFA / RTF / streams vs `max_streams`), estimates for smaller GPUs, the CPU numbers, and tuning notes (e.g. the first request after the GPU idles pays +100–300 ms until it clocks up).

### Web UI in Docker

```bash
docker compose -f docker/docker-compose.yml --profile gpu up   # or --profile cpu → http://localhost:7860
```

See [docs/Deploy.md](docs/Deploy.md) for production builds and images.

### Legacy v2 API server (LMDeploy) — deprecated

> [!WARNING]
> **Deprecated.** This LMDeploy server and the `remote` mode only work with **VieNeu-TTS v2**, which is no longer updated. They are kept for existing deployments. For v3 Turbo use the streaming API above.

<details>
<summary><b>Legacy v2 server instructions (Docker + remote mode)</b></summary>

Deploy VieNeu-TTS as a high-performance API Server (powered by LMDeploy) with a single command.

### 1. Run with Docker

**Requirement**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is required for GPU support.

**Start the Server with a Public Tunnel (No port forwarding needed):**
```bash
docker run --gpus all -p 23333:23333 -v huggingface_cache:/root/.cache/huggingface pnnbao/vieneu-tts:latest --tunnel
```

*   **Default**: The server loads the `VieNeu-TTS-v2` model for maximum quality.
*   **Tunneling**: The Docker image includes a built-in `bore` tunnel. Check the container logs to find your public address (e.g., `bore.pub:31631`).

### 2. Using the SDK (Remote Mode)

Once the server is running, you can connect from anywhere (Colab, Web Apps, etc.) without loading heavy models locally.

**Installation**:
```bash
pip install "vieneu[legacy]"
```

**Usage**:
```python
from vieneu import Vieneu
import os

# Configuration
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Or bore tunnel URL
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-v2"

# Initialization (LIGHTWEIGHT - only loads small codec locally)
# Default emotion is "natural" (conversational) - set emotion="storytelling" for storytelling mode
vieneu = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID, emotion="natural")
os.makedirs("outputs", exist_ok=True)

# List remote voices
available_voices = vieneu.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Use specific voice (dynamically select second voice)
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = vieneu.get_preset_voice(my_voice_id)
    audio_spec = vieneu.infer(text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.", voice=voice_data)
    vieneu.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Saved synthesis to: outputs/remote_{my_voice_id}.wav")

# Standard synthesis (uses default voice)
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
audio = vieneu.infer(text=text_input)
vieneu.save(audio, "outputs/remote_output.wav")
print("💾 Saved remote synthesis to: outputs/remote_output.wav")

# Zero-shot voice cloning (encodes audio locally, sends codes to server)
if os.path.exists("examples/audio_ref/example_ngoc_huyen.wav"):
    cloned_audio = vieneu.infer(
        text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
        ref_audio="examples/audio_ref/example_ngoc_huyen.wav",
        ref_text="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
    )
    vieneu.save(cloned_audio, "outputs/remote_cloned_output.wav")
    print("💾 Saved remote cloned voice to: outputs/remote_cloned_output.wav")
```
*For full implementation details, see: [examples/main_remote.py](examples/main_remote.py)*

### Voice Preset Specification (v1.0)
VieNeu-TTS uses the official `vieneu.voice.presets` specification to define reusable voice assets. Only `voices.json` files following this spec are guaranteed to be compatible with VieNeu-TTS SDK ≥ v1.x.

### 3. Advanced Configuration

Customize the server to run specific versions or your own fine-tuned models.

**Run the 0.3B Model (Faster):**
```bash
docker run --gpus all pnnbao/vieneu-tts:serve --model pnnbao-ump/VieNeu-TTS-0.3B --tunnel
```

**Fine-tuned v3 Turbo models** are not served by this container (it hosts the v1/v2 LMDeploy backends). Load them with the SDK instead — see [Fine-tuning (LoRA)](#finetune):

```python
tts = Vieneu(mode="v3turbo", backbone_repo="finetune/output/my_voice/merged")
```

</details>

---

## 📊 4. Benchmarks <a name="benchmarks"></a>

Every speed and latency number quoted in this README, measured on one machine so they are comparable:
**RTX 3060 12 GB** · **Intel Core i5 12th gen (6 P-cores, 12 threads)** · Windows 11 · torch 2.8 + cu128 (bf16) · ONNX Runtime 1.24 (fp32/int8, 6 threads) · `vieneu` 3.8.x · September 2026. Reproduce with the snippets in [§2](#sdk) and `examples/openai_speech_client.py --bench N`.

**RTF** = generation time ÷ audio duration — lower is faster, **< 1 is faster than real-time** (0.02 = 50× real-time). **TTFA** = time to the first audio chunk when streaming.

### Throughput — how fast a whole text is synthesized

| Engine / mode | RTF ↓ | Example | Notes |
|---|---|---|---|
| **GPU, batched** (`infer_batch`, or one long `infer`) | **0.011–0.02** | 30 sentences / 130 s of audio in **1.5–2.3 s**; 154 s in 2.8 s | first call per batch size +~0.5 s (CUDA-graph capture) |
| **GPU, one sentence** (`infer`) | 0.10 | 3.5 s sentence in 0.36 s | launch-bound: a short sentence cannot fill the GPU |
| **CPU v3 Turbo fp32** (default on CPU) | 0.55–0.62 | 10 s sentence in ~6 s | 48 kHz, load ~19 s |
| **CPU v3 Turbo int8** (`precision="int8"`) | 0.35–0.37 | 10 s sentence in ~3.6 s | needs VNNI (AVX-512 VNNI / AVX-VNNI); load ~14 s |
| **CPU v3 Nano**, 16 steps (default) | 0.22 | — | 24 kHz, lower quality; load ~3 s |
| **CPU v3 Nano**, 8 steps, sway −1 | 0.11 | — | fastest, lowest quality |

The GPU is **~25–50× faster than the CPU** on bulk work; on a *single short sentence* the gap shrinks to ~5× because the work is too small to fill the GPU.

### Streaming — first-audio latency and concurrency (`infer_stream`, OpenAI API)

| Backend | Concurrent streams | TTFA (first audio) | RTF per stream |
|---|---|---|---|
| **GPU**, 1 stream | 1 | **~115 ms** (106 ms over HTTP) | 0.49 |
| **GPU**, 8 streams | 8 | 164 ms | 0.56 |
| **GPU**, 16 streams (default `max_streams`) | 16 | 185 ms median (max 339 when all 16 start at once); **134 ms** for a request arriving while 15 play | 0.59 |
| **GPU**, 32 streams (`max_streams=32`) | 32 | ~450 ms | 0.93 — still real-time, no margin |
| **CPU** fp32 | 1 | 260–400 ms | 0.55–0.61 (two at once → 1.19, both stall) |
| **CPU** int8 | 2 | 140–195 ms | 0.35 (two at once → 0.58–0.67) |

Streams ≠ users: a stream lives only while a reply plays, so 16 streams ≈ 45–80 active voice-chat users. Set `max_streams` to the real load — every reserved slot adds ~2.5 ms to each codec call. After a few idle seconds the GPU parks at low clocks and the *first* request pays +100–300 ms (lock clocks with `nvidia-smi -lgc` or "prefer maximum performance"). VRAM: 1.1 GB peak at 16 streams. Estimates for other GPUs and the full method: **[docs/streaming.md](docs/streaming.md)**.

---

## 🎓 5. Fine-tuning (LoRA) <a name="finetune"></a>

v3 Turbo already clones a voice from a clip of a few seconds. Fine-tune with **LoRA** when you need a tighter match than cloning, a specific reading style (storytelling, news, narration…), or better reading on your own domain. A LoRA teaches **one voice**: about **10–30 minutes** of clean audio from one speaker (one LoRA per voice if you need several). Only a few million parameters are trained, so a ~6 GB GPU is enough.

```bash
uv sync --extra finetune
uv run python finetune/prepare_dataset.py --dataset-dir finetune/dataset   # CPU, torch-free; all clips = one speaker
uv run python finetune/train_lora.py --data finetune/dataset/train.parquet --run my_voice --merge
uv run python finetune/make_voice.py --audio ref.wav --name "My voice" --out finetune/output/my_voice/merged
```

```python
tts = Vieneu(mode="v3turbo", backbone_repo="finetune/output/my_voice/merged")   # or your Hub repo
audio = tts.infer("Xin chào!", voice="My voice")        # packed voice — no reference audio needed
```

The merged model speaks that one voice from its packed speaker embedding (no reference audio) and keeps the v3 Turbo API (presets, batching, streaming) on the PyTorch/GPU backend; use the base model to clone other voices. Data layout, options and tips: [`finetune/README.md`](finetune/README.md).

---

## 🔬 6. Model Overview <a name="backbones"></a>

| Model | Status | Format | Device | Bilingual | Features | Speed ([§4](#benchmarks)) |
|---|---|---|---|---|---|---|
| **VieNeu-TTS-v3** | 🔜 **Coming soon** | PyTorch | **GPU** | ✅ | ? | ? |
| **VieNeu-TTS-v3-Turbo** *(default)* | 🦜 **Current** | PyTorch/ONNX | **GPU/CPU** | ✅ | **48 kHz, 25 preset voices, Cloning, Emotion cues, Conversation, Streaming (OpenAI-compatible API)** | **Ultra Fast** — GPU: RTF ≈ 0.02 batched, 16 real-time streams; CPU: RTF ≈ 0.5 (int8 0.35) |
| **VieNeu-TTS-v3-Nano** | 🧪 Preview | ONNX | **weak CPU / edge** | ⚠️ weak | 24 kHz, 11 preset voices, cloning, emotion cues — **lower quality (esp. English / En-Vi)** | Fastest on CPU (RTF 0.11–0.22) |
| VieNeu-TTS-v2 | ⛔ Deprecated | PyTorch | GPU | ✅ | Podcast, En-Vi CS | Fast (LMDeploy) |
| VieNeu-v2-CPU | ⛔ Deprecated | GGUF/ONNX | CPU/Edge | ✅ | Podcast, En-Vi CS | Medium |
| VieNeu-v2-Turbo | ⛔ Deprecated | GGUF/ONNX | CPU/Edge | ✅ | Lightweight En-Vi | Fast |
| VieNeu-TTS (v1) | ⛔ Deprecated | PyTorch | GPU/CPU | ❌ | Stable (Vi only) | Slow |

> ⛔ **Deprecated** models are no longer updated and are kept only for existing deployments; new projects should use **v3 Turbo** (and **v3** once released).

---

## 🚀 7. Roadmap <a name="roadmap"></a>

- [x] **VieNeu-TTS v3 Turbo** *(on-device, personal use)*: from-scratch 48 kHz architecture — preset voices, instant voice cloning, emotion cues, batched generation, multi-speaker conversation, frame-level streaming; torch-free on CPU.
- [x] **VieNeu-TTS v3 Nano** *(preview)*: 48M flow-matching model for weak CPUs / edge devices — 11 preset voices + cloning, torch-free.
- [x] **LoRA fine-tuning** for v3 Turbo — train your own voice or reading style on one consumer GPU.
- [x] **VieNeu-TTS v3 Turbo GPU streaming server**: OpenAI-compatible `/v1/audio/speech` with continuous batching on one CUDA graph — first audio ~115 ms, 16 concurrent streams on an RTX 3060, Docker profiles `api-gpu` / `api-cpu` ([docs/streaming.md](docs/streaming.md)).
- [ ] **VieNeu-TTS v3 (GPU, server release)**: the full v3 model for API / server deployment — finalized quality, stable emotion control, more voices.
- [ ] **Mobile SDK**: official Android / iOS deployment.

---

## 🤝 8. Support & Contact <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/pnnbao97)
- **License:** Apache 2.0 (Free to use).

---
## 📑 9. Citation <a name="citation"></a>

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS: Advanced Vietnamese Text-to-Speech with Instant Voice Cloning},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

---

## 🌟 Star History

<a href="https://github.com/pnnbao97/VieNeu-TTS/stargazers">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pnnbao97/star-charts/main/charts/pnnbao97/VieNeu-TTS/dark.svg" />
   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pnnbao97/star-charts/main/charts/pnnbao97/VieNeu-TTS/light.svg" />
   <img alt="Star History Chart" src="https://raw.githubusercontent.com/pnnbao97/star-charts/main/charts/pnnbao97/VieNeu-TTS/light.svg" />
 </picture>
</a>

---

## 🤝 Contributors

Thanks to all the amazing people who have contributed to this project!

<a href="https://github.com/pnnbao97/VieNeu-TTS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pnnbao97/VieNeu-TTS" />
</a>

---

## 🙏 Acknowledgements

This project uses [neucodec](https://huggingface.co/neuphonic/neucodec) (v1/v2) and [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) (v3 Turbo) for audio coding, and [sea-g2p](https://github.com/pnnbao97/sea-g2p) for text normalization and phonemization.

**Made with ❤️ for the Vietnamese TTS community**
