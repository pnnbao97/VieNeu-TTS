# 🦜 VieNeu-TTS
# các lệnh chạy dự án
 - phiên bản web: uv run vieneu-web
 - phiên bản stream: uv run vieneu-stream

 - lệnh chạy server mặc định địa chỉ 8000: uv run python -m apps.openai_speech

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face VieNeu-TTS-v3-Turbo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v3--Turbo-red)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
[![Hugging Face VieNeu-TTS-v2](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS** là thế hệ tiếp theo của mô hình chuyển văn bản thành giọng nói (TTS) tiếng Việt chạy trên thiết bị: **10.000+ giờ dữ liệu** huấn luyện song ngữ, **clone giọng tức thì**, chế độ **Podcast/Hội thoại** chuyên dụng, và **cực nhanh** — streaming thời gian thực với chunk đầu ~115 ms, **RTF ≈ 0,01–0,02** khi batch trên GPU phổ thông (RTX 3060), nhanh ~2× thời gian thực trên CPU thường ([benchmark](#benchmarks)).

> [!IMPORTANT]
> **🦜 VieNeu-TTS v4 — đã có trên [vieneu.io](https://www.vieneu.io)**
>
> VieNeu-TTS v4 clone giọng với độ trung thực **gần như bản gốc**: chỉ cần một clip ngắn là tái tạo được giọng với độ giống rất cao.
>
> Vì khả năng clone quá mạnh và nguy cơ bị lạm dụng, **v4 là bản độc quyền, không mã nguồn mở**. v4 chỉ có qua **VieNeu API / vieneu.io**.
>
> **VieNeu-TTS v3 Turbo vẫn là bản mã nguồn mở mới nhất trong repo này.** Các bản mở tiếp theo, kể cả v3.x, cũng sẽ được phát hành ở đây.

> [!NOTE]
> **🦜 VieNeu-TTS v3 Turbo đã chính thức ra mắt!**
> Kiến trúc hoàn toàn mới, **do Phạm Nguyễn Ngọc Bảo thiết kế và huấn luyện từ đầu** (codec: [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano); phiên âm: [sea-g2p](https://github.com/pnnbao97/sea-g2p)):
> - Âm thanh **48 kHz** chất lượng cao (trước đây 24 kHz).
> - **25 giọng dựng sẵn** (10 giọng tuyển chọn) — ổn định, nhất quán, không cần clip mẫu.
> - **Phong cách đọc tự nhiên** ở mọi nơi — phong cách đi theo giọng mẫu (tham số `style` đã bỏ, truyền vào cũng bị bỏ qua).
> - **Tag cảm xúc / phi ngôn từ** *(thử nghiệm)*: chèn `[cười]`, `[thở dài]`, `[hắng giọng]` thẳng vào văn bản.
> - **Sinh theo lô** (batch tới 32), gồm chế độ **Hội thoại** nhiều người nói batch cả kịch bản bất kể người nói.
> - **Clone giọng tức thì** từ clip 3–8 giây, tự khử nhiễu clip mẫu.
> - **Streaming thời gian thực + API chuẩn OpenAI + Docker** — `POST /v1/audio/speech` thay thẳng cho OpenAI SDK / Pipecat / LiveKit; chunk đầu **~115 ms**, **16 luồng đồng thời dưới 200 ms** trên một RTX 3060 (tối đa 32), máy chỉ có CPU cũng stream được. Xem [§3](#docker-remote), [§4 Benchmark](#benchmarks) và [docs/streaming.vi.md](docs/streaming.vi.md).
> - **Fine-tune LoRA** — train giọng hoặc phong cách đọc riêng trên một GPU phổ thông ([§5](#finetune)).
>
> Dùng thử trong Web UI (backbone **"VieNeu-TTS-v3-Turbo"**) hoặc SDK (`Vieneu(mode="v3turbo")`, là mặc định).

<h3>🎬 Demos</h3>

> [!TIP]
> Các tính năng **Lồng tiếng (Dubbing)**, **Bài giảng (Lecture)** và **Sách nói (Audiobook)** chỉ có trong [ứng dụng VieNeu chính thức](https://www.vieneu.io/#/download). Repo GitHub này chỉ cung cấp giao diện demo Gradio đơn giản và SDK lõi cho lập trình viên.

<table>
  <!-- Hàng 1 -->
  <tr>
    <td align="center" width="50%">
      <b>Voice Cloning</b><br><br>
      <video
        src="https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd"
        controls
        width="100%">
      </video>
    </td>
    <td align="center" width="50%">
      <b>Dubbing</b><br><br>
      <video
        src="https://github.com/user-attachments/assets/5888aea1-4f32-4397-9dd9-9c7b743d31bd"
        controls
        width="100%">
      </video>
    </td>
  </tr>
  <!-- Hàng 2 -->
  <tr>
    <td align="center" width="50%">
      <b>Dubbing / Conversation</b><br><br>
      <video
        src="https://github.com/user-attachments/assets/28104b78-2d55-4914-85b7-5f425a7e99da"
        controls
        width="100%">
      </video>
    </td>
    <td align="center" width="50%">
      <b>Lecture</b><br><br>
      <video
        src="https://github.com/user-attachments/assets/de3b2d4e-4c50-4164-acdc-e3de90840e26"
        controls
        width="100%">
      </video>
    </td>
  </tr>
</table>

## 📌 Mục lục

1. [🦜 Cài đặt & Giao diện Web](#installation)
2. [📦 Sử dụng Python SDK](#sdk)
3. [🐳 API Server & Docker](#docker-remote) — API streaming chuẩn OpenAI (v3 Turbo) · server v2 cũ
4. [📊 Benchmark](#benchmarks) — mọi số đo tốc độ / độ trễ ở một chỗ (CPU vs GPU, batch, streaming, Nano)
5. [🎓 Fine-tune (LoRA)](#finetune)
6. [🔬 Tổng quan mô hình](#backbones)
7. [🚀 Lộ trình phát triển](#roadmap)
8. [🤝 Hỗ trợ & Liên hệ](#support)
9. [📑 Trích dẫn](#citation)

---

## 🦜 1. Cài đặt & Giao diện Web <a name="installation"></a>
> [!TIP]
> **Dùng Windows?** Cách nhanh nhất là bộ cài độc lập tại **[vieneu.io/#/download](https://www.vieneu.io/#/download)** — không cần cài `uv` hay clone repo.
> **macOS**: bộ cài tương tự sẽ có trong bản sắp tới; hiện tại dùng các bước `uv sync` bên dưới.
> **Dùng Docker?** Bỏ qua các bước dưới: `--profile api-gpu` / `api-cpu` (API streaming chuẩn OpenAI) — xem [§3 API Server & Docker](#docker-remote).

### Thiết lập với `uv` (Khuyến nghị)
`uv` là cách nhanh nhất để quản lý các phụ thuộc.
```bash
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Cài đặt các phụ thuộc:**
   > 📊 **Chọn cái nào?** CPU ≈ **RTF 0,5** (nhanh 2× thời gian thực, một luồng) · GPU ≈ **RTF 0,02** khi batch (~50× thời gian thực) và **16 luồng streaming real-time** — toàn bộ số đo ở [§4 Benchmark](#benchmarks).

   - **Lựa chọn 1: CPU & macOS (tối giản, không cần torch)** — RTF ≈ 0,5, không cần GPU

     ```bash
     uv sync
     ```
   - **Lựa chọn 2: GPU** — **v3 Turbo chạy trên GPU (PyTorch)** — RTF ≈ 0,02 khi batch, 16 luồng streaming real-time

     ```bash
     uv sync --extra cuda
     ```

3. **Khởi chạy Giao diện Web:**
   ```bash
   uv run vieneu-web
   ```
   Truy cập giao diện tại `http://127.0.0.1:7860`.


---


## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

SDK `vieneu` **mặc định dùng VieNeu-TTS v3 Turbo (48 kHz)**. Bản cài tối giản **không cần torch**: trên CPU mọi thứ chạy bằng **ONNX Runtime** (PyTorch không bao giờ được import), còn trên máy CUDA nó tự chuyển sang engine PyTorch — nơi suy luận được **batch tự động** (cùng API, không đổi code).

> ⚡ **Trên CPU, backbone chạy `fp32` theo mặc định** (chất lượng tối đa). Cần nhanh hơn? Truyền `Vieneu(precision="int8")` — nhanh ~1.6× và nhẹ ~4×, nhưng cần CPU hỗ trợ VNNI (AVX-512 VNNI / AVX-VNNI); trên CPU đời cũ int8 có thể cho audio méo/vô nghĩa. `precision` chỉ ảnh hưởng đường CPU/ONNX; trên GPU nó bị bỏ qua (PyTorch).
>
> 🪶 **Vẫn quá chậm, hoặc cần deploy trên điện thoại / board ARM?** Dùng **[VieNeu-TTS v3 Nano (preview)](#v3-nano)** — `Vieneu(mode="v3nano")`, nhanh hơn Turbo fp32 ~3× trên CPU (RTF 0.11–0.22 trên CPU desktop), nhưng **chất lượng kém hơn rõ rệt** (nhất là tiếng Anh / song ngữ), 24 kHz, 11 giọng có sẵn + clone giọng. Xem chi tiết và các hạn chế ở [mục v3 Nano](#v3-nano) bên dưới.
>
> ```python
> vieneu = Vieneu()                    # backbone fp32 (mặc định, chất lượng tối đa)
> vieneu = Vieneu(precision="int8")    # backbone int8 (nhanh hơn trên CPU có VNNI)
> ```

### Bắt đầu nhanh
**CPU (mặc định)** — không cần torch, chạy v3 Turbo bằng ONNX Runtime. Đa số người dùng chọn cái này — **RTF ≈ 0,5** trên Core i5 thế hệ 12 (số đo ở [§4 Benchmark](#benchmarks)). Cần nhanh hơn thì `Vieneu(precision="int8")` — nhanh ~1,6× (RTF ≈ 0,35), cần CPU có VNNI:

```bash
pip install vieneu
```

**GPU (CUDA)** — chỉ khi bạn có GPU NVIDIA; **nhanh hơn CPU ~25 lần** (RTF ≈ 0,02 khi batch, 16 luồng streaming real-time — [§4 Benchmark](#benchmarks)). Trên Linux `pip install "vieneu[cuda]"` là đủ (torch trên PyPI đã kèm CUDA); trên Windows cài torch CUDA **trước** như dưới. Trên CUDA, batch tự bật — cùng API, không đổi code:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6"   # Qwen3 backbone + MOSS codec (bản ổn định nhất cho SDK GPU)
pip install vieneu
```

> ⚡ Trên GPU mỗi khung âm thanh là **một CUDA graph** (acoustic + sampling + phạt
> lặp + backbone gộp một lần phát, không cần `torch.compile` hay trình biên dịch
> C++). Lần gọi đầu cho mỗi cỡ batch tốn thêm ~0,5 s để capture graph (giữ lại cho
> các lần sau; server gọi `warm_fused()` lúc khởi động). `VIENEU_FUSED_FRAME=0`
> quay về vòng lặp thường.

```python
from vieneu import Vieneu

# Mặc định = v3 Turbo (48 kHz). GPU → PyTorch (tự nhận diện).
vieneu = Vieneu()
# 💡 Trên máy GPU vẫn có thể chuyển sang ONNX/CPU nếu muốn: Vieneu(backend="onnx")

# 1. Giọng dựng sẵn theo tên — không cần audio mẫu
print("🔊 Đang sinh giọng nói...")
audio = vieneu.infer("Xin chào, đây là VieNeu-TTS.", voice="Hải Đăng")
vieneu.save(audio, "output.wav")
print("✅ Đã lưu vào output.wav")

# Liệt kê các giọng dựng sẵn
voices = vieneu.list_preset_voices()
print(f"\n🎙️  Có {len(voices)} giọng dựng sẵn:")
for label, voice_id in voices:
    print(f"  - {label} ({voice_id})")

# 2. ⚡ Batch trên GPU: infer_batch() chạy nhiều text trong MỘT lần forward — cùng API.
#    Trên GPU CUDA, các chunk của mọi text dùng chung mỗi bước forward (throughput cao
#    hơn nhiều); trên CPU vẫn CHẠY ĐƯỢC (không lỗi), chỉ là tuần tự. Batch tối đa
#    max_batch_size (mặc định 32; hoặc infer_batch(..., batch_size=64); batch_size=1 để
#    tắt). Một infer() cho text dài cũng tự batch các chunk. Cần realtime thì dùng
#    infer_stream() — bản streaming cùng API (GPU: 16 luồng đồng thời, xem mục
#    "Streaming" bên dưới). Bỏ comment để thử (nên dùng GPU):
#
# import time
# texts = [
#     "Chào cả nhà, hôm nay mình sẽ hướng dẫn các bạn cách cài đặt và sử dụng bộ giọng đọc mới.",
#     "Giọng nghe cực kỳ tự nhiên và truyền cảm, lại có thể chuyển đổi biểu cảm một cách linh hoạt.",
#     "Nếu thấy hữu ích, các bạn nhớ để lại một lượt thích và chia sẻ video này cho mọi người nhé!",
# ] * 10   # 30 câu — đủ lấp đầy batch để thấy rõ sức mạnh throughput của GPU
# t0 = time.time()
# audios = vieneu.infer_batch(texts, voice="Hải Đăng")
# elapsed = time.time() - t0
# total_audio = sum(len(a) for a in audios) / 48_000
# print(f"⚡ {len(texts)} câu | audio {total_audio:.1f}s | thời gian {elapsed:.1f}s | RTF {elapsed/total_audio:.3f}")
# for i, a in enumerate(audios):
#     vieneu.save(a, f"batch_{i}.wav")
```

### Streaming thời gian thực 🔊

v3 Turbo stream **theo từng frame** trên cả hai backend. **GPU** (PyTorch): audio đầu sau **~115 ms**, **16 luồng đồng thời** trên một RTX 3060 (continuous batching — một CUDA graph phục vụ mọi lời gọi `infer_stream`, mỗi luồng giữ RTF ≈ 0,5–0,6). **CPU** (ONNX): audio đầu ~140 ms (int8) / ~300 ms (fp32), một luồng (int8: hai). Chỉ cần lặp `infer_stream`:

```python
from vieneu import Vieneu
vieneu = Vieneu()                                  # có GPU → PyTorch + scheduler stream; không → ONNX/CPU
for chunk in vieneu.infer_stream("Xin chào các bạn!", voice="Mai Anh"):
    play(chunk)                                   # np.float32 @ 48 kHz — phát/ghi ngay khi có
```

Gọi `infer_stream` từ nhiều thread cùng lúc chính là cách phục vụ nhiều người nghe trên GPU (`Vieneu(max_streams=16)` đặt trần).

**API streaming chuẩn OpenAI** (`POST /v1/audio/speech`, `pcm`/`wav`, chunked hoặc SSE — dùng được với OpenAI SDK, Pipecat, LiveKit, …) nằm ở [`apps/openai_speech.py`](apps/openai_speech.py):

```bash
# Chọn MỘT trong ba cách — đều phục vụ http://localhost:8000/v1/audio/speech
uv run python -m apps.openai_speech                                  # chạy từ repo (tự nhận GPU/CPU)
docker compose -f docker/docker-compose.yml --profile api-gpu up     # hoặc: Docker, GPU
docker compose -f docker/docker-compose.yml --profile api-cpu up     # hoặc: Docker, chỉ CPU
```

📊 **[docs/streaming.vi.md](docs/streaming.vi.md)** — toàn bộ số đo trên RTX 3060 (TTFA / RTF / số luồng theo `max_streams`), dự đoán cho GPU nhỏ hơn, và số đo CPU. Bản demo trình duyệt cũ vẫn ở [`apps/web_stream.py`](apps/web_stream.py).

#### Các giọng dựng sẵn

v3 Turbo có **25 giọng dựng sẵn** phủ **3 miền** (Bắc, Trung, Nam), đủ giới tính và phong cách đọc. `list_preset_voices()` (cũng như danh sách giọng trên Web UI / API) hiển thị theo đúng thứ tự này:

- ⭐ **Giọng tuyển chọn** — 10 giọng chúng tôi khuyên dùng trước, chọn tay theo độ tự nhiên và ổn định: **Adam bựa, Trúc Ly, Thiện Minh, Mai Anh, Hải Đăng** *(mặc định)*, **Thùy Dung, Thiền Tâm Đức, Ngọc Huyền, Quang Sơn, Ngọc Trân**
- **Miền Bắc**: Minh Đức, Phạm Tuyên, Xuân Vĩnh, Thanh Bình, Ngọc Linh, Đoan Trang, Quỳnh Anh, Quốc Tuấn (+ các giọng tuyển ở trên)
- **Miền Trung**: Quang Sơn, Ngọc Trân
- **Miền Nam**: Adam, Thái Sơn, Thục Đoan, Minh Triết, Mỹ Duyên, Đức Trí, Kim Thanh (+ Thùy Dung)

### Phong cách đọc — **đã bỏ (deprecated)** ⚠️

> [!WARNING]
> **`style` không còn tác dụng trên v3 Turbo.** Phong cách đọc đã được *ám sẵn trong
> reference* (speaker embedding + ref codes của giọng dựng sẵn hoặc của clip bạn clone),
> nên mô hình luôn bám theo reference và đọc ở phong cách **tự nhiên**.
>
> Tham số `style` **vẫn được chấp nhận** ở `infer`, `infer_stream`, `infer_batch` và
> `add_voice` để code cũ không vỡ — truyền gì (`"tin_tuc"`, `"doc_truyen"`, …) cũng bị
> bỏ qua. Code mới nên bỏ hẳn tham số này.

```python
# Code cũ — vẫn chạy, nhưng `style` bị bỏ qua
audio = vieneu.infer("Bản tin sáng nay.", voice="Hải Đăng", style="tin_tuc")

# Code mới — chọn chất giọng/cách đọc bằng chính giọng mẫu hoặc clip reference
audio = vieneu.infer("Bản tin sáng nay.", voice="Hải Đăng")
```

### Tag cảm xúc (thử nghiệm)

Chèn trực tiếp trong văn bản: `[cười]`, `[thở dài]`, `[hắng giọng]`.

```python
audio = vieneu.infer("Nghe hay quá đi [cười]. Để mình nói tiếp [hắng giọng].", voice="Hải Đăng")
```

> [!TIP]
> Temperature ~0.8 ổn định nhất.

### 🦜 Clone giọng nói Zero-shot (SDK) <a name="cloning"></a>
Clone bất kỳ giọng nào từ một clip ngắn. Clip mẫu được **tự khử nhiễu nền** và **cắt còn ≤ 8 giây** trước khi clone — cứ để `denoise=True` trừ khi clip đã sạch.

```python
from vieneu import Vieneu

vieneu = Vieneu()

# Clone trực tiếp từ clip mẫu (3–8 giây)
audio = vieneu.infer(
    text="Đây là giọng được nhân bản tức thì.",
    ref_audio="examples/audio_ref/example.wav",
    denoise=True,          # mặc định; đặt False nếu clip đã sạch
)
vieneu.save(audio, "cloned_voice.wav")
```

#### Lưu & tái dùng giọng đã clone
Đăng ký clip một lần bằng `add_voice`, sau đó gọi theo tên như giọng dựng sẵn (dùng được cả ở chế độ Hội thoại).

```python
# Đăng ký giọng (tự denoise + trích hồ sơ giọng một lần)
vieneu.add_voice("Giọng của tôi", "my_voice.wav")

audio = vieneu.infer("Câu này dùng giọng đã lưu.", voice="Giọng của tôi")

# Lưu lại để lần sau vẫn còn
vieneu.save_voices()
# vieneu.remove_voice("Giọng của tôi")

# Thêm giọng bạn đã tự làm sạch → bỏ qua bước denoise
vieneu.add_voice("Giọng sạch", "already_clean.wav", denoise=False)
```

#### Chỉ khử nhiễu một clip
Lấy audio đã khử nhiễu mà không tổng hợp gì (để nghe/lưu lại):

```python
wav, sr = vieneu.denoise("noisy.wav", out_path="clean.wav")   # 44.1 kHz mono
```

> **Lưu ý:** `denoise`, `add_voice` và voice cloning chạy trên mọi backend — kể cả bản cài CPU/ONNX không torch (toàn bộ pipeline cloning chạy bằng onnxruntime + soxr + kaldi-native-fbank). **v3 Nano** bên dưới clone theo đúng cách này (các đồ thị clone tải ở lần dùng đầu).

<a id="v3-nano"></a>
### v3 Nano (preview) — chỉ dành cho edge device / máy CPU yếu 🪶

> [!WARNING]
> **v3 Turbo vẫn là mặc định và là bản được khuyến nghị.** Chỉ dùng v3 Nano khi Turbo quá chậm
> trên máy của bạn (laptop cũ, mini PC, board ARM, CPU không có AVX-512/VNNI khiến bản Turbo int8
> bị méo tiếng). Nano là model flow-matching 48M tham số (ONNX, CPU, không cần torch) và
> **đánh đổi chất lượng lấy tốc độ**:
> - **Chất lượng thấp hơn v3 Turbo — rõ nhất ở tiếng Anh và câu song ngữ Anh-Việt.**
>   Tiếng Việt gần tương đương; từ tiếng Anh đọc mang giọng Việt và kém ổn định hơn.
> - Âm thanh **24 kHz** (Turbo: 48 kHz).
> - **11 giọng có sẵn + clone giọng** (`ref_audio`, `add_voice`, `encode_reference` dùng như Turbo; ba đồ thị clone ~110 MB tải ở lần dùng đầu).
> - **Không streaming theo frame** — `infer_stream` trả từng chunk đã hoàn chỉnh.

Trên cùng Core i5 thế hệ 12, Nano đạt **RTF 0,22** (16 bước) hoặc **0,11** (8 bước) so với Turbo 0,62 fp32 / 0,37 int8 — nhanh hơn Turbo fp32 khoảng **3 lần**, dung lượng tải 282 MB, nạp ~3 s. Bảng đầy đủ ở [§4 Benchmark](#benchmarks).

```python
from vieneu import Vieneu

tts = Vieneu(mode="v3nano")                      # ONNX, CPU, không cần torch
audio = tts.infer("Xin chào, mình là giọng đọc của VieNeu Nano.", voice="Minh Quân")
tts.save(audio, "nano.wav")                      # 24 kHz

tts.list_preset_voices()                         # Adam, Ái Hân, Mỹ Duyên, Đức Trí, Hữu Quân, Xuân Tiên, Mai Anh, Trúc Ly, Anh Khôi, Minh Quân, Mạnh Dũng
audio = tts.infer("Bản nhanh cho máy rất yếu.", voice="Ái Hân", steps=8, sway=-1)   # nhanh gấp ~2
```

Tham số: `steps` (số bước Euler, mặc định 16; 8 nhanh gấp ~2, hơi thô hơn — đi kèm `sway=-1`),
`cfg` (classifier-free guidance, mặc định 3.0; `cfg=0` giảm nửa tính toán nhưng kém rõ chữ),
`speed`, `seed`, `threads`. Tag cảm xúc `[cười]` `[thở dài]` `[hắng giọng]` dùng như Turbo.

---

## 🐳 3. API Server & Docker <a name="docker-remote"></a>

### API streaming — chuẩn OpenAI (v3 Turbo, CPU hoặc GPU)

`apps/openai_speech.py` phục vụ `POST /v1/audio/speech` giống hệt endpoint TTS của OpenAI (`pcm`/`wav`, body chunked hoặc SSE), nên **OpenAI SDK, Pipecat, LiveKit Agents, Vercel AI SDK, …** dùng được chỉ bằng đổi `base_url`. Audio phát ra ngay khi sinh: chunk đầu **~115 ms**, **16 luồng đồng thời** trên RTX 3060 (continuous batching); trên CPU ~140–300 ms và 1–2 luồng.

```bash
# Bật server — chọn MỘT trong ba cách (đều nghe ở http://localhost:8000):
uv run python -m apps.openai_speech                                  # chạy từ repo (tự nhận GPU/CPU)
docker compose -f docker/docker-compose.yml --profile api-gpu up     # hoặc: Docker, container GPU
docker compose -f docker/docker-compose.yml --profile api-cpu up     # hoặc: Docker, container CPU (không torch)

# Rồi ở terminal khác: tự đo TTFA / RTF trên máy bạn
uv run python examples/openai_speech_client.py --bench 8
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
with client.audio.speech.with_streaming_response.create(
    model="vieneu-v3-turbo", voice="Mai Anh", input="Xin chào! Đây là chế độ streaming của VieNeu, phát tới đâu nghe tới đó.", response_format="pcm",
) as r:
    for chunk in r.iter_bytes(4096):   # s16le 48 kHz mono, tới đâu phát tới đó
        play(chunk)
```

Endpoint: `POST /v1/audio/speech`, `GET /v1/models`, `GET /v1/voices`, `POST /v1/voices` (nhân bản giọng từ clip upload), `GET /health`. Số luồng giới hạn theo backend (`VIENEU_MAX_STREAMS`, mặc định 16 trên GPU / 1 trên CPU), có hàng đợi nhỏ, quá thì `429`.

📊 **[docs/streaming.vi.md](docs/streaming.vi.md)** — toàn bộ số đo trên RTX 3060 (TTFA / RTF / số luồng theo `max_streams`), dự đoán cho GPU nhỏ hơn, số đo CPU, và các lưu ý tinh chỉnh (ví dụ request đầu tiên sau khi GPU rỗi chậm thêm 100–300 ms vì GPU hạ xung).

### Web UI trong Docker

```bash
docker compose -f docker/docker-compose.yml --profile gpu up   # hoặc --profile cpu → http://localhost:7860
```

Xem [docs/Deploy.vi.md](docs/Deploy.vi.md) cho build production và image.

### Server API v2 cũ (LMDeploy) — đã ngừng

> [!WARNING]
> **Đã ngừng cập nhật.** Server LMDeploy và chế độ `remote` này chỉ chạy với **VieNeu-TTS v2**, bản v2 không còn được cập nhật. Phần này giữ lại cho các hệ thống đang chạy. Với v3 Turbo hãy dùng API streaming ở trên.

<details>
<summary><b>Hướng dẫn server v2 cũ (Docker + chế độ remote)</b></summary>

Triển khai VieNeu-TTS dưới dạng API Server hiệu suất cao (được hỗ trợ bởi LMDeploy) chỉ bằng một câu lệnh duy nhất.

### 1. Chạy với Docker

**Yêu cầu**: Cần cài đặt [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) để hỗ trợ GPU.

**Khởi chạy Server với Đường hầm công khai (Không cần mở cổng modem):**
```bash
docker run --gpus all -p 23333:23333 -v huggingface_cache:/root/.cache/huggingface pnnbao/vieneu-tts:latest --tunnel
```

*   **Mặc định**: Server sẽ tải model `VieNeu-TTS-v2` để đạt chất lượng tối đa.
*   **Tunneling**: Docker image tích hợp sẵn đường hầm `bore`. Kiểm tra container logs để tìm địa chỉ công khai của bạn (VD: `bore.pub:31631`).

### 2. Sử dụng SDK (Chế độ Remote)

Khi server đã chạy, bạn có thể kết nối từ bất kỳ đâu (Colab, Web App, v.v.) mà không cần tải các model nặng cục bộ.

**Cài đặt**:
```bash
pip install "vieneu[legacy]"
```

**Sử dụng**:
```python
from vieneu import Vieneu
import os

# Cấu hình
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Hoặc URL từ bore tunnel
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS-v2"

# Khởi tạo (Cực kỳ NHẸ - chỉ tải codec nhỏ cục bộ)
# Cảm xúc mặc định là "natural" (tự nhiên) - đặt emotion="storytelling" cho chế độ kể chuyện
vieneu = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID, emotion="natural")
os.makedirs("outputs", exist_ok=True)

# Liệt kê các giọng mẫu trên server
available_voices = vieneu.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Sử dụng giọng cụ thể (chọn động giọng thứ hai)
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = vieneu.get_preset_voice(my_voice_id)
    audio_spec = vieneu.infer(text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.", voice=voice_data)
    vieneu.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Đã lưu kết quả tại: outputs/remote_{my_voice_id}.wav")

# Tổng hợp chuẩn (dùng giọng mặc định)
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
audio = vieneu.infer(text=text_input)
vieneu.save(audio, "outputs/remote_output.wav")
print("💾 Đã lưu kết quả remote_output.wav")

# Clone giọng Zero-shot (Mã hóa âm thanh cục bộ, gửi code lên server)
if os.path.exists("examples/audio_ref/example_ngoc_huyen.wav"):
    cloned_audio = vieneu.infer(
        text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
        ref_audio="examples/audio_ref/example_ngoc_huyen.wav",
        ref_text="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
    )
    vieneu.save(cloned_audio, "outputs/remote_cloned_output.wav")
    print("💾 Đã lưu kết quả remote_cloned_output.wav")
```
*Chi tiết xem tại: [examples/main_remote.py](examples/main_remote.py)*

### Quy chuẩn Voice Preset (v1.0)
VieNeu-TTS sử dụng quy chuẩn chính thức `vieneu.voice.presets` để định nghĩa các tài nguyên giọng nói có thể tái sử dụng. Chỉ các tệp `voices.json` tuân theo quy chuẩn này mới đảm bảo tương thích với VieNeu-TTS SDK ≥ v1.x.

### 3. Cấu hình Nâng cao

Tùy chỉnh server để chạy các phiên bản cụ thể hoặc các model đã được fine-tune của riêng bạn.

**Chạy model 0.3B (Nhanh hơn):**
```bash
docker run --gpus all pnnbao/vieneu-tts:serve --model pnnbao-ump/VieNeu-TTS-0.3B --tunnel
```

**Model v3 Turbo đã fine-tune** không chạy qua container này (container chỉ phục vụ backend LMDeploy của v1/v2). Hãy nạp bằng SDK — xem [Fine-tune (LoRA)](#finetune):

```python
tts = Vieneu(mode="v3turbo", backbone_repo="finetune/output/my_voice/merged")
```

</details>

---

## 📊 4. Benchmark <a name="benchmarks"></a>

Mọi con số tốc độ / độ trễ trong README này, đo trên **cùng một máy** để so sánh được với nhau:
**RTX 3060 12 GB** · **Intel Core i5 thế hệ 12 (6 P-core, 12 luồng)** · Windows 11 · torch 2.8 + cu128 (bf16) · ONNX Runtime 1.24 (fp32/int8, 6 luồng) · `vieneu` 3.8.x · tháng 9/2026. Tự đo lại bằng các đoạn mã ở [§2](#sdk) và `examples/openai_speech_client.py --bench N`.

**RTF** = thời gian sinh ÷ thời lượng audio — càng thấp càng nhanh, **< 1 là nhanh hơn thời gian thực** (0,02 = nhanh 50×). **TTFA** = thời gian tới chunk audio đầu tiên khi streaming.

### Thông lượng — sinh cả văn bản mất bao lâu

| Engine / chế độ | RTF ↓ | Ví dụ | Ghi chú |
|---|---|---|---|
| **GPU, batch** (`infer_batch`, hoặc một `infer` văn bản dài) | **0,011–0,02** | 30 câu / 130 s audio trong **1,5–2,3 s**; 154 s trong 2,8 s | lần gọi đầu mỗi cỡ batch +~0,5 s (capture CUDA graph) |
| **GPU, một câu** (`infer`) | 0,10 | câu 3,5 s trong 0,36 s | bị launch-bound: câu ngắn không lấp đầy GPU |
| **CPU v3 Turbo fp32** (mặc định trên CPU) | 0,55–0,62 | câu 10 s trong ~6 s | 48 kHz, nạp ~19 s |
| **CPU v3 Turbo int8** (`precision="int8"`) | 0,35–0,37 | câu 10 s trong ~3,6 s | cần VNNI (AVX-512 VNNI / AVX-VNNI); nạp ~14 s |
| **CPU v3 Nano**, 16 bước (mặc định) | 0,22 | — | 24 kHz, chất lượng thấp hơn; nạp ~3 s |
| **CPU v3 Nano**, 8 bước, sway −1 | 0,11 | — | nhanh nhất, chất lượng thấp nhất |

GPU **nhanh hơn CPU ~25–50 lần** khi sinh hàng loạt; với *một câu ngắn* chênh lệch chỉ còn ~5 lần vì khối lượng việc quá nhỏ để lấp đầy GPU.

### Streaming — độ trễ chunk đầu và số luồng đồng thời (`infer_stream`, API OpenAI)

| Backend | Luồng đồng thời | TTFA (chunk đầu) | RTF mỗi luồng |
|---|---|---|---|
| **GPU**, 1 luồng | 1 | **~115 ms** (106 ms qua HTTP) | 0,49 |
| **GPU**, 8 luồng | 8 | 164 ms | 0,56 |
| **GPU**, 16 luồng (`max_streams` mặc định) | 16 | median 185 ms (max 339 khi cả 16 bắt đầu cùng lúc); **134 ms** cho request mới đến giữa 15 luồng đang phát | 0,59 |
| **GPU**, 32 luồng (`max_streams=32`) | 32 | ~450 ms | 0,93 — vẫn real-time nhưng không còn biên |
| **CPU** fp32 | 1 | 260–400 ms | 0,55–0,61 (2 request cùng lúc → 1,19, cả hai đứt) |
| **CPU** int8 | 2 | 140–195 ms | 0,35 (2 cùng lúc → 0,58–0,67) |

Luồng ≠ người dùng: một luồng chỉ tồn tại lúc phát một câu trả lời, nên 16 luồng ≈ 45–80 người dùng chatbot thoại. Đặt `max_streams` sát tải thật — mỗi slot thừa cộng ~2,5 ms vào mọi lời gọi codec. GPU rỗi vài giây sẽ hạ xung và request *đầu tiên* sau đó chậm thêm 100–300 ms (khoá xung bằng `nvidia-smi -lgc` hoặc "Prefer maximum performance"). VRAM: 1,1 GB đỉnh ở 16 luồng. Dự đoán cho GPU khác và phương pháp đo: **[docs/streaming.vi.md](docs/streaming.vi.md)**.

---

## 🎓 5. Fine-tune (LoRA) <a name="finetune"></a>

v3 Turbo đã clone giọng từ một clip vài giây. Chỉ fine-tune bằng **LoRA** khi cần bám giọng chặt hơn clone, một phong cách đọc riêng (đọc truyện, tin tức, thuyết minh…), hoặc đọc tốt hơn trên miền văn bản của bạn. Một LoRA dạy **một giọng**: khoảng **10–30 phút** audio sạch của một người nói (cần nhiều giọng thì mỗi giọng một LoRA). Chỉ vài triệu tham số được train nên GPU ~6 GB là đủ.

```bash
uv sync --extra finetune
uv run python finetune/prepare_dataset.py --dataset-dir finetune/dataset   # CPU, không cần torch; mọi clip = một người nói
uv run python finetune/train_lora.py --data finetune/dataset/train.parquet --run my_voice --merge
uv run python finetune/make_voice.py --audio ref.wav --name "Giọng của tôi" --out finetune/output/my_voice/merged
```

```python
tts = Vieneu(mode="v3turbo", backbone_repo="finetune/output/my_voice/merged")   # hoặc repo Hub của bạn
audio = tts.infer("Xin chào!", voice="Giọng của tôi")   # giọng đóng gói sẵn — không cần audio mẫu
```

Model merge đọc bằng đúng giọng đó từ speaker embedding đã đóng gói (không cần audio mẫu) và giữ API của v3 Turbo (preset, batch, streaming) trên backend PyTorch/GPU; muốn clone giọng khác thì dùng model gốc. Định dạng dữ liệu, tuỳ chọn và mẹo: [`finetune/README.md`](finetune/README.md).

---

## 🔬 6. Tổng quan mô hình <a name="backbones"></a>

| Model | Trạng thái | Định dạng | Thiết bị | Song ngữ | Tính năng | Tốc độ ([§4](#benchmarks)) |
|---|---|---|---|---|---|---|
| **VieNeu-TTS-v3** | 🔜 **Sắp ra mắt** | PyTorch | **GPU** | ✅ | ? | ? |
| **VieNeu-TTS-v3-Turbo** *(mặc định)* | ✅ **Hiện hành** | PyTorch/ONNX | **GPU/CPU** | ✅ | **48 kHz, 25 giọng dựng sẵn, clone giọng, tag cảm xúc, hội thoại, streaming (API chuẩn OpenAI)** | **Cực nhanh** — GPU: RTF ≈ 0,02 khi batch, 16 luồng real-time; CPU: RTF ≈ 0,5 (int8 0,35) |
| **VieNeu-TTS-v3-Nano** | 🧪 Preview | ONNX | **CPU yếu / edge** | ⚠️ yếu | 24 kHz, 11 giọng dựng sẵn, clone giọng, tag cảm xúc — **chất lượng thấp hơn (nhất là tiếng Anh / Anh-Việt)** | Nhanh nhất trên CPU (RTF 0,11–0,22) |
| VieNeu-TTS-v2 | ⛔ Đã ngừng | PyTorch | GPU | ✅ | Podcast, Anh-Việt CS | Nhanh (LMDeploy) |
| VieNeu-v2-CPU | ⛔ Đã ngừng | GGUF/ONNX | CPU/Edge | ✅ | Podcast, Anh-Việt CS | Trung bình |
| VieNeu-v2-Turbo | ⛔ Đã ngừng | GGUF/ONNX | CPU/Edge | ✅ | Anh-Việt gọn nhẹ | Nhanh |
| VieNeu-TTS (v1) | ⛔ Đã ngừng | PyTorch | GPU/CPU | ❌ | Ổn định (chỉ tiếng Việt) | Chậm |

> ⛔ Các model **đã ngừng** không còn được cập nhật, chỉ giữ cho hệ thống đang chạy; dự án mới nên dùng **v3 Turbo** (và **v3** khi ra mắt).

> [!TIP]
> Trên **CPU**, backbone chạy `fp32` mặc định (chất lượng tối đa); dùng `Vieneu(precision="int8")` nếu cần nhanh hơn (cần CPU có VNNI). Trên **GPU (CUDA)**, suy luận **tự động batch** — cùng API, không đổi code. Máy quá yếu hoặc deploy trên điện thoại: xem [v3 Nano (preview)](#v3-nano).

---

## 🚀 7. Lộ trình phát triển <a name="roadmap"></a>

- [x] **VieNeu-TTS v3 Turbo** *(chạy trên thiết bị, người dùng cá nhân)*: kiến trúc 48 kHz huấn luyện từ đầu — giọng dựng sẵn, clone giọng tức thì, tag cảm xúc, sinh theo lô, hội thoại nhiều người nói, streaming theo frame; chạy CPU không cần torch.
- [x] **VieNeu-TTS v3 Nano** *(preview)*: model flow-matching 48M cho CPU yếu / thiết bị edge — 11 giọng dựng sẵn + clone giọng, không cần torch.
- [x] **Fine-tune LoRA** cho v3 Turbo — train giọng hoặc phong cách đọc riêng trên một GPU phổ thông.
- [x] **Server streaming GPU cho v3 Turbo**: API chuẩn OpenAI `/v1/audio/speech`, continuous batching trên một CUDA graph — chunk đầu ~115 ms, 16 luồng đồng thời trên RTX 3060, profile Docker `api-gpu` / `api-cpu` ([docs/streaming.vi.md](docs/streaming.vi.md)).
- [ ] **VieNeu-TTS v3 (GPU, bản server)**: model v3 đầy đủ để deploy API / server — chất lượng chốt, điều khiển cảm xúc ổn định, thêm giọng.
- [ ] **Mobile SDK**: hỗ trợ chính thức Android / iOS.

---

## 🤝 8. Hỗ trợ & Liên hệ <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/pnnbao97)
- **Giấy phép:** Apache 2.0 (Sử dụng tự do).

---
## 📑 9. Trích dẫn <a name="citation"></a>

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

## 🤝 Người đóng góp

Cảm ơn tất cả những người tuyệt vời đã đóng góp cho dự án này!

<a href="https://github.com/pnnbao97/VieNeu-TTS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pnnbao97/VieNeu-TTS" />
</a>

---

## 🙏 Lời cảm ơn

Dự án này sử dụng [neucodec](https://huggingface.co/neuphonic/neucodec) (v1/v2) và [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) (v3 Turbo) để mã hoá âm thanh, và [sea-g2p](https://github.com/pnnbao97/sea-g2p) để chuẩn hóa văn bản và phiên âm.

**Được thực hiện với ❤️ dành cho cộng đồng TTS Việt Nam**
