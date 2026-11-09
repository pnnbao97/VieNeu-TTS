# 🔊 Streaming TTS qua API chuẩn OpenAI — số luồng, RTF, độ trễ chunk đầu

Tài liệu này mô tả server `apps/openai_speech.py` (endpoint `POST /v1/audio/speech` tương thích OpenAI), cơ chế streaming của VieNeu-TTS v3 Turbo trên **GPU** và trên **CPU**, toàn bộ số đo trên RTX 3060, cách chọn số luồng, và dự đoán cho các máy khác.

> Bản tiếng Anh: [streaming.md](streaming.md)

## Mục lục

- [Tóm tắt nhanh](#tóm-tắt-nhanh)
- [Chạy server](#chạy-server)
- [API](#api)
- [Ba con số cần hiểu: TTFA, RTF, lead](#ba-con-số-cần-hiểu-ttfa-rtf-lead)
- [GPU: cơ chế streaming](#gpu-cơ-chế-streaming)
- [GPU: số đo trên RTX 3060](#gpu-số-đo-trên-rtx-3060)
- [GPU: chọn `max_streams` và quy đổi ra số người dùng](#gpu-chọn-max_streams-và-quy-đổi-ra-số-người-dùng)
- [GPU: dự đoán cho các máy khác](#gpu-dự-đoán-cho-các-máy-khác)
- [CPU: cơ chế streaming và số đo](#cpu-cơ-chế-streaming-và-số-đo)
- [Tự đo trên máy của bạn](#tự-đo-trên-máy-của-bạn)
- [Tinh chỉnh và sự cố thường gặp](#tinh-chỉnh-và-sự-cố-thường-gặp)

---

## Tóm tắt nhanh

| | GPU (RTX 3060 12 GB, PyTorch bf16) | CPU (6 nhân desktop, ONNX) |
|---|---|---|
| Chunk audio đầu tiên (TTFA), 1 request | **~105–115 ms** qua HTTP | fp32 **~260–400 ms** · int8 **~140–195 ms** |
| RTF (thời gian sinh / thời lượng audio) | 0,49 (1 luồng) → 0,59 (16 luồng) | fp32 0,55–0,61 · int8 0,35 |
| Số luồng phát đồng thời, vẫn real-time | **16** (mặc định), tối đa 32 | fp32 **1** · int8 **2** |
| TTFA khi đủ tải | 16 luồng cùng lúc: ~200 ms; request mới giữa 15 luồng: ~135 ms | request thứ hai xếp hàng hoặc chia đôi tốc độ |
| VRAM | 0,9 GB rỗi, 1,1 GB đỉnh @16 luồng | — |

Cả hai đường dùng **cùng một API**; server tự chọn backend (`VIENEU_BACKEND=auto`: có CUDA → PyTorch, không → ONNX) và tự đặt trần số luồng.

---

## Chạy server

Chọn **một** trong các cách sau — cách nào cũng nghe ở `http://0.0.0.0:8000`:

```bash
# (a) Chạy từ repo — có CUDA thì GPU, không thì CPU
uv run python -m apps.openai_speech

# (b) Chạy từ repo, ép CPU int8 (nhanh ~2x fp32, cần CPU có VNNI — xem phần CPU)
VIENEU_BACKEND=onnx VIENEU_PRECISION=int8 uv run python -m apps.openai_speech

# (c) Docker, container GPU
docker compose -f docker/docker-compose.yml --profile api-gpu up

# (d) Docker, container chỉ CPU
docker compose -f docker/docker-compose.yml --profile api-cpu up
```

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VIENEU_BACKEND` | `auto` | `pytorch` (GPU) · `onnx` (CPU) |
| `VIENEU_DEVICE` | `auto` | `cuda` · `cpu` |
| `VIENEU_PRECISION` | `fp32` | CPU: `fp32` · `int8` |
| `VIENEU_MAX_STREAMS` | GPU 16 · CPU 1 (int8: 2) | Số luồng phục vụ đồng thời. GPU: kích thước batch liên tục — **đặt sát tải thật** (xem bên dưới) |
| `VIENEU_QUEUE` | = `MAX_STREAMS` | Số request được xếp hàng chờ slot; quá → `429` ngay |
| `VIENEU_QUEUE_TIMEOUT` | 10 | Giây chờ tối đa trong hàng; hết → `429` |
| `VIENEU_API_KEY` | (trống) | Nếu đặt, yêu cầu `Authorization: Bearer <key>`. Để trống mà `HOST` không phải loopback → cảnh báo lúc khởi động |
| `VIENEU_WATERMARK` | 1 | Đóng watermark âm thanh (Perth) từng chunk |
| `HOST` / `PORT` | `127.0.0.1` / 8000 | `HOST=0.0.0.0` để máy khác gọi được (các profile Docker đã đặt sẵn) — kèm `VIENEU_API_KEY` |

Server chỉ chạy **một worker** (`uvicorn workers=1`): model và scheduler nằm trong tiến trình; GPU phục vụ nhiều luồng bằng batching bên trong chứ không bằng nhiều tiến trình. Muốn scale ngang thì chạy nhiều container, mỗi container một GPU.

Lúc khởi động server tự chạy một câu ngắn để nạp graph (GPU: capture CUDA graph ~0,7 s; CPU: nạp ONNX), nên request đầu tiên của client không phải trả giá này. Tổng thời gian sẵn sàng ~12 s trên máy đo (đã có model trong cache HF).

---

## API

### `POST /v1/audio/speech`

Body JSON — các trường của OpenAI cộng vài trường mở rộng:

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `input` | (bắt buộc) | Văn bản, tối đa 20 000 ký tự; server tự cắt chunk ≤ `max_chars` theo câu |
| `model` | `vieneu-v3-turbo` | Chuỗi bất kỳ, chỉ để tương thích |
| `voice` | giọng mặc định | Tên preset trong `GET /v1/voices` hoặc giọng đã thêm bằng `POST /v1/voices` |
| `response_format` | `wav` | `pcm` (s16le mono, không header) · `wav` (header độ dài "không xác định", player phát ngay). `mp3/opus/aac/flac` → `400` |
| `stream_format` | `audio` | `audio`: body là bytes theo chunked transfer · `sse`: Server-Sent Events (bên dưới) |
| `sample_rate` | 48000 | 48000 (gốc) · 24000 (chuẩn `pcm` của OpenAI) · 16000 · 8000 — resample từng chunk bằng soxr, không thêm trễ |
| `speed`, `instructions` | — | Nhận để tương thích, **bỏ qua** (header `X-VieNeu-Ignored` cho biết) |
| `temperature`, `top_k`, `top_p`, `repetition_penalty` | 0,8 / 25 / 0,95 / 1,2 | Sampling; cho phép 0–2 / 1–1024 / (0, 1] / 1–2, ngoài khoảng (hoặc NaN) → `400` |
| `max_chars` | 256 | Độ dài chunk văn bản tối đa, 64–512 |

Response: `200` với `Transfer-Encoding: chunked`; header `X-Request-Id`, `X-Sample-Rate`. Lỗi trả đúng dạng OpenAI `{"error": {"message", "type", "code"}}`: `400` tham số sai, `401` sai key, `429` hết slot (kèm `Retry-After`).

**`stream_format=sse`** — `Content-Type: text/event-stream`, mỗi event một dòng `data:`:

```
data: {"type":"speech.audio.delta","audio":"<base64 PCM s16le>"}
data: {"type":"speech.audio.delta","audio":"..."}
data: {"type":"speech.audio.done","usage":{"output_samples":222720,"sample_rate":48000,"seconds":4.64}}
```

Với `response_format=wav` + `sse`, event đầu tiên là header WAV (base64).

### Các endpoint khác

| | |
|---|---|
| `GET /v1/models` | Model, `sample_rate`, `backend`, `max_streams`, định dạng hỗ trợ |
| `GET /v1/voices` | Danh sách preset (`id`, `name`, `description`, `gender`) |
| `POST /v1/voices` | multipart `name`, `file` (clip 3–8 s), `denoise` — nhân bản giọng, giữ trong bộ nhớ tiến trình. Trùng tên giọng có sẵn hoặc alias → `409` |
| `GET /health` | `active` (đang phát), `waiting` (đang xếp hàng), `max_streams`. `503` khi scheduler streaming GPU đã chết (không tự hồi phục — healthcheck của Docker sẽ restart container) |

### Client

Python, SDK OpenAI:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")   # api_key = VIENEU_API_KEY nếu có đặt
with client.audio.speech.with_streaming_response.create(
    model="vieneu-v3-turbo", voice="Mai Anh", input="Xin chào! Đây là chế độ streaming của VieNeu, phát tới đâu nghe tới đó.", response_format="pcm",
) as r:
    for chunk in r.iter_bytes(4096):       # s16le 48 kHz mono
        player.write(chunk)
```

curl:

```bash
curl -N http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input":"Xin chào các bạn.","voice":"Mai Anh","response_format":"wav"}' | ffplay -i - -nodisp -autoexit
```

Trình duyệt (SSE):

```js
const r = await fetch("/v1/audio/speech", {method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({input: "Xin chào", response_format: "pcm", stream_format: "sse", sample_rate: 24000})});
for await (const line of readLines(r.body)) {           // tách theo "\n\n", bỏ tiền tố "data: "
  const ev = JSON.parse(line);
  if (ev.type === "speech.audio.delta") playPcm(atob(ev.audio));
}
```

Pipecat / LiveKit Agents / Vercel AI SDK: dùng plugin TTS OpenAI của họ, đổi `base_url`, đặt `response_format="pcm"` và, nếu framework giả định 24 kHz, thêm `sample_rate: 24000` (extra body) hoặc cấu hình sample rate của framework thành 48 000.

`examples/openai_speech_client.py` là client mẫu có đo TTFA/RTF và chế độ `--bench N`.

---

## Ba con số cần hiểu: TTFA, RTF, lead

- **TTFA** (time to first audio): từ lúc gửi request tới khi nhận **byte audio đầu tiên**. Với `wav`, header tới sau ~2 ms nhưng chưa phải audio; số TTFA trong tài liệu này luôn tính theo audio thật.
- **RTF** (real-time factor) = thời gian sinh / thời lượng audio. RTF < 1 là sinh nhanh hơn phát; luồng **không bao giờ đứt** nếu RTF < 1 từ đầu tới cuối. RTF 0,5 nghĩa là còn dư một nửa thời gian.
- **Lead**: audio đã nhận trừ thời gian đã trôi kể từ chunk đầu. Client phát ngay khi có chunk đầu thì lead âm = tiếng bị đứt. Trên GPU lead tối thiểu đo được là +80 ms ở mọi mức tải ≤ 16 luồng; trên CPU là +320 ms (lead-in 4 khung). Client nên **prebuffer 150–300 ms** để hấp thụ jitter mạng.

Một khung codec = 80 ms audio (12,5 khung/giây). Chunk đầu tiên gồm 2 khung (GPU) hoặc 4 khung (CPU); các chunk sau 4 khung = 320 ms.

---

## GPU: cơ chế streaming

Code: `src/vieneu/v3_turbo_serve/stream.py` (+ `fused.py`). Tóm tắt:

1. **Một CUDA graph, B slot** (`max_streams`). Mỗi khung (80 ms audio) là **một lần replay graph** cho tất cả slot cùng lúc: acoustic decoder 16 codebook + sampling + phạt lặp + bước backbone. 7 ms/khung ở B = 1, 10 ms ở B = 32 — gần như không phụ thuộc số luồng.
2. **Continuous batching**: request mới được prefill (HF, ~21 ms cho một prompt) rồi **nạp vào một slot trống của graph đang chạy** (`StaticBackbone.load_row`: KV cache là ring buffer, prompt được ghi căn phải theo con trỏ ghi chung; RoPE tính theo vị trí riêng từng row). Request xong (EOS hoặc chạm trần khung) thì slot trống ngay, không phải đợi ai. Tham số sampling là tensor theo row nên mỗi request giữ temperature/top-k/top-p/penalty của mình.
3. **Codec MOSS một phiên streaming chung**: cứ 4 khung, mã của mọi row đang chạy đi qua **một** lời gọi `batch_decode(streaming=True)`; row mới được decode sớm ngay khi có 2 khung (lead-in) thay vì đợi mốc 4. Decoder không lookahead, audio stream khớp decode đầy đủ (sai số 6e-4).
4. **Một worker thread** sở hữu GPU. `infer_stream` từ N thread (N request HTTP) chỉ là N hàng đợi audio; server không tạo thêm tiến trình.
5. **Capture chạy một mình.** Ghi một CUDA graph (graph của scheduler này, hoặc graph mới mà batch engine của `infer`/`infer_batch` cần) phải giữ một khoá chung cả tiến trình (`_v3_turbo_engine/capture_lock.py`); mọi bước suy luận GPU khác của v3 Turbo giữ khoá ấy ở chế độ chia sẻ (trừ lúc nạp model, và trừ model v1/v2 chạy cùng tiến trình). Một thread khác cấp phát bộ nhớ, đồng bộ hay lấy mẫu đúng lúc capture có thể làm capture hỏng, và capture hỏng thì bộ sinh số ngẫu nhiên của PyTorch kẹt ở trạng thái "đang capture" (`Offset increment outside graph capture`) tới khi khởi động lại tiến trình. Cái giá: mỗi lần cần graph mới, các stream đang phát dừng khoảng 0,5 s chờ capture xong. Việc CUDA của chính bạn trong cùng tiến trình (model khác, `torch.cuda.empty_cache()`) không được khoá này che: đừng chạy nó song song với VieNeu, hoặc bọc nó trong `with CAPTURE_LOCK.shared():` (`from vieneu._v3_turbo_engine.capture_lock import CAPTURE_LOCK`, từ 3.8.4; dành cho người dùng nâng cao: đường dẫn nội bộ, có thể đổi ở bản minor sau). Chỉ bọc đúng các lệnh CUDA của bạn: đừng gọi VieNeu (`infer`, `infer_batch`, `infer_stream`, `encode_reference`, `warm_fused`) hay chờ thread khác bên trong khối này, nếu không tiến trình có thể treo. Lời gọi VieNeu nào phải chờ thread khác sẽ báo `RuntimeError` ngay tại đó thay vì treo.

Chi phí thực (RTX 3060, bf16):

| Thành phần | Chi phí | Ghi chú |
|---|---|---|
| Prefill (HF) | 21 ms (1 prompt) · 39 (8) · 73 (16) · 125 (32) | Trả một lần mỗi request, chạy giữa hai khung |
| Khung graph | 7 ms (B=1) · 8 (8) · 9 (16) · 10 (32) | Cho **toàn bộ** batch |
| **Codec** (4 khung) | **~65 ms + ~2,5 ms × số slot đã cấp** | Không phụ thuộc số khung hay số row đang thực dùng; launch-bound |

Điểm quan trọng: **codec là toàn bộ chi phí**, và chi phí ấy tỷ lệ với `max_streams` (số slot phiên codec giữ), không phải số luồng thực đang chạy. Vì thế `max_streams=32` làm cả một luồng đơn lẻ cũng chậm đi. Chế độ `use_cuda_graph` sẵn có của codec nhanh gấp 3 nhưng cho audio sai (tương quan 0,2 với decode thường) nên không dùng.

Đường cũ (trước 15/09/2026): engine đơn không CUDA graph, TTFA 280–360 ms, giữ khoá cả lượt → 1 luồng, RTF 0,85.

---

## GPU: số đo trên RTX 3060

Máy đo: RTX 3060 12 GB, i5 thế hệ 12 (6 P-core), Windows 11, torch 2.8 + cu128, bf16, giọng preset (ref 69 khung), câu 88–147 ký tự, `apply_watermark=False` khi đo trong tiến trình. Ngày 15/09/2026.

### Gọi `infer_stream` trực tiếp (trong tiến trình), N thread bắt đầu cùng lúc

`max_streams = 16` (mặc định):

| N luồng cùng lúc | TTFA median | TTFA max | RTF xấu nhất | Lead min |
|---|---|---|---|---|
| 1 | **115 ms** | 115 | 0,49 | +160 ms |
| 2 | 130 | 130 | 0,51 | +80 |
| 4 | 130 | 130 | 0,52 | +80 |
| 8 | 164 | 165 | 0,56 | +80 |
| 16 | 185 | 339 | **0,59** | +80 |

Request mới đến khi các luồng khác đang phát dở (`max_streams=16`): giữa 3 luồng **115 ms**, giữa 7 luồng **145 ms**, giữa 15 luồng **134 ms**.

`max_streams = 8`:

| N | TTFA median | RTF xấu nhất | Request mới giữa N−1 |
|---|---|---|---|
| 1 | **89 ms** | 0,39 | — |
| 2 | 151 | 0,44 | — |
| 4 | 147 | 0,43 | 108 ms |
| 8 | 162 | 0,45 | 114 ms |

`max_streams = 32`:

| N | TTFA median | TTFA max | RTF xấu nhất | Lead min | Request mới giữa N−1 |
|---|---|---|---|---|---|
| 1 | 141 ms | 141 | 0,70 | +160 | — |
| 2 | 165 | 165 | 0,71 | +80 | — |
| 4 | 174 | 174 | 0,73 | +80 | 274 ms |
| 8 | 207 | 208 | 0,77 | +80 | 180 ms |
| 16 | 220 | 389 | 0,82 | +66 | 227 ms |
| 24 | 448 | 453 | 0,86 | **−1 ms** | — |
| 32 | 456 | 629 | **0,93** | **−28 ms** | 181 ms |

VRAM (model 0,5 GB đã tính): `max_streams=8` 0,64 GB rỗi / 0,72 GB đỉnh; 16: 0,91 / 1,10; 32: 1,45 / 1,82.

### Qua HTTP (server `apps/openai_speech.py`, client cùng máy, watermark bật)

| Kịch bản | TTFA | RTF |
|---|---|---|
| 1 request, `pcm` 48 kHz | **106 ms** | 0,47 |
| 1 request, `pcm` + `sample_rate=24000` | 105 ms | 0,46 |
| 1 request, `sse` | 99 ms | — |
| 4 client cùng lúc | median 160 ms | 0,51 |
| 8 client cùng lúc | 207 ms | 0,53 |
| 16 client cùng lúc | 197 ms (max 336) | 0,59 |
| 40 client cùng lúc (16 slot + hàng đợi 16) | 32 OK, 8 × `429`; 16 request xếp hàng có TTFA 1,5–3 s | — |

TTFA qua HTTP xấp xỉ trong tiến trình (+~0–10 ms); qua mạng thật cộng thêm RTT.

### GPU "nguội" sau khi rỗi: request đầu tiên chậm thêm 100–300 ms

Mọi số ở trên là lúc GPU đang **ấm** (có request trong vòng ~2 s trước). Khi server rỗi, driver NVIDIA hạ GPU về trạng thái tiết kiệm điện trong ~2 s (RTX 3060: **P8, 210 MHz, 12 W**); request đầu tiên sau đó phải chạy prefill, 2 khung lead-in và lời gọi codec ở xung thấp rồi mới kéo xung lên (~1950 MHz). Đo được:

| Kịch bản | TTFA |
|---|---|
| 1 request, GPU ấm | 118 ms |
| 1 request sau 8 s rỗi | **403 ms** |
| 16 client cùng lúc, GPU ấm | median 194 ms, max 354 |
| 16 client cùng lúc sau 5–8 s rỗi | median 330–400 ms, max 450–540 |
| 16 client ngay sau *một* request bất kỳ | median 179 ms |

Bất kỳ request nào cũng làm GPU ấm lại; trong một cuộc hội thoại liên tục chỉ câu đầu tiên chịu phí này. Giữ ấm từ trong tiến trình **không hiệu quả**: đã thử matmul 2048² và replay cả graph mỗi 100–250 ms, driver vẫn hạ về P5/P8 (nó cần tải liên tục). Cách xử lý thuộc về máy chủ:

- **Khoá xung** (cần quyền admin/root): `nvidia-smi -lgc 1500,2100` (mở lại: `nvidia-smi -rgc`). Linux thêm `nvidia-smi -pm 1`. Trong Docker phải chạy trên host, không phải trong container.
- **Windows**: NVIDIA Control Panel → Manage 3D settings → *Power management mode* = **Prefer maximum performance** (đặt riêng cho `python.exe` của venv để không ảnh hưởng máy).
- Hoặc chấp nhận: chỉ request đầu tiên sau > 2 s rỗi bị ảnh hưởng.

Trên các card khác cũng có hiện tượng này (mọi GPU NVIDIA đều có P-state); mức chênh tuỳ độ chênh xung idle/boost của card.

### Sự khác nhau giữa "N cùng lúc" và "request mới giữa N−1"

"N cùng lúc" là trường hợp xấu nhất: N prefill dồn vào cùng vài tick (mỗi tick nhận tối đa 8 request), N lời gọi codec lead-in chồng nhau. Tải thực tế là request đến rải rác, tương ứng cột "request mới giữa N−1": ở 16 luồng vẫn ~135 ms.

---

## GPU: chọn `max_streams` và quy đổi ra số người dùng

| Mục tiêu | `max_streams` | Kết quả trên 3060 |
|---|---|---|
| Độ trễ thấp nhất, ít người dùng | 8 | 1 luồng 89 ms, 8 luồng 162 ms, GPU dùng 45 % |
| **Cân bằng (mặc định)** | **16** | 1 luồng 115 ms, 16 luồng ≤ 200 ms, GPU dùng 59 % |
| Thông lượng tối đa | 32 | 32 luồng vẫn không đứt (RTF 0,93) nhưng chunk đầu 450 ms và không còn biên; 24 là mức chịu được |

Quy tắc: `max_streams` = số luồng **đồng thời** cao nhất bạn cần, không hơn — mỗi slot thừa cộng ~2,5 ms vào **mọi** lời gọi codec cho **mọi** người.

**Luồng ≠ người dùng.** Một luồng chỉ tồn tại trong lúc phát một câu trả lời (vài giây). Với chatbot thoại, mỗi người dùng chiếm luồng ~20–35 % thời gian → 16 luồng ≈ **45–80 người dùng hoạt động**; với đọc tin/sách liên tục thì 16 luồng = 16 người. Tính: `người dùng ≈ max_streams / tỷ lệ thời gian nghe TTS`.

---

## GPU: dự đoán cho các máy khác

**Chưa đo trên máy nào ngoài RTX 3060** — bảng dưới là suy luận từ cấu trúc chi phí, dùng để lập kế hoạch rồi kiểm chứng bằng phần [Tự đo](#tự-đo-trên-máy-của-bạn).

Ba điều rút ra từ số đo giúp dự đoán:

1. **VRAM không phải giới hạn.** 16 luồng đỉnh 1,1 GB, 32 luồng 1,8 GB. Card 4 GB đủ cho 16 luồng, 6 GB đủ cho 32.
2. **Giới hạn là chi phí "mỗi lời gọi" (launch-bound), không phải FLOPS.** Backbone 12 tầng 768 chiều và codec 11M tham số đều bé; thời gian là hàng trăm kernel launch nhỏ. Vì vậy GPU mạnh hơn nhiều (4090, A100) **không** cho TTFA thấp hơn nhiều; CPU host nhanh, driver, PCIe và việc không chia sẻ GPU với việc khác quan trọng ngang GPU.
3. **Kiến trúc GPU ảnh hưởng qua dtype và CUDA graph.** Ampere trở lên (RTX 30/40, A-series) chạy bf16 như máy đo. Turing/Pascal (GTX 16, RTX 20, T4, GTX 10) không có bf16 → code tự chuyển **fp16**; đường này **chưa được kiểm chất lượng** (nguy cơ tràn số ở activation). CUDA graph cần driver CUDA ≥ 11, có trên mọi card từ Maxwell.

| Máy | VRAM | Dự đoán TTFA 1 luồng | Dự đoán số luồng real-time | Ghi chú |
|---|---|---|---|---|
| RTX 3060 12 GB (đo) | 12 | 105–115 ms | 16 thoải mái, 32 tối đa | bf16 |
| RTX 3050 / 4060 8 GB, RTX 3060 Ti/4070 | 6–12 | 100–130 ms | 16, tối đa ~32 | Cùng thế hệ, launch-bound nên gần như bằng 3060; 4060 laptop bị power limit có thể chậm 10–20 % |
| RTX 3050 laptop 4 GB, RTX A2000 | 4 | 120–160 ms | 8–12 (`max_streams=8`–`12`) | VRAM đủ; clock thấp và chia sẻ với hệ điều hành → giữ `max_streams` nhỏ |
| RTX 4090 / A10 / L4 | 16–24 | 90–110 ms | 24–32 | Phần codec nhanh hơn, phần launch không; trần thực do CPU host |
| T4 16 GB (cloud) | 16 | 150–220 ms | 8–12 | Turing → fp16 (chưa kiểm chất lượng); clock thấp; VM cloud thường CPU yếu → overhead launch tăng |
| GTX 1660 / RTX 2060 6 GB | 6 | 140–200 ms | 8–12 | Như T4 nhưng desktop CPU nhanh hơn |
| GTX 1650 4 GB, GTX 1050 Ti | 4 | 180–250 ms | 4–8 | fp16 (Pascal không có tensor core: codec fp32 chậm hơn ~1,5–2×) |
| GPU < 4 GB | <4 | — | 2–4 hoặc chạy CPU | Cần `max_streams` ≤ 4 để cache tĩnh + codec vừa; không có lợi rõ so với CPU int8 |

Cách đọc bảng: TTFA ≈ prefill (20–40 ms) + 2 khung graph (15–25 ms) + một lời gọi codec (60–120 ms tuỳ card/CPU host). Số luồng ≈ (80 ms − khung graph) / (chi phí codec ÷ 4 khung), giữ RTF ≤ 0,6–0,7 để còn biên.

Với card 4–6 GB, đặt `VIENEU_MAX_STREAMS=8` (hoặc 12) ngay từ đầu: vừa giảm chi phí codec, vừa giữ VRAM đỉnh < 1 GB.

---

## CPU: cơ chế streaming và số đo

Code: `src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py` (`infer_stream`). Máy chỉ có CPU chạy **hoàn toàn không torch**: backbone và acoustic decoder là ONNX, codec là `moss_audio_tokenizer_decode_step.onnx` — bản streaming **bit-exact** với decoder đầy đủ (không lookahead, không click ở mép chunk).

Cơ chế:

1. Prefill prompt (ref codes + phoneme) một lần, rồi mỗi khung: acoustic decoder 16 codebook (16 lời gọi ONNX nhỏ) + một bước backbone với KV cache.
2. **Lead-in 4 khung** (320 ms audio) rồi mới decode chunk đầu — có sẵn 320 ms đệm để client không đứt khi tốc độ sinh dao động.
3. Sau đó **pacing thích nghi** (`_target_frames`): nếu audio đã phát vượt thời gian trôi < 0,2 s thì decode từng 4 khung, < 0,55 s → 6, < 1,1 s → 8, còn dư nhiều → tới 25 khung một lần (ít lời gọi codec hơn, RTF tốt hơn).
4. ORT dùng `intra_op_threads` = số nhân vật lý (tối đa 8, chỉnh bằng `Vieneu(threads=...)`), `inter_op = 1`, không spin — để không chiếm hết CPU khi rỗi.
5. Khoá theo **từng khung** (RLock), nên hai request cùng lúc **xen kẽ** nhau chứ không xếp hàng hoàn toàn — nhưng cùng chia số nhân, mỗi luồng chậm đi tương ứng.

Số đo (i5 thế hệ 12, 6 P-core, 6 thread ORT; câu 27 / 88 / 147 ký tự):

| Precision | TTFA | RTF 1 luồng | Lead min | 2 request cùng lúc |
|---|---|---|---|---|
| **fp32** (mặc định) | 264–282 · 313–357 · 345–401 ms | 0,61 · 0,57 · 0,55 | +320 ms | TTFA 399 / 564 ms, **RTF 1,19 / 1,19 → cả hai đứt tiếng** |
| **int8** | 141–144 · 163–169 · 192–194 ms | 0,35 · 0,35 · 0,35 | +320 ms | TTFA 224 / 336 ms, RTF 0,58 / 0,67 → vẫn real-time |

Kết luận cho CPU:

- **fp32: đúng 1 luồng.** Server mặc định `max_streams=1`; request thứ hai chờ trong hàng đợi (`VIENEU_QUEUE`, mặc định = `max_streams`; profile docker `api-cpu` đặt 4) tối đa `VIENEU_QUEUE_TIMEOUT` giây rồi `429`. Thời gian chờ = thời lượng câu đang phát (một câu 7 s audio sinh mất ~4 s).
- **int8: 2 luồng** (mặc định `max_streams=2`) trên 6 nhân, RTF ~0,65 mỗi luồng; 3 luồng sẽ chạm 1,0. Máy 8–12 nhân vật lý có thể đặt 3–4, đo trước khi tin.
- TTFA phụ thuộc **độ dài câu đầu tiên** (prefill dài hơn) và **số nhân**: laptop 4 nhân sẽ thấy fp32 ~400–600 ms, int8 ~200–300 ms; RTF fp32 có thể vượt 1 trên CPU 2–4 nhân → chỉ dùng int8.
- int8 **cần CPU có VNNI** (Intel từ Cascade Lake/Ice Lake/Alder Lake, AMD Zen 4). Trên CPU không VNNI, acoustic decoder int8 bão hoà và sinh tiếng lảm nhảm — dùng fp32 (xem memory dự án về lỗi này).
- Client cần **prebuffer ≥ 300 ms** — bằng đúng lead-in — vì CPU bị ảnh hưởng bởi mọi tiến trình khác trên máy.
- Không có scheduler batching trên CPU: nhiều request là chia nhân, không có "miễn phí" như trên GPU.

Muốn nhiều luồng hơn trên máy không GPU: chạy nhiều container, mỗi container ghim vào một nhóm nhân (`docker --cpuset-cpus`), hoặc chấp nhận `429` và để client thử lại.

---

## Tự đo trên máy của bạn

```bash
uv run python -m apps.openai_speech &                    # hoặc profile docker api-gpu / api-cpu
uv run python examples/openai_speech_client.py           # 1 request: TTFA, RTF, out_stream.wav
uv run python examples/openai_speech_client.py --bench 8 # 8 client cùng lúc
uv run python examples/openai_speech_client.py --bench 16
```

Đọc kết quả:

- `RTF max` phải **< 1**, tốt nhất ≤ 0,7. Nếu `--bench N` cho RTF > 0,8 thì N đã vượt sức máy → giảm `VIENEU_MAX_STREAMS`.
- `TTFA max` lớn hơn median nhiều (như 339 vs 185 ở 16 luồng) là do prefill dồn; tải thực rải rác sẽ sát median.
- Log server in mỗi request: `ttfa=… total=… audio=… rtf=… active=…`.
- GPU: `nvidia-smi` khi bench để xem VRAM; `torch.cuda.max_memory_allocated()` trong tiến trình cho số chính xác hơn.

Muốn đo sâu hơn (từng thành phần: prefill / khung / codec), xem docstring `src/vieneu/v3_turbo_serve/stream.py` và test `tests/fused/cases.py`.

---

## Tinh chỉnh và sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Chunk đầu chậm dù ít người dùng (GPU) | `max_streams` quá lớn → codec đệm nhiều slot | Hạ `VIENEU_MAX_STREAMS` sát tải (8 cho ≤ 8 luồng) |
| Request đầu tiên sau vài giây rỗi chậm hơn 100–300 ms (GPU) | GPU hạ xung khi rỗi (P8) | Khoá xung `nvidia-smi -lgc` hoặc "Prefer maximum performance" — xem mục *GPU "nguội"* |
| `429` thường xuyên | Hết slot + hàng đợi | Tăng `VIENEU_QUEUE`/`VIENEU_QUEUE_TIMEOUT` nếu chấp nhận chờ; hoặc thêm GPU/container |
| Tiếng đứt quãng ở client | Client không prebuffer, hoặc RTF > 1 (CPU fp32 nhiều request) | Prebuffer 150–300 ms (GPU) / ≥ 300 ms (CPU); CPU dùng int8, giữ 1–2 luồng |
| Request đầu tiên sau khởi động chậm 1–2 s | Warm-up chưa xong | Server đã tự warm; chờ `/health` trả `ok` trước khi mở traffic |
| Mọi request lỗi `Offset increment outside graph capture encountered unexpectedly` (GPU) | Một lần capture CUDA graph bị hỏng (SDK < 3.8.4, hoặc việc CUDA khác chạy cùng lúc capture) | Khởi động lại tiến trình; nâng lên `vieneu>=3.8.4`; chỉ bọc lệnh CUDA của bạn trong `CAPTURE_LOCK.shared()` (mục 5 phần *GPU: cơ chế streaming*); cùng lắm thì `VIENEU_FUSED_FRAME=0` |
| `RuntimeError: ... while this thread holds CAPTURE_LOCK` | Gọi VieNeu bên trong khối `CAPTURE_LOCK.shared()` của bạn | Chỉ để lệnh CUDA của bạn trong khối; gọi VieNeu ở ngoài |
| Muốn tắt đường CUDA graph (gỡ lỗi) | — | `VIENEU_FUSED_FRAME=0` → `infer_stream` quay về engine đơn (1 luồng, ~300 ms) |
| GPU Turing/Pascal, tiếng lạ | fp16 fallback chưa kiểm chứng | Thử `Vieneu(dtype="float32")` (chậm hơn ~1,5×) hoặc báo issue kèm mẫu |
| Cần mp3/opus | Chưa hỗ trợ | Đặt reverse proxy/ffmpeg phía trước, hoặc client nhận `pcm` rồi tự mã hoá; thêm ~20–40 ms trễ |
| `repetition_window` khác mặc định không có tác dụng (GPU) | Cửa sổ phạt lặp là kích thước ring cố định của scheduler | Dùng `repetition_penalty` để chỉnh; window mặc định dùng chung |
| Ngắt kết nối giữa chừng có tốn tài nguyên không? | Không | Đóng response → generator đóng → slot GPU được giải phóng ở tick kế tiếp (đã kiểm) |
