# 🔊 Streaming TTS through an OpenAI-compatible API — streams, RTF, first-audio latency

This document covers the server `apps/openai_speech.py` (an OpenAI-compatible `POST /v1/audio/speech`), how VieNeu-TTS v3 Turbo streams on **GPU** and on **CPU**, every measurement taken on an RTX 3060, how to pick the number of streams, and estimates for other machines.

> Tiếng Việt: [streaming.vi.md](streaming.vi.md)

## Contents

- [At a glance](#at-a-glance)
- [Running the server](#running-the-server)
- [API](#api)
- [Three numbers to understand: TTFA, RTF, lead](#three-numbers-to-understand-ttfa-rtf-lead)
- [GPU: how streaming works](#gpu-how-streaming-works)
- [GPU: measurements on an RTX 3060](#gpu-measurements-on-an-rtx-3060)
- [GPU: choosing `max_streams`, streams vs. users](#gpu-choosing-max_streams-streams-vs-users)
- [GPU: estimates for other machines](#gpu-estimates-for-other-machines)
- [CPU: how streaming works, measurements](#cpu-how-streaming-works-measurements)
- [Measure it on your machine](#measure-it-on-your-machine)
- [Tuning and troubleshooting](#tuning-and-troubleshooting)

---

## At a glance

| | GPU (RTX 3060 12 GB, PyTorch bf16) | CPU (6-core desktop, ONNX) |
|---|---|---|
| First audio chunk (TTFA), one request | **~105–115 ms** over HTTP | fp32 **~260–400 ms** · int8 **~140–195 ms** |
| RTF (generation time / audio duration) | 0.49 (1 stream) → 0.59 (16 streams) | fp32 0.55–0.61 · int8 0.35 |
| Concurrent streams that stay real-time | **16** (default), 32 at most | fp32 **1** · int8 **2** |
| TTFA under load | 16 starting at once: ~200 ms; a new request among 15 playing: ~135 ms | second request queues, or both slow down |
| VRAM | 0.9 GB idle, 1.1 GB peak @16 streams | — |

Both paths sit behind **the same API**; the server picks the backend (`VIENEU_BACKEND=auto`: CUDA → PyTorch, otherwise ONNX) and the stream ceiling by itself.

---

## Running the server

Pick **one** of the following — every variant listens on `http://0.0.0.0:8000`:

```bash
# (a) From the repo — GPU when CUDA is present, otherwise CPU
uv run python -m apps.openai_speech

# (b) From the repo, forcing CPU int8 (~2x faster than fp32; needs a CPU with VNNI — see the CPU section)
VIENEU_BACKEND=onnx VIENEU_PRECISION=int8 uv run python -m apps.openai_speech

# (c) Docker, GPU container
docker compose -f docker/docker-compose.yml --profile api-gpu up

# (d) Docker, CPU-only container
docker compose -f docker/docker-compose.yml --profile api-cpu up
```

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `VIENEU_BACKEND` | `auto` | `pytorch` (GPU) · `onnx` (CPU) |
| `VIENEU_DEVICE` | `auto` | `cuda` · `cpu` |
| `VIENEU_PRECISION` | `fp32` | CPU: `fp32` · `int8` |
| `VIENEU_MAX_STREAMS` | GPU 16 · CPU 1 (int8: 2) | Streams served at once. On GPU this is the continuous-batch size — **match it to your real load** (see below) |
| `VIENEU_QUEUE` | = `MAX_STREAMS` | Requests allowed to wait for a slot; beyond that → immediate `429` |
| `VIENEU_QUEUE_TIMEOUT` | 10 | Max seconds in the queue; then `429` |
| `VIENEU_API_KEY` | (empty) | If set, requires `Authorization: Bearer <key>`. Unset on a non-loopback `HOST` → a warning at start-up |
| `VIENEU_WATERMARK` | 1 | Perth audio watermark, per chunk |
| `HOST` / `PORT` | `127.0.0.1` / 8000 | `HOST=0.0.0.0` to serve other machines (the Docker profiles set it) — together with `VIENEU_API_KEY` |

The server runs **one worker** (`uvicorn workers=1`): the model and the scheduler live in the process; the GPU serves many streams by batching inside, not by forking. Scale out by running several containers, one GPU each.

On start-up the server synthesizes a short sentence to warm up (GPU: CUDA-graph capture, ~0.7 s; CPU: ONNX session load), so the first client does not pay for it. Ready in ~12 s on the test machine with the model already in the HF cache.

---

## API

### `POST /v1/audio/speech`

JSON body — OpenAI's fields plus a few extras:

| Field | Default | Notes |
|---|---|---|
| `input` | (required) | Text, up to 20,000 chars; split into chunks ≤ `max_chars` at sentence boundaries |
| `model` | `vieneu-v3-turbo` | Any string, for compatibility only |
| `voice` | default voice | A preset from `GET /v1/voices` or a voice added with `POST /v1/voices` |
| `response_format` | `wav` | `pcm` (s16le mono, headerless) · `wav` (header with "unknown" length — players start at once). `mp3/opus/aac/flac` → `400` |
| `stream_format` | `audio` | `audio`: raw bytes, chunked transfer · `sse`: Server-Sent Events (below) |
| `sample_rate` | 48000 | 48000 (native) · 24000 (OpenAI's `pcm` rate) · 16000 · 8000 — resampled per chunk with soxr, no added latency |
| `speed`, `instructions` | — | Accepted for compatibility, **ignored** (`X-VieNeu-Ignored` header says so) |
| `temperature`, `top_k`, `top_p`, `repetition_penalty` | 0.8 / 25 / 0.95 / 1.2 | Sampling; allowed 0–2 / 1–1024 / (0, 1] / 1–2, anything else (or NaN) → `400` |
| `max_chars` | 256 | Max text chunk length, 64–512 |

Response: `200` with `Transfer-Encoding: chunked`; headers `X-Request-Id`, `X-Sample-Rate`. Errors use OpenAI's shape `{"error": {"message", "type", "code"}}`: `400` bad parameter, `401` bad key, `429` no slot (with `Retry-After`).

**`stream_format=sse`** — `Content-Type: text/event-stream`, one `data:` line per event:

```
data: {"type":"speech.audio.delta","audio":"<base64 PCM s16le>"}
data: {"type":"speech.audio.delta","audio":"..."}
data: {"type":"speech.audio.done","usage":{"output_samples":222720,"sample_rate":48000,"seconds":4.64}}
```

With `response_format=wav` + `sse`, the first event is the WAV header (base64).

### Other endpoints

| | |
|---|---|
| `GET /v1/models` | Model, `sample_rate`, `backend`, `max_streams`, supported formats |
| `GET /v1/voices` | Presets (`id`, `name`, `description`, `gender`) |
| `POST /v1/voices` | multipart `name`, `file` (3–8 s clip), `denoise` — clones a voice, kept in process memory. The name of a built-in voice or alias → `409` |
| `GET /health` | `active` (playing), `waiting` (queued), `max_streams`. `503` once the GPU stream scheduler has died (it does not recover — the Docker healthcheck restarts the container) |

### Clients

Python, OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")   # api_key = VIENEU_API_KEY if set
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

Browser (SSE):

```js
const r = await fetch("/v1/audio/speech", {method: "POST", headers: {"Content-Type": "application/json"},
  body: JSON.stringify({input: "Xin chào", response_format: "pcm", stream_format: "sse", sample_rate: 24000})});
for await (const line of readLines(r.body)) {           // split on "\n\n", strip "data: "
  const ev = JSON.parse(line);
  if (ev.type === "speech.audio.delta") playPcm(atob(ev.audio));
}
```

Pipecat / LiveKit Agents / Vercel AI SDK: use their OpenAI TTS plugin, change `base_url`, set `response_format="pcm"` and, if the framework assumes 24 kHz, add `sample_rate: 24000` as an extra body field (or set the framework's sample rate to 48,000).

`examples/openai_speech_client.py` is a sample client that reports TTFA/RTF and has a `--bench N` mode.

---

## Three numbers to understand: TTFA, RTF, lead

- **TTFA** (time to first audio): from sending the request to receiving the **first audio byte**. With `wav` the header arrives after ~2 ms but is not audio; every TTFA in this document counts real audio.
- **RTF** (real-time factor) = generation time / audio duration. RTF < 1 means audio is produced faster than it plays; a stream **never stalls** if RTF stays < 1 throughout. RTF 0.5 leaves half the time spare.
- **Lead**: audio received minus wall time elapsed since the first chunk. If the client plays as soon as the first chunk arrives, negative lead = a gap. On GPU the measured minimum lead is +80 ms at every load ≤ 16 streams; on CPU +320 ms (4-frame lead-in). Clients should **pre-buffer 150–300 ms** to absorb network jitter.

One codec frame = 80 ms of audio (12.5 frames/s). The first chunk holds 2 frames (GPU) or 4 (CPU); later chunks hold 4 frames = 320 ms.

---

## GPU: how streaming works

Code: `src/vieneu/v3_turbo_serve/stream.py` (+ `fused.py`). In short:

1. **One CUDA graph, B slots** (`max_streams`). Every frame (80 ms of audio) is **one graph replay** for all slots at once: the 16-codebook acoustic decoder + sampling + repetition penalty + one backbone step. 7 ms per frame at B = 1, 10 ms at B = 32 — nearly independent of the number of streams.
2. **Continuous batching**: a new request is prefilled (HF, ~21 ms for one prompt) and **written into a free slot of the running graph** (`StaticBackbone.load_row`: the KV cache is a ring buffer, the prompt lands right-aligned to the shared write index, rotary positions are per row). When a request finishes (EOS or its frame cap) its slot frees immediately, waiting for nobody. Sampling settings are per-row tensors, so each request keeps its own temperature/top-k/top-p/penalty.
3. **One shared streaming codec session** (MOSS): every 4 frames the codes of every running row go through **one** `batch_decode(streaming=True)` call; a new row is decoded early once it has 2 frames (lead-in) instead of waiting for the 4-frame boundary. The decoder has no lookahead; streamed audio matches the full decode (6e-4).
4. **One worker thread** owns the GPU. `infer_stream` from N threads (N HTTP requests) is just N audio queues; the server never forks.
5. **Captures run alone.** Recording a CUDA graph (this scheduler's, or a new one that the batch engine of `infer`/`infer_batch` needs) takes a process-wide lock (`_v3_turbo_engine/capture_lock.py`) that every other v3 Turbo inference step holds shared (not model loading, and not v1/v2 models in the same process). Another thread's allocation, sync or sampling during a capture can break it, and a broken capture leaves PyTorch's RNG stuck "capturing" (`Offset increment outside graph capture`) until the process restarts. The cost: each time a new graph is needed, playing streams pause for about the length of the capture (~0.5 s). Your own CUDA work in the same process (another model, `torch.cuda.empty_cache()`) is not covered: do not run it alongside VieNeu, or wrap it in `with CAPTURE_LOCK.shared():` (`from vieneu._v3_turbo_engine.capture_lock import CAPTURE_LOCK`, 3.8.4+; advanced: an internal path that may move in a later minor release). Put only your own CUDA calls inside that block: never call VieNeu (`infer`, `infer_batch`, `infer_stream`, `encode_reference`, `warm_fused`) or wait for another thread in it, or the process can hang. VieNeu calls that would wait for another thread raise `RuntimeError` there instead of hanging.

Real costs (RTX 3060, bf16):

| Component | Cost | Notes |
|---|---|---|
| Prefill (HF) | 21 ms (1 prompt) · 39 (8) · 73 (16) · 125 (32) | Once per request, between two frames |
| Graph frame | 7 ms (B=1) · 8 (8) · 9 (16) · 10 (32) | For the **whole** batch |
| **Codec** (4 frames) | **~65 ms + ~2.5 ms × reserved slots** | Independent of frame count and of rows actually in use; launch-bound |

The key point: **the codec is the whole cost**, and it scales with `max_streams` (slots the codec session reserves), not with the streams actually running. That is why `max_streams=32` slows down even a single stream. The codec's built-in `use_cuda_graph` mode is 3x cheaper but produces wrong audio (correlation 0.2 with the plain decode), so it is not used.

The old path (before 2026-09-15): single-sequence engine without CUDA graphs, TTFA 280–360 ms, a lock for the whole utterance → 1 stream, RTF 0.85.

---

## GPU: measurements on an RTX 3060

Test machine: RTX 3060 12 GB, 12th-gen i5 (6 P-cores), Windows 11, torch 2.8 + cu128, bf16, preset voice (69-frame reference), sentences of 88–147 characters, `apply_watermark=False` for in-process runs. 2026-09-15.

### Direct `infer_stream` (in process), N threads starting together

`max_streams = 16` (default):

| N streams at once | TTFA median | TTFA max | Worst RTF | Min lead |
|---|---|---|---|---|
| 1 | **115 ms** | 115 | 0.49 | +160 ms |
| 2 | 130 | 130 | 0.51 | +80 |
| 4 | 130 | 130 | 0.52 | +80 |
| 8 | 164 | 165 | 0.56 | +80 |
| 16 | 185 | 339 | **0.59** | +80 |

A new request arriving while others are playing (`max_streams=16`): among 3 streams **115 ms**, among 7 **145 ms**, among 15 **134 ms**.

`max_streams = 8`:

| N | TTFA median | Worst RTF | New request among N−1 |
|---|---|---|---|
| 1 | **89 ms** | 0.39 | — |
| 2 | 151 | 0.44 | — |
| 4 | 147 | 0.43 | 108 ms |
| 8 | 162 | 0.45 | 114 ms |

`max_streams = 32`:

| N | TTFA median | TTFA max | Worst RTF | Min lead | New request among N−1 |
|---|---|---|---|---|---|
| 1 | 141 ms | 141 | 0.70 | +160 | — |
| 2 | 165 | 165 | 0.71 | +80 | — |
| 4 | 174 | 174 | 0.73 | +80 | 274 ms |
| 8 | 207 | 208 | 0.77 | +80 | 180 ms |
| 16 | 220 | 389 | 0.82 | +66 | 227 ms |
| 24 | 448 | 453 | 0.86 | **−1 ms** | — |
| 32 | 456 | 629 | **0.93** | **−28 ms** | 181 ms |

VRAM (0.5 GB model included): `max_streams=8` 0.64 GB idle / 0.72 GB peak; 16: 0.91 / 1.10; 32: 1.45 / 1.82.

### Over HTTP (`apps/openai_speech.py`, client on the same machine, watermark on)

| Scenario | TTFA | RTF |
|---|---|---|
| 1 request, `pcm` 48 kHz | **106 ms** | 0.47 |
| 1 request, `pcm` + `sample_rate=24000` | 105 ms | 0.46 |
| 1 request, `sse` | 99 ms | — |
| 4 clients at once | median 160 ms | 0.51 |
| 8 clients at once | 207 ms | 0.53 |
| 16 clients at once | 197 ms (max 336) | 0.59 |
| 40 clients at once (16 slots + queue of 16) | 32 OK, 8 × `429`; the 16 queued ones see TTFA 1.5–3 s | — |

HTTP TTFA is within ~0–10 ms of the in-process number; over a real network add the RTT.

### A "cold" GPU after idling: the first request pays 100–300 ms extra

Every number above is with a **warm** GPU (a request within the last ~2 s). When the server idles, the NVIDIA driver parks the GPU in its power-saving state within ~2 s (RTX 3060: **P8, 210 MHz, 12 W**); the next request runs its prefill, 2 lead-in frames and the codec call at those clocks before the GPU ramps back to ~1950 MHz. Measured:

| Scenario | TTFA |
|---|---|
| 1 request, warm GPU | 118 ms |
| 1 request after 8 s idle | **403 ms** |
| 16 clients at once, warm GPU | median 194 ms, max 354 |
| 16 clients at once after 5–8 s idle | median 330–400 ms, max 450–540 |
| 16 clients right after *any* single request | median 179 ms |

Any request warms the GPU again; in a continuous conversation only the first sentence pays. Keeping it warm from inside the process **does not work**: a 2048² matmul or whole-graph replays every 100–250 ms still let the driver park the GPU (it wants sustained load). The fix belongs to the host:

- **Lock the clocks** (admin/root): `nvidia-smi -lgc 1500,2100` (undo: `nvidia-smi -rgc`); on Linux also `nvidia-smi -pm 1`. With Docker, run it on the host, not in the container.
- **Windows**: NVIDIA Control Panel → Manage 3D settings → *Power management mode* = **Prefer maximum performance** (set it per program for the venv's `python.exe`).
- Or accept it: only the first request after > 2 s of idle is affected.

Other cards behave the same (every NVIDIA GPU has P-states); the size of the penalty depends on the card's idle/boost clock gap.

### "N at once" vs. "a new request among N−1"

"N at once" is the worst case: N prefills land in the same few ticks (at most 8 admitted per tick) and N lead-in codec calls pile up. Real traffic arrives spread out, which is the "new request among N−1" column: ~135 ms at 16 streams.

---

## GPU: choosing `max_streams`, streams vs. users

| Goal | `max_streams` | On a 3060 |
|---|---|---|
| Lowest latency, few users | 8 | 1 stream 89 ms, 8 streams 162 ms, GPU 45 % busy |
| **Balanced (default)** | **16** | 1 stream 115 ms, 16 streams ≤ 200 ms, GPU 59 % busy |
| Maximum throughput | 32 | 32 streams do not stall (RTF 0.93) but first audio takes 450 ms and there is no margin; 24 is tolerable |

Rule: `max_streams` = the highest number of **simultaneous** streams you need, no more — every spare slot adds ~2.5 ms to **every** codec call for **everyone**.

**Streams ≠ users.** A stream exists only while one reply plays (a few seconds). For a voice chatbot each user holds a stream ~20–35 % of the time → 16 streams ≈ **45–80 active users**; for continuous reading (news, books) 16 streams = 16 users. Rule of thumb: `users ≈ max_streams / fraction of time spent listening to TTS`.

---

## GPU: estimates for other machines

**Nothing but the RTX 3060 has been measured** — the table below is reasoning from the cost structure, meant for planning and to be verified with [Measure it on your machine](#measure-it-on-your-machine).

Three facts from the measurements drive the estimates:

1. **VRAM is not the limit.** 16 streams peak at 1.1 GB, 32 at 1.8 GB. A 4 GB card fits 16 streams, 6 GB fits 32.
2. **The limit is per-call overhead (launch-bound), not FLOPS.** The 12-layer, 768-wide backbone and the 11M-parameter codec are tiny; the time goes to hundreds of small kernel launches. A much bigger GPU (4090, A100) therefore does **not** cut TTFA much; a fast host CPU, the driver, PCIe and not sharing the GPU matter as much as the GPU itself.
3. **GPU generation matters through dtype and CUDA graphs.** Ampere and newer (RTX 30/40, A-series) run bf16 like the test machine. Turing/Pascal (GTX 16, RTX 20, T4, GTX 10) lack bf16 → the code falls back to **fp16**; that path is **unverified for quality** (activation overflow is possible). CUDA graphs need a CUDA ≥ 11 driver, available on every card since Maxwell.

| Machine | VRAM | Est. TTFA, 1 stream | Est. real-time streams | Notes |
|---|---|---|---|---|
| RTX 3060 12 GB (measured) | 12 | 105–115 ms | 16 comfortably, 32 max | bf16 |
| RTX 3050 / 4060 8 GB, RTX 3060 Ti / 4070 | 6–12 | 100–130 ms | 16, up to ~32 | Same generation and launch-bound, so about equal to the 3060; a power-limited laptop 4060 may be 10–20 % slower |
| RTX 3050 laptop 4 GB, RTX A2000 | 4 | 120–160 ms | 8–12 (`max_streams=8`–`12`) | VRAM suffices; low clocks and sharing with the OS → keep `max_streams` small |
| RTX 4090 / A10 / L4 | 16–24 | 90–110 ms | 24–32 | Codec part faster, launch part not; the real ceiling is the host CPU |
| T4 16 GB (cloud) | 16 | 150–220 ms | 8–12 | Turing → fp16 (unverified); low clocks; cloud VMs often have weak CPUs → more launch overhead |
| GTX 1660 / RTX 2060 6 GB | 6 | 140–200 ms | 8–12 | As T4, with a faster desktop CPU |
| GTX 1650 4 GB, GTX 1050 Ti | 4 | 180–250 ms | 4–8 | fp16 (Pascal has no tensor cores: fp32 codec ~1.5–2× slower) |
| GPU < 4 GB | <4 | — | 2–4, or use the CPU | Needs `max_streams` ≤ 4 for the static cache + codec; no clear win over CPU int8 |

How to read it: TTFA ≈ prefill (20–40 ms) + 2 graph frames (15–25 ms) + one codec call (60–120 ms depending on card/host). Streams ≈ (80 ms − graph frame) / (codec cost ÷ 4 frames), keeping RTF ≤ 0.6–0.7 for margin.

On 4–6 GB cards set `VIENEU_MAX_STREAMS=8` (or 12) from the start: cheaper codec calls and peak VRAM under 1 GB.

---

## CPU: how streaming works, measurements

Code: `src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py` (`infer_stream`). A CPU-only machine runs **without torch**: backbone and acoustic decoder are ONNX, the codec is `moss_audio_tokenizer_decode_step.onnx` — a streaming decoder that is **bit-exact** with the full one (no lookahead, no clicks at chunk edges).

Mechanism:

1. Prefill the prompt (reference codes + phonemes) once; then per frame: the 16-codebook acoustic decoder (16 small ONNX calls) + one backbone step with a KV cache.
2. **4-frame lead-in** (320 ms of audio) before the first chunk is decoded — 320 ms of buffer so the client survives fluctuations in generation speed.
3. Then **adaptive pacing** (`_target_frames`): if emitted audio exceeds elapsed time by < 0.2 s, decode every 4 frames; < 0.55 s → 6; < 1.1 s → 8; with a comfortable lead up to 25 frames per call (fewer codec calls, better RTF).
4. ORT `intra_op_threads` = physical cores (max 8, `Vieneu(threads=...)`), `inter_op = 1`, no spinning — so an idle server does not burn CPU.
5. A **per-frame** lock (RLock): two simultaneous requests **interleave** rather than fully queue — but they share the cores and each slows down accordingly.

Measurements (12th-gen i5, 6 P-cores, 6 ORT threads; sentences of 27 / 88 / 147 characters):

| Precision | TTFA | RTF, 1 stream | Min lead | 2 requests at once |
|---|---|---|---|---|
| **fp32** (default) | 264–282 · 313–357 · 345–401 ms | 0.61 · 0.57 · 0.55 | +320 ms | TTFA 399 / 564 ms, **RTF 1.19 / 1.19 → both stall** |
| **int8** | 141–144 · 163–169 · 192–194 ms | 0.35 · 0.35 · 0.35 | +320 ms | TTFA 224 / 336 ms, RTF 0.58 / 0.67 → still real-time |

Conclusions for CPU:

- **fp32: exactly one stream.** The server defaults to `max_streams=1`; a second request waits in the queue (`VIENEU_QUEUE`, default = `max_streams`; the docker `api-cpu` profile sets 4) for up to `VIENEU_QUEUE_TIMEOUT` seconds, then gets `429`. The wait equals the remaining generation of the sentence playing (a 7 s sentence generates in ~4 s).
- **int8: two streams** (default `max_streams=2`) on 6 cores at RTF ~0.65 each; a third would hit 1.0. 8–12 physical cores may sustain 3–4 — measure before trusting it.
- TTFA depends on the **length of the first sentence** (longer prefill) and the **core count**: a 4-core laptop will see fp32 ~400–600 ms, int8 ~200–300 ms; on 2–4 cores fp32 RTF can exceed 1 → use int8 only.
- int8 **needs a CPU with VNNI** (Intel from Cascade Lake / Ice Lake / Alder Lake, AMD Zen 4). Without VNNI the int8 acoustic decoder saturates and babbles — use fp32.
- Clients should **pre-buffer ≥ 300 ms** — exactly the lead-in — because every other process on the machine competes for the CPU.
- There is no batching scheduler on CPU: more requests means splitting cores, nothing comes "for free" as on the GPU.

For more streams without a GPU: run several containers pinned to core groups (`docker --cpuset-cpus`), or accept `429` and let clients retry.

---

## Measure it on your machine

```bash
uv run python -m apps.openai_speech &                    # or the docker profiles api-gpu / api-cpu
uv run python examples/openai_speech_client.py           # 1 request: TTFA, RTF, out_stream.wav
uv run python examples/openai_speech_client.py --bench 8 # 8 clients at once
uv run python examples/openai_speech_client.py --bench 16
```

Reading the output:

- `RTF max` must be **< 1**, ideally ≤ 0.7. If `--bench N` shows RTF > 0.8, N exceeds the machine → lower `VIENEU_MAX_STREAMS`.
- A `TTFA max` far above the median (339 vs 185 at 16 streams) is prefill pile-up; spread-out traffic sits near the median.
- The server logs every request: `ttfa=… total=… audio=… rtf=… active=…`.
- GPU: watch `nvidia-smi` during the bench; `torch.cuda.max_memory_allocated()` in-process is more precise.

For component-level numbers (prefill / frame / codec) see the docstring of `src/vieneu/v3_turbo_serve/stream.py` and the tests in `tests/fused/cases.py`.

---

## Tuning and troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Slow first chunk even with few users (GPU) | `max_streams` too large → codec pads many slots | Lower `VIENEU_MAX_STREAMS` to the real load (8 for ≤ 8 streams) |
| First request after a few idle seconds is 100–300 ms slower (GPU) | GPU parked at low clocks (P8) | Lock clocks with `nvidia-smi -lgc` or "Prefer maximum performance" — see *A "cold" GPU* |
| Frequent `429` | Slots and queue exhausted | Raise `VIENEU_QUEUE` / `VIENEU_QUEUE_TIMEOUT` if waiting is acceptable; otherwise add a GPU/container |
| Choppy audio at the client | No pre-buffer, or RTF > 1 (CPU fp32 with several requests) | Pre-buffer 150–300 ms (GPU) / ≥ 300 ms (CPU); on CPU use int8 and keep 1–2 streams |
| First request after start-up takes 1–2 s | Warm-up not finished | The server warms itself; wait for `/health` to return `ok` before sending traffic |
| Every request fails with `Offset increment outside graph capture encountered unexpectedly` (GPU) | A CUDA graph capture broke (SDK < 3.8.4, or other CUDA work ran beside a capture) | Restart the process; upgrade to `vieneu>=3.8.4`; wrap only your own CUDA calls in `CAPTURE_LOCK.shared()` (item 5 of *GPU: how streaming works*); last resort `VIENEU_FUSED_FRAME=0` |
| `RuntimeError: ... while this thread holds CAPTURE_LOCK` | VieNeu was called inside your own `CAPTURE_LOCK.shared()` block | Keep only your own CUDA calls in the block; call VieNeu outside it |
| Disable the CUDA-graph path (debugging) | — | `VIENEU_FUSED_FRAME=0` → `infer_stream` falls back to the single engine (1 stream, ~300 ms) |
| Turing/Pascal GPU, odd audio | fp16 fallback is unverified | Try `Vieneu(dtype="float32")` (~1.5× slower) or open an issue with a sample |
| Need mp3/opus | Not supported | Put a reverse proxy/ffmpeg in front, or take `pcm` and encode client-side; adds ~20–40 ms |
| `repetition_window` other than the default has no effect (GPU) | The penalty window is the scheduler's fixed ring size | Tune `repetition_penalty` instead; the default window is shared |
| Does a dropped connection keep a slot? | No | Closing the response closes the generator → the GPU slot frees at the next tick (verified) |
