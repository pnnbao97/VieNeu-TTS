import gc
import os
import queue
import sys
import tempfile
import threading
import time
from functools import lru_cache

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from app_config import (
    BACKBONE_CONFIGS,
    CODEC_CONFIGS,
    GGUF_ALLOWED_VOICES,
    MAX_CHARS_PER_CHUNK,
    SPEAKER_MODE_TO_KEY,
    VOICE_SAMPLES,
)
from utils.core_utils import split_text_into_chunks
from vieneu_tts import FastVieNeuTTS, VieNeuTTS

# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False


@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]

    if sys.platform == "darwin":
        if torch.backends.mps.is_available():
            devices.append("MPS")
    else:
        if torch.cuda.is_available():
            devices.append("CUDA")

    return devices


def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    if sys.platform == "darwin":
        return False

    if "gguf" in backbone_choice.lower():
        return False

    if device_choice == "Auto":
        has_gpu = torch.cuda.is_available()
    elif device_choice == "CUDA":
        has_gpu = torch.cuda.is_available()
    else:
        has_gpu = False

    return has_gpu


def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def load_model(backbone_choice: str, codec_choice: str, device_choice: str,
               enable_triton: bool, max_batch_size: int):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None

    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False),
        gr.update(interactive=False)
    )

    try:
        if model_loaded and tts is not None:
            del tts
            cleanup_gpu_memory()

        backbone_config = BACKBONE_CONFIGS[backbone_choice]
        codec_config = CODEC_CONFIGS[codec_choice]

        use_lmdeploy = should_use_lmdeploy(backbone_choice, device_choice)

        if use_lmdeploy:
            lmdeploy_error_reason = None
            print("🚀 Using LMDeploy backend with optimizations")

            backbone_device = "cuda"

            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cuda" if torch.cuda.is_available() else "cpu"

            print("🚀 Loading optimized model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: {'Enabled' if enable_triton else 'Disabled'}")
            print(f"   Max Batch Size: {max_batch_size}")

            try:
                tts = FastVieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=True,
                    enable_triton=enable_triton,
                    max_batch_size=max_batch_size,
                )
                using_lmdeploy = True

                print("🚀 Pre-caching voice references...")
                for voice_name, voice_info in VOICE_SAMPLES.items():
                    audio_path = voice_info["audio"]
                    text_path = voice_info["text"]
                    if os.path.exists(audio_path) and os.path.exists(text_path):
                        ref_text = get_ref_text_cached(text_path)
                        tts.get_cached_reference(voice_name, audio_path, ref_text)
                print(f"   ✅ Cached {len(VOICE_SAMPLES)} voices")

            except Exception as e:
                import traceback
                traceback.print_exc()

                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"

                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False

        if not use_lmdeploy:
            print("🚀 Using original backend")

            if device_choice == "Auto":
                if "gguf" in backbone_choice.lower():
                    if sys.platform == "darwin":
                        backbone_device = "gpu"
                    else:
                        backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                else:
                    if sys.platform == "darwin":
                        backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                    else:
                        backbone_device = "cuda" if torch.cuda.is_available() else "cpu"

                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                elif sys.platform == "darwin":
                    codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                else:
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_choice.lower() and backbone_device == "cuda":
                backbone_device = "gpu"

            print("🚀 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")

            tts = VieNeuTTS(
                backbone_repo=backbone_config["repo"],
                backbone_device=backbone_device,
                codec_repo=codec_config["repo"],
                codec_device=codec_device
            )
            using_lmdeploy = False

        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True

        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "🚀 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")

        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""

        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                "\n\n⚡ Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • Max Batch Size: {max_batch_size}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                "\n  • Prefix Caching: ✅"
            )

        warning_msg = ""
        if lmdeploy_error_reason:
            warning_msg = (
                "\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy (Optimized Backend) do lỗi sau:\n"
                f"⚠️ {lmdeploy_error_reason}\n"
                "⚠️ Hệ thống đã tự động chuyển về chế độ Standard (chậm hơn)."
            )

        success_msg = (
            "✅ Model đã tải thành công!\n\n"
            f"🚀 Backend: {backend_name}\n"
            f"🦜 Model Device: {device_info.upper()}\n"
            f"🎛 Codec Device: {codec_device.upper()}{preencoded_note}\n"
            f"🌊 Streaming: {streaming_support}{opt_info}{warning_msg}"
        )

        yield (
            success_msg,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        using_lmdeploy = False

        if "$env:CUDA_PATH" in str(e):
            yield (
                "❌ Lỗi khi tải model: Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit (https://developer.nvidia.com/cuda/toolkit)",
                gr.update(interactive=False),
                gr.update(interactive=True)
            )
        else:
            yield (
                f"❌ Lỗi khi tải model: {str(e)}",
                gr.update(interactive=False),
                gr.update(interactive=True)
            )


def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str,
                     mode_tab: str, generation_mode: str, use_batch: bool):
    """Synthesis with optimization support and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy

    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return

    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return

    raw_text = text.strip()

    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']

    if mode_tab == "custom_mode":
        if custom_audio is None or not custom_text:
            yield None, "⚠️ Thiếu Audio hoặc Text mẫu custom."
            return
        ref_audio_path = custom_audio
        ref_text_raw = custom_text
        ref_codes_path = None
    else:
        if voice_choice not in VOICE_SAMPLES:
            yield None, "⚠️ Vui lòng chọn giọng mẫu."
            return
        ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        text_path = VOICE_SAMPLES[voice_choice]["text"]
        ref_codes_path = VOICE_SAMPLES[voice_choice]["codes"]

        if not os.path.exists(ref_audio_path):
            yield None, "❌ Không tìm thấy file audio mẫu."
            return

        ref_text_raw = get_ref_text_cached(text_path)

    yield None, "🔄 Đang xử lý Reference..."

    try:
        if use_preencoded and ref_codes_path and os.path.exists(ref_codes_path):
            ref_codes = torch.load(ref_codes_path, map_location="cpu", weights_only=True)
        else:
            if using_lmdeploy and hasattr(tts, 'get_cached_reference') and mode_tab == "preset_mode":
                ref_codes = tts.get_cached_reference(voice_choice, ref_audio_path, ref_text_raw)
            else:
                ref_codes = tts.encode_reference(ref_audio_path)

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
    except Exception as e:
        yield None, f"❌ Lỗi xử lý reference: {e}"
        return

    text_chunks = split_text_into_chunks(raw_text, max_chars=MAX_CHARS_PER_CHUNK)
    total_chunks = len(text_chunks)

    if generation_mode == "Standard (Một lần)":
        backend_name = "LMDeploy" if using_lmdeploy else "Standard"
        batch_info = " (Batch Mode)" if use_batch and using_lmdeploy and total_chunks > 1 else ""

        batch_size_info = ""
        if use_batch and using_lmdeploy and hasattr(tts, 'max_batch_size'):
            batch_size_info = f" [Max batch: {tts.max_batch_size}]"

        yield None, f"🚀 Bắt đầu tổng hợp {backend_name}{batch_info}{batch_size_info} ({total_chunks} đoạn)..."

        all_audio_segments = []
        sr = 24000
        silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)

        start_time = time.time()

        try:
            if use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                batch_size = tts.max_batch_size if hasattr(tts, 'max_batch_size') else 8
                num_batches = (total_chunks + batch_size - 1) // batch_size

                yield None, f"⚡ Xử lý {num_batches} mini-batch(es) (max {batch_size} đoạn/batch)..."

                chunk_wavs = tts.infer_batch(text_chunks, ref_codes, ref_text_raw)

                for i, chunk_wav in enumerate(chunk_wavs):
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_audio_segments.append(chunk_wav)
                        if i < total_chunks - 1:
                            all_audio_segments.append(silence_pad)
            else:
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."

                    chunk_wav = tts.infer(chunk, ref_codes, ref_text_raw)

                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_audio_segments.append(chunk_wav)
                        if i < total_chunks - 1:
                            all_audio_segments.append(silence_pad)

            if not all_audio_segments:
                yield None, "❌ Không sinh được audio nào."
                return

            yield None, "🔄 Đang ghép file và lưu..."

            final_wav = np.concatenate(all_audio_segments)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name

            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard ⚡'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""

            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"

            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            cleanup_gpu_memory()

        except torch.cuda.OutOfMemoryError as e:
            cleanup_gpu_memory()
            yield None, (
                "❌ GPU hết VRAM! Hãy thử:\n"
                f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                "• Giảm độ dài văn bản\n\n"
                f"Chi tiết: {str(e)}"
            )
            return

        except Exception as e:
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return

    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3

        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""

        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None

                for chunk_text in text_chunks:
                    stream_gen = tts.infer_stream(chunk_text, ref_codes, ref_text_raw)

                    for audio_part in stream_gen:
                        if audio_part is None or len(audio_part) == 0:
                            continue

                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

                                blended = (audio_part[:overlap] * fade_in +
                                           previous_tail[-overlap:] * fade_out)

                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])

                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part

                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))

                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)

        threading.Thread(target=producer_thread, daemon=True).start()

        yield (sr, np.zeros(int(sr * 0.05))), "🔄 Đang buffering..."

        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break

        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "⚡ Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."

        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue

        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"

            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            cleanup_gpu_memory()


def _parse_line_order(line_order, line_count):
    order = []
    if isinstance(line_order, str):
        for part in line_order.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= line_count and idx not in order:
                    order.append(idx)
    if not order:
        order = list(range(1, line_count + 1))
    else:
        for idx in range(1, line_count + 1):
            if idx not in order:
                order.append(idx)
    return order


def synthesize_multi_voice(line_count, line_order, use_batch, *line_inputs):
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy

    if not model_loaded or tts is None:
        yield None, "Vui lòng tải model trước."
        return

    try:
        line_count = int(line_count)
    except (TypeError, ValueError):
        line_count = 0

    if line_count <= 0:
        yield None, "Vui lòng thêm người nói."
        return

    line_fields = 5
    segments = []
    order = _parse_line_order(line_order, line_count)

    for line_idx in order:
        base = (line_idx - 1) * line_fields
        if base + line_fields > len(line_inputs):
            continue
        mode_label = line_inputs[base]
        voice_choice = line_inputs[base + 1]
        custom_audio = line_inputs[base + 2]
        custom_text = line_inputs[base + 3]
        line_text = line_inputs[base + 4]

        if not line_text or not str(line_text).strip():
            continue

        segments.append({
            "mode": SPEAKER_MODE_TO_KEY.get(mode_label, "preset"),
            "voice_choice": voice_choice,
            "custom_audio": custom_audio,
            "custom_text": custom_text,
            "text": str(line_text).strip(),
        })

    if not segments:
        yield None, "Vui lòng nhập nội dung hội thoại."
        return

    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config["use_preencoded"]

    yield None, "Đang chuẩn bị mẫu tham chiếu..."

    ref_cache = {}
    try:
        for segment in segments:
            mode_key = segment["mode"]
            voice_choice = segment["voice_choice"]
            custom_audio = segment["custom_audio"]
            custom_text = segment["custom_text"] or ""

            if mode_key == "preset":
                cache_key = ("preset", voice_choice)
            else:
                cache_key = ("custom", custom_audio, custom_text)

            if cache_key in ref_cache:
                continue

            if mode_key == "preset":
                if voice_choice not in VOICE_SAMPLES:
                    yield None, f"Giọng mẫu không hợp lệ: {voice_choice}"
                    return
                if "gguf" in current_backbone.lower() and voice_choice not in GGUF_ALLOWED_VOICES:
                    yield None, f"Giọng không hỗ trợ cho GGUF: {voice_choice}"
                    return

                ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
                text_path = VOICE_SAMPLES[voice_choice]["text"]
                ref_codes_path = VOICE_SAMPLES[voice_choice]["codes"]

                if not os.path.exists(ref_audio_path):
                    yield None, f"Thiếu audio giọng mẫu: {voice_choice}"
                    return

                ref_text_raw = get_ref_text_cached(text_path)
            else:
                ref_audio_path = custom_audio
                ref_text_raw = custom_text
                ref_codes_path = None

                if not ref_audio_path or not ref_text_raw:
                    yield None, "Thiếu audio hoặc văn bản cho giọng tùy chỉnh."
                    return
                if not os.path.exists(ref_audio_path):
                    yield None, "Không tìm thấy file audio giọng tùy chỉnh."
                    return

            if use_preencoded and ref_codes_path and os.path.exists(ref_codes_path):
                ref_codes = torch.load(ref_codes_path, map_location="cpu", weights_only=True)
            else:
                if mode_key == "preset" and using_lmdeploy and hasattr(tts, "get_cached_reference"):
                    ref_codes = tts.get_cached_reference(voice_choice, ref_audio_path, ref_text_raw)
                else:
                    ref_codes = tts.encode_reference(ref_audio_path)

            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

            ref_cache[cache_key] = {
                "codes": ref_codes,
                "ref_text": ref_text_raw,
            }
    except Exception as e:
        yield None, f"Lỗi tham chiếu: {e}"
        return

    total_segments = len(segments)
    sr = 24000
    chunk_silence = np.zeros(int(sr * 0.15), dtype=np.float32)
    segment_silence = np.zeros(int(sr * 0.2), dtype=np.float32)

    all_audio_segments = []
    start_time = time.time()

    try:
        for idx, segment in enumerate(segments):
            mode_key = segment["mode"]
            text = segment["text"]
            if mode_key == "preset":
                cache_key = ("preset", segment["voice_choice"])
            else:
                cache_key = ("custom", segment["custom_audio"], segment["custom_text"] or "")

            ref_codes = ref_cache[cache_key]["codes"]
            ref_text_raw = ref_cache[cache_key]["ref_text"]

            text_chunks = split_text_into_chunks(text, max_chars=MAX_CHARS_PER_CHUNK)
            yield None, f"Đang xử lý {idx+1}/{total_segments} ({len(text_chunks)} đoạn)..."

            segment_audio_parts = []

            if use_batch and using_lmdeploy and hasattr(tts, "infer_batch") and len(text_chunks) > 1:
                chunk_wavs = tts.infer_batch(text_chunks, ref_codes, ref_text_raw)
                for i, chunk_wav in enumerate(chunk_wavs):
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        segment_audio_parts.append(chunk_wav)
                        if i < len(text_chunks) - 1:
                            segment_audio_parts.append(chunk_silence)
            else:
                for i, chunk in enumerate(text_chunks):
                    chunk_wav = tts.infer(chunk, ref_codes, ref_text_raw)
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        segment_audio_parts.append(chunk_wav)
                        if i < len(text_chunks) - 1:
                            segment_audio_parts.append(chunk_silence)

            if not segment_audio_parts:
                continue

            if all_audio_segments:
                all_audio_segments.append(segment_silence)
            all_audio_segments.append(np.concatenate(segment_audio_parts))

        if not all_audio_segments:
            yield None, "Không sinh được audio."
            return

        yield None, "Đang hoàn tất audio..."
        final_wav = np.concatenate(all_audio_segments)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            output_path = tmp.name

        process_time = time.time() - start_time
        backend_info = f" (Backend: {'LMDeploy' if using_lmdeploy else 'Standard'})"
        speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""

        yield output_path, f"Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"

        if using_lmdeploy and hasattr(tts, "cleanup_memory"):
            tts.cleanup_memory()
        cleanup_gpu_memory()

    except torch.cuda.OutOfMemoryError as e:
        cleanup_gpu_memory()
        yield None, (
            "GPU hết VRAM. Hãy giảm max batch size hoặc độ dài văn bản.\n\n"
            f"Chi tiết: {str(e)}"
        )
        return
    except Exception as e:
        import traceback
        traceback.print_exc()
        cleanup_gpu_memory()
        yield None, f"Lỗi đa giọng: {str(e)}"
        return


def synthesize_router(text, voice_choice, custom_audio, custom_text,
                      mode_tab, generation_mode, use_batch,
                      multi_line_count, multi_line_order, *multi_line_inputs):
    if mode_tab == "multi_mode":
        yield from synthesize_multi_voice(multi_line_count, multi_line_order, use_batch, *multi_line_inputs)
    else:
        yield from synthesize_speech(text, voice_choice, custom_audio, custom_text, mode_tab, generation_mode, use_batch)
