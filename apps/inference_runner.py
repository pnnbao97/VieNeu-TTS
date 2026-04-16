import time
import queue
import threading
import tempfile
import numpy as np
import soundfile as sf
import sys
from vieneu_utils.core_utils import join_audio_chunks, get_silence_duration_v2, split_text_into_chunks, split_into_chunks_v2
from vieneu_utils.phonemize_text import phonemize_with_dict
import apps.model_manager as model_manager

def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str,
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int, _text_normalizer, is_xpu: bool = False):
    """Synthesis with optimization support and max batch size control"""

    if not model_manager.model_loaded or model_manager.tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return

    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return

    raw_text = text.strip()

    # Setup Reference
    yield None, "📄 Đang xử lý Reference..."

    try:
        ref_codes = None
        ref_text_raw = ""

        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu.")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng mẫu khả dụng. Vui lòng chuyển sang Tab Voice Cloning.")

            voice_data = model_manager.tts.get_preset_voice(voice_choice)
            ref_codes = voice_data['codes']
            ref_text_raw = voice_data['text']

        elif mode_tab == "custom_mode":
            if custom_audio is None:
                raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")

            is_turbo = "v2-Turbo" in (model_manager.current_backbone or "")
            if not is_turbo and (not custom_text or not custom_text.strip()):
                raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")

            ref_text_raw = custom_text.strip() if custom_text else ""
            ref_codes = model_manager.tts.encode_reference(custom_audio)

        if 'torch' in sys.modules:
            import torch
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return

    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        if is_xpu:
            backend_name = "Intel XPU"
        else:
            backend_name = "LMDeploy" if model_manager.using_lmdeploy else "Standard"
        normalized_text = _text_normalizer.normalize(raw_text)
        is_v2_turbo = "v2-Turbo" in (model_manager.current_backbone or "")

        if is_v2_turbo:
            phonemes = phonemize_with_dict(normalized_text, skip_normalize=True)
            text_chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars_chunk)
        else:
            text_chunks = split_text_into_chunks(normalized_text, max_chars=max_chars_chunk)

        total_chunks = len(text_chunks)
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name} ({total_chunks} đoạn)..."

        all_wavs = []
        sr = 24000
        start_time = time.time()

        try:
            if is_xpu:
                if use_batch and total_chunks > 1:
                    for i in range(0, total_chunks, max_batch_size_run):
                        yield None, f"⏳ Đang xử lý batch {i//max_batch_size_run + 1} ..."
                        batch_chunks = text_chunks[i : i + max_batch_size_run]
                        batch_results = model_manager.tts.infer_batch(
                            texts=batch_chunks, ref_codes=ref_codes, ref_text=ref_text_raw,
                            temperature=temperature, skip_normalize=True
                        )
                        if batch_results:
                            all_wavs.extend(batch_results)
                else:
                    for i, chunk in enumerate(text_chunks):
                        yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks} (XPU Bfloat16)..."
                        chunk_wav = model_manager.tts.infer(
                            chunk, ref_codes=ref_codes, ref_text=ref_text_raw,
                            temperature=temperature, skip_normalize=True
                        )
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_wavs.append(chunk_wav)
            elif is_v2_turbo:
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⚡ Turbo v2: Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wav = model_manager.tts.infer(
                        chunk.text, ref_codes=ref_codes, temperature=temperature,
                        max_chars=max_chars_chunk, skip_normalize=True, skip_phonemize=True
                    )
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
                        if i < total_chunks - 1:
                            sil_dur = get_silence_duration_v2(chunk)
                            sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                            all_wavs.append(sil_wav)

            elif use_batch and model_manager.using_lmdeploy and hasattr(model_manager.tts, 'infer_batch') and total_chunks > 1:
                chunk_wavs = model_manager.tts.infer_batch(
                    text_chunks, ref_codes=ref_codes, ref_text=ref_text_raw,
                    max_batch_size=max_batch_size_run, temperature=temperature, skip_normalize=True
                )
                for chunk_wav in chunk_wavs:
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
            else:
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wav = model_manager.tts.infer(
                        chunk, ref_codes=ref_codes, ref_text=ref_text_raw,
                        temperature=temperature, max_chars=max_chars_chunk, skip_normalize=True
                    )
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)

            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return

            yield None, "💾 Đang ghép file và lưu..."
            silence_p = 0.15 if not is_v2_turbo else 0.0
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=silence_p)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name

            process_time = time.time() - start_time
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            backend_note = f" (Backend: {backend_name})"
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_note}"

            if model_manager.using_lmdeploy and hasattr(model_manager.tts, 'cleanup_memory'):
                model_manager.tts.cleanup_memory()
            model_manager.cleanup_gpu_memory()

        except Exception as e:
            model_manager.cleanup_gpu_memory()
            yield None, f"❌ Lỗi: {str(e)}"
            return

    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3

        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""

        normalized_text = _text_normalizer.normalize(raw_text)
        is_v2_turbo = "v2-Turbo" in (model_manager.current_backbone or "")
        if is_v2_turbo:
            phonemes = phonemize_with_dict(normalized_text, skip_normalize=True)
            text_chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars_chunk)
        else:
            text_chunks = split_text_into_chunks(normalized_text, max_chars=max_chars_chunk)

        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                for i, chunk_text in enumerate(text_chunks):
                    if is_v2_turbo:
                        stream_gen = model_manager.tts.infer_stream(
                            chunk_text, ref_codes=ref_codes, temperature=temperature,
                            max_chars=max_chars_chunk, skip_normalize=True, skip_phonemize=True
                        )
                    else:
                        stream_gen = model_manager.tts.infer_stream(
                            chunk_text, ref_codes=ref_codes, ref_text=ref_text_raw,
                            temperature=temperature, max_chars=max_chars_chunk, skip_normalize=True
                        )

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
                                    blended, audio_part[overlap:]
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

                    if is_v2_turbo and i < len(text_chunks) - 1:
                        sil_dur = get_silence_duration_v2(chunk_text)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        audio_queue.put((sr, sil_wav))

                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))

            except Exception as e:
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)

        threading.Thread(target=producer_thread, daemon=True).start()
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."

        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None: break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break

        full_audio_buffer = []
        if is_xpu:
            backend_info = "Intel XPU"
        else:
            backend_info = "🚀 LMDeploy" if model_manager.using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."

        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None: break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty(): break
                continue

        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"
            if model_manager.using_lmdeploy and hasattr(model_manager.tts, 'cleanup_memory'):
                model_manager.tts.cleanup_memory()
            model_manager.cleanup_gpu_memory()
