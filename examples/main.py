"""
VieNeu-TTS SDK Example: Standard Mode (Local Inference)

This example demonstrates how to run VieNeu-TTS locally on your machine.
Ideal for offline apps, local development, or private deployments.
"""

import os

from vieneu import Vieneu


def main():
    print("🚀 Initializing local VieNeu engine...")

    os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------------------------
    # PART 1: INITIALIZATION
    # ---------------------------------------------------------
    # Mode="standard" (default) runs locally.
    # By default, it uses "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf" (Backbone)
    # and "neuphonic/distill-neucodec" (Codec) for maximum speed.
    tts = Vieneu()

    # Optional: If you want to force use a specific PyTorch model:
    # tts = Vieneu(backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B", codec_repo="neuphonic/distill-neucodec", backbone_device="cuda", codec_device="cuda")

    # ---------------------------------------------------------
    # PART 2: LIST PRESET VOICES
    # ---------------------------------------------------------
    # The SDK returns (Description, ID) tuples
    available_voices = tts.list_preset_voices()
    print(f"📋 Found {len(available_voices)} preset voices.")

    if available_voices:
        print("   Showing all voices:")
        for desc, name in available_voices:
            print(f"   - {desc} (ID: {name})")

    # ---------------------------------------------------------
    # PART 3: USE SPECIFIC VOICE ID
    # ---------------------------------------------------------
    if available_voices:
        print("\n--- PART 3: Using Specific Voice ID ---")
        # Example: Select Tuyên (nam miền Bắc) - usually ID is 'Tuyen'
        _, my_voice_id = available_voices[1] if len(available_voices) > 1 else available_voices[0]
        print(f"👤 Selecting voice: {my_voice_id}")

        # Get reference data for this specific voice
        voice_data = tts.get_preset_voice(my_voice_id)

        test_text = "Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên."
        audio_spec = tts.infer(text=test_text, voice=voice_data)

        tts.save(audio_spec, f"outputs/standard_{my_voice_id}.wav")
        print(f"💾 Saved {my_voice_id} synthesis to: outputs/standard_{my_voice_id}.wav")

    # ---------------------------------------------------------
    # PART 4: STANDARD SPEECH SYNTHESIS (DEFAULT)
    # ---------------------------------------------------------
    print("\n--- PART 4: Standard Synthesis (Default) ---")
    text = "Xin chào, tôi là VieNeu. Tôi có thể giúp bạn đọc sách, làm chatbot thời gian thực, hoặc thậm chí clone giọng nói của bạn."

    print("🎧 Synthesizing speech...")
    # By default, it uses the model's 'default_voice'
    audio = tts.infer(text=text)
    tts.save(audio, "outputs/standard_output.wav")
    print("💾 Saved synthesized speech to: outputs/standard_output.wav")

    # ---------------------------------------------------------
    # PART 5: ZERO-SHOT VOICE CLONING (LOCAL)
    # ---------------------------------------------------------
    # You can clone any voice using a short audio sample (3-5s) and its transcript
    ref_audio = "examples/audio_ref/example_ngoc_huyen.wav"
    ref_text = "Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."

    if os.path.exists(ref_audio):
        print("\n--- PART 5: Voice Cloning ---")
        print(f"🦜 Cloning voice from: {ref_audio}")
        cloned_audio = tts.infer(
            text="Đây là giọng nói đã được clone thành công từ file mẫu.",
            ref_audio=ref_audio,
            ref_text=ref_text
        )
        tts.save(cloned_audio, "outputs/standard_cloned_output.wav")
        print("💾 Saved cloned voice to: outputs/standard_cloned_output.wav")

    # ---------------------------------------------------------
    # PART 6: CLEANUP
    # ---------------------------------------------------------
    # Explicitly release resources
    tts.close()
    print("\n✅ All tasks completed!")

if __name__ == "__main__":
    main()
