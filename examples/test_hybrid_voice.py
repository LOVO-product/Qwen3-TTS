#!/usr/bin/env python3
"""
Test hybrid voice generation: voice cloning + instruction control.

This demonstrates how to clone a voice from reference audio while also
applying style/emotion instructions.

Usage:
    python examples/test_hybrid_voice.py
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.hybrid_voice import generate_hybrid_voice


def main():
    # Load CustomVoice model (has instruction support)
    # You can also use Base model, but CustomVoice may have better instruction following
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Reference audio for voice cloning
    ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"

    # Test 1: Clone voice without instruction
    print("\n=== Test 1: Clone voice (no instruction) ===")
    wavs, sr = generate_hybrid_voice(
        model=model,
        text="Hello, this is a test of voice cloning without any style instruction.",
        ref_audio=ref_audio,
        language="English",
    )
    sf.write("output_hybrid_no_instruct.wav", wavs[0], sr)
    print("Saved: output_hybrid_no_instruct.wav")

    # Test 2: Clone voice with happy instruction
    print("\n=== Test 2: Clone voice + happy instruction ===")
    wavs, sr = generate_hybrid_voice(
        model=model,
        text="Hello, this is a test of voice cloning with a happy and excited tone!",
        ref_audio=ref_audio,
        instruct="Speak with excitement and happiness, like you just received great news!",
        language="English",
    )
    sf.write("output_hybrid_happy.wav", wavs[0], sr)
    print("Saved: output_hybrid_happy.wav")

    # Test 3: Clone voice with sad instruction
    print("\n=== Test 3: Clone voice + sad instruction ===")
    wavs, sr = generate_hybrid_voice(
        model=model,
        text="I'm sorry to tell you this news. It's been a difficult day.",
        ref_audio=ref_audio,
        instruct="Speak with a sad, melancholic tone, as if delivering bad news.",
        language="English",
    )
    sf.write("output_hybrid_sad.wav", wavs[0], sr)
    print("Saved: output_hybrid_sad.wav")

    # Test 4: Clone voice with angry instruction
    print("\n=== Test 4: Clone voice + angry instruction ===")
    wavs, sr = generate_hybrid_voice(
        model=model,
        text="This is absolutely unacceptable! I demand an explanation right now!",
        ref_audio=ref_audio,
        instruct="Speak with anger and frustration, raising your voice.",
        language="English",
    )
    sf.write("output_hybrid_angry.wav", wavs[0], sr)
    print("Saved: output_hybrid_angry.wav")

    print("\n=== All tests completed! ===")
    print("Compare the output files to hear the difference in emotional delivery.")


if __name__ == "__main__":
    main()
