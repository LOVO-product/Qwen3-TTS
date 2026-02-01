# coding=utf-8
# Hybrid voice generation: combines voice cloning with instruction control
# This allows using reference audio to clone a voice while also controlling style via instructions

import librosa
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .qwen3_tts_model import Qwen3TTSModel, AudioLike


@dataclass
class HybridVoicePrompt:
    """
    Pre-computed voice prompt for hybrid generation.

    This allows you to extract speaker embedding once and reuse it for multiple generations,
    significantly speeding up inference when using the same reference audio.

    Fields:
        speaker_embedding: Speaker embedding tensor extracted from reference audio
        ref_code: Optional audio codes for ICL mode (when ref_text was provided)
        ref_text: Optional reference text for ICL mode
        use_icl: Whether ICL mode is enabled (True when ref_text was provided)
    """
    speaker_embedding: torch.Tensor
    ref_code: Optional[torch.Tensor] = None
    ref_text: Optional[str] = None
    use_icl: bool = False


# Cache for Base model used for speaker embedding extraction
_speaker_encoder_model = None


def _get_speaker_encoder_model(device, dtype):
    """Load Base model just for speaker embedding extraction (cached)."""
    global _speaker_encoder_model
    if _speaker_encoder_model is None:
        print("Loading Base model for speaker embedding extraction...")
        _speaker_encoder_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=dtype,
        )
    return _speaker_encoder_model


def create_hybrid_voice_prompt(
    model: Qwen3TTSModel,
    ref_audio: AudioLike,
    ref_text: Optional[str] = None,
) -> HybridVoicePrompt:
    """
    Pre-compute a voice prompt from reference audio for fast repeated generation.

    This extracts the speaker embedding (and optionally audio codes for ICL mode) once,
    allowing you to reuse it for multiple generations without re-processing the audio.

    Args:
        model: A Qwen3TTSModel instance
        ref_audio: Reference audio (path, URL, base64, or (numpy, sr) tuple)
        ref_text: Optional transcript of reference audio. If provided, enables ICL mode
                  for higher quality cloning.

    Returns:
        HybridVoicePrompt: Pre-computed prompt that can be passed to generate_hybrid_voice()

    Example:
        # Pre-compute once
        prompt = create_hybrid_voice_prompt(model, "reference.wav", "Hello, my name is John.")

        # Reuse many times (fast!)
        wavs1, sr = generate_hybrid_voice(model, "Good morning!", voice_prompt=prompt, instruct="Speak happily")
        wavs2, sr = generate_hybrid_voice(model, "I'm sorry.", voice_prompt=prompt, instruct="Speak sadly")
    """
    # Check if current model has speaker encoder
    has_speaker_encoder = (
        hasattr(model.model, 'speaker_encoder') and
        model.model.speaker_encoder is not None
    )

    if has_speaker_encoder:
        encoder_model = model
    else:
        encoder_model = _get_speaker_encoder_model(
            device=str(model.device),
            dtype=model.model.dtype
        )

    # Normalize audio input
    normalized = model._normalize_audio_inputs([ref_audio])
    wav, sr = normalized[0]

    # Extract speaker embedding
    target_sr = encoder_model.model.speaker_encoder_sample_rate
    wav_resampled = wav
    if sr != target_sr:
        wav_resampled = librosa.resample(
            y=wav.astype(np.float32),
            orig_sr=int(sr),
            target_sr=target_sr
        )
    speaker_embedding = encoder_model.model.extract_speaker_embedding(
        audio=wav_resampled,
        sr=target_sr
    )

    # If ref_text provided, encode audio for ICL mode
    use_icl = ref_text is not None and ref_text.strip() != ""
    ref_code = None
    if use_icl:
        enc = encoder_model.model.speech_tokenizer.encode(wav, sr=sr)
        ref_code = enc.audio_codes[0]

    return HybridVoicePrompt(
        speaker_embedding=speaker_embedding,
        ref_code=ref_code,
        ref_text=ref_text.strip() if ref_text else None,
        use_icl=use_icl,
    )


def generate_hybrid_voice(
    model: Qwen3TTSModel,
    text: Union[str, List[str]],
    ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
    ref_text: Optional[Union[str, List[str]]] = None,
    voice_prompt: Optional[Union[HybridVoicePrompt, List[HybridVoicePrompt]]] = None,
    instruct: Optional[Union[str, List[str]]] = None,
    language: Union[str, List[str]] = None,
    non_streaming_mode: bool = True,
    prioritize_instruction: bool = True,
    **kwargs,
) -> Tuple[List[np.ndarray], int]:
    """
    Generate speech by cloning a voice from reference audio while applying instruction-based style control.

    This combines the best of both worlds:
    - Voice cloning: Uses speaker embedding extracted from reference audio
    - Instruction control: Applies style/emotion instructions like CustomVoice model

    You can provide either:
    - ref_audio (and optionally ref_text) to extract embeddings on-the-fly, OR
    - voice_prompt from create_hybrid_voice_prompt() for faster repeated generation

    Note: This function requires both Base model (for speaker extraction) and CustomVoice model
    (for instruction-controlled generation). The Base model is loaded automatically on first use.

    IMPORTANT: ICL mode (when ref_text is provided) and instruction control are partially incompatible.
    In ICL mode, the model learns to copy prosody/style from the reference audio, which can override
    instruction-based style control. When both ref_text and instruct are provided:
    - If prioritize_instruction=True (default): Uses x_vector_only mode to ensure instruction is followed
    - If prioritize_instruction=False: Uses ICL mode for higher cloning quality but instruction may be ignored

    Args:
        model: A Qwen3TTSModel instance (should be CustomVoice for best instruction following)
        text: Text(s) to synthesize
        ref_audio: Reference audio for voice cloning (path, URL, base64, or (numpy, sr) tuple).
                   Not needed if voice_prompt is provided.
        ref_text: Optional transcript of the reference audio. If provided AND prioritize_instruction=False,
                  enables ICL mode for higher quality cloning. Not needed if voice_prompt is provided.
        voice_prompt: Pre-computed prompt from create_hybrid_voice_prompt(). Use this for
                      faster repeated generation with the same reference audio.
        instruct: Optional instruction(s) for style control (e.g., "Speak with excitement")
        language: Language(s) for synthesis
        non_streaming_mode: Use non-streaming mode (default True)
        prioritize_instruction: When both ref_text and instruct are provided, if True (default),
                                disable ICL mode to ensure instruction is followed. If False,
                                use ICL mode for higher quality cloning but instruction may be ignored.
        **kwargs: Additional generation parameters (temperature, top_k, etc.)

    Returns:
        Tuple[List[np.ndarray], int]: (waveforms, sample_rate)

    Example:
        # Option 1: Direct (extracts embedding each time)
        wavs, sr = generate_hybrid_voice(model, "Hello!", ref_audio="ref.wav", instruct="Speak happily")

        # Option 2: With pre-computed prompt (faster for repeated use)
        prompt = create_hybrid_voice_prompt(model, "ref.wav", "Reference transcript")
        wavs, sr = generate_hybrid_voice(model, "Hello!", voice_prompt=prompt, instruct="Speak happily")

        # Option 3: High-quality clone without instruction (uses ICL mode)
        wavs, sr = generate_hybrid_voice(model, "Hello!", ref_audio="ref.wav", ref_text="ref transcript")

        # Option 4: Force ICL mode even with instruction (instruction may be ignored)
        wavs, sr = generate_hybrid_voice(model, "Hello!", ref_audio="ref.wav", ref_text="ref transcript",
                                         instruct="Speak happily", prioritize_instruction=False)
    """
    # Normalize inputs to lists
    texts = model._ensure_list(text)
    languages = model._ensure_list(language) if isinstance(language, list) else (
        [language] * len(texts) if language is not None else ["Auto"] * len(texts)
    )
    instructs = model._ensure_list(instruct) if isinstance(instruct, list) else (
        [instruct] * len(texts) if instruct is not None else [""] * len(texts)
    )

    # Expand single values to match batch size
    if len(languages) == 1 and len(texts) > 1:
        languages = languages * len(texts)
    if len(instructs) == 1 and len(texts) > 1:
        instructs = instructs * len(texts)

    model._validate_languages(languages)

    # Handle voice_prompt vs ref_audio
    if voice_prompt is not None:
        # Use pre-computed prompts
        prompts = model._ensure_list(voice_prompt)
        if len(prompts) == 1 and len(texts) > 1:
            prompts = prompts * len(texts)

        if len(prompts) != len(texts):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, voice_prompt={len(prompts)}")

        speaker_embeddings = [p.speaker_embedding for p in prompts]
        ref_codes = [p.ref_code for p in prompts]
        ref_texts = [p.ref_text for p in prompts]
        # Determine ICL mode, potentially overriding based on instruction
        use_icl = []
        for i, p in enumerate(prompts):
            has_instruction = instructs[i] is not None and instructs[i] != ""
            if p.use_icl and has_instruction and prioritize_instruction:
                # Disable ICL mode to ensure instruction is followed
                use_icl.append(False)
            else:
                use_icl.append(p.use_icl)
    else:
        # Extract from ref_audio
        if ref_audio is None:
            raise ValueError("Either ref_audio or voice_prompt must be provided.")

        ref_audios = model._ensure_list(ref_audio)
        ref_texts = model._ensure_list(ref_text) if isinstance(ref_text, list) else (
            [ref_text] * len(texts) if ref_text is not None else [None] * len(texts)
        )

        if len(ref_audios) == 1 and len(texts) > 1:
            ref_audios = ref_audios * len(texts)
        if len(ref_texts) == 1 and len(texts) > 1:
            ref_texts = ref_texts * len(texts)

        if not (len(texts) == len(ref_audios) == len(ref_texts)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, ref_audio={len(ref_audios)}, ref_text={len(ref_texts)}"
            )

        # Determine if using ICL mode (when ref_text is provided)
        # When instruction is provided and prioritize_instruction=True, disable ICL to ensure instruction is followed
        use_icl = []
        for i, rt in enumerate(ref_texts):
            has_ref_text = rt is not None and rt.strip() != ""
            has_instruction = instructs[i] is not None and instructs[i] != ""
            if has_ref_text and has_instruction and prioritize_instruction:
                # Disable ICL mode to ensure instruction is followed
                use_icl.append(False)
            else:
                use_icl.append(has_ref_text)

        # Check if current model has speaker encoder, otherwise use Base model
        has_speaker_encoder = (
            hasattr(model.model, 'speaker_encoder') and
            model.model.speaker_encoder is not None
        )

        if has_speaker_encoder:
            encoder_model = model
        else:
            encoder_model = _get_speaker_encoder_model(
                device=str(model.device),
                dtype=model.model.dtype
            )

        # Extract speaker embeddings from reference audio
        normalized_audios = model._normalize_audio_inputs(ref_audios)
        speaker_embeddings = []
        ref_codes = []

        for i, (wav, sr) in enumerate(normalized_audios):
            target_sr = encoder_model.model.speaker_encoder_sample_rate
            wav_resampled = wav
            if sr != target_sr:
                wav_resampled = librosa.resample(
                    y=wav.astype(np.float32),
                    orig_sr=int(sr),
                    target_sr=target_sr
                )
            spk_emb = encoder_model.model.extract_speaker_embedding(
                audio=wav_resampled,
                sr=target_sr
            )
            speaker_embeddings.append(spk_emb)

            if use_icl[i]:
                enc = encoder_model.model.speech_tokenizer.encode(wav, sr=sr)
                ref_codes.append(enc.audio_codes[0])
            else:
                ref_codes.append(None)

    # Validate final batch sizes
    if not (len(texts) == len(languages) == len(instructs) == len(speaker_embeddings)):
        raise ValueError(
            f"Batch size mismatch: text={len(texts)}, language={len(languages)}, "
            f"instruct={len(instructs)}, embeddings={len(speaker_embeddings)}"
        )

    # Build voice_clone_prompt dict for model.generate()
    voice_clone_prompt_dict = {
        "ref_code": ref_codes,
        "ref_spk_embedding": speaker_embeddings,
        "x_vector_only_mode": [not icl for icl in use_icl],
        "icl_mode": use_icl,
    }

    # Tokenize input texts
    input_ids = model._tokenize_texts([model._build_assistant_text(t) for t in texts])

    # Tokenize instructions
    instruct_ids: List[Optional[torch.Tensor]] = []
    for ins in instructs:
        if ins is None or ins == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(model._tokenize_texts([model._build_instruct_text(ins)])[0])

    # Tokenize reference texts for ICL mode
    ref_ids: List[Optional[torch.Tensor]] = []
    for i, rt in enumerate(ref_texts):
        if use_icl[i] and rt is not None and rt.strip() != "":
            ref_ids.append(model._tokenize_texts([model._build_ref_text(rt.strip())])[0])
        else:
            ref_ids.append(None)

    # Merge generation kwargs
    gen_kwargs = model._merge_generate_kwargs(**kwargs)

    # Generate with both voice_clone_prompt AND instruct_ids
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        ref_ids=ref_ids if any(use_icl) else None,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=languages,
        speakers=[None] * len(texts),  # Not using preset speakers
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )

    # For ICL mode, prepend ref_code to generated codes before decoding
    codes_for_decode = []
    for i, codes in enumerate(talker_codes_list):
        if use_icl[i] and ref_codes[i] is not None:
            codes_for_decode.append(torch.cat([ref_codes[i].to(codes.device), codes], dim=0))
        else:
            codes_for_decode.append(codes)

    # Decode to audio
    wavs_all, fs = model.model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in codes_for_decode]
    )

    # For ICL mode, strip the reference audio portion from the output
    wavs_out: List[np.ndarray] = []
    for i, wav in enumerate(wavs_all):
        if use_icl[i] and ref_codes[i] is not None:
            ref_len = int(ref_codes[i].shape[0])
            total_len = int(codes_for_decode[i].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wavs_out.append(wav[cut:])
        else:
            wavs_out.append(wav)

    return wavs_out, fs
