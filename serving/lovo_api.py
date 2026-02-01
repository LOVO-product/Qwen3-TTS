"""
LOVO API Wrapper for Qwen3-TTS

Provides a unified interface for thirdpartytts integration, abstracting
the different Qwen3 generation modes (custom_voice, voice_design, base).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.hybrid_voice import generate_hybrid_voice, create_hybrid_voice_prompt

from .postprocessing import postprocess_audio

logger = logging.getLogger(__name__)

# Default path to speakers config (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "speakers_config.json"


@dataclass
class TtsOutput:
    """Output container matching thirdpartytts expectations."""
    wav_bytes: bytes
    sample_rate: int
    text_replacement_by_user: List[Dict] = None
    skip_post_process: bool = True  # Qwen handles its own audio processing


class LovoQwenTTS:
    """
    LOVO wrapper for Qwen3-TTS that provides a unified interface
    compatible with thirdpartytts service.

    Usage:
        model = LovoQwenTTS.from_pretrained("path/to/model")
        output = model.synthesize(
            text="Hello world",
            speaker="Amy",
            speed=1.0,
        )
    """

    def __init__(
        self,
        model: Qwen3TTSModel,
        speakers_config: Dict[str, Any],
    ):
        """
        Initialize with a loaded model and speaker configuration.

        Args:
            model: Loaded Qwen3TTSModel instance
            speakers_config: Speaker configuration dictionary
        """
        self.model = model
        self.speakers_config = speakers_config
        self.sample_rate = None  # Will be set after first generation

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        **model_kwargs,
    ) -> "LovoQwenTTS":
        """
        Load model and configuration in HuggingFace style.

        Args:
            model_path: Path to Qwen3-TTS model (HuggingFace repo or local)
            config_path: Path to speakers_config.json (default: serving/speakers_config.json)
            **model_kwargs: Additional arguments for Qwen3TTSModel.from_pretrained()

        Returns:
            LovoQwenTTS instance
        """
        # Load speaker config
        config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        with open(config_path) as f:
            speakers_config = json.load(f)

        # Load model
        logger.info(f"Loading Qwen3-TTS model from {model_path}")
        model = Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)
        logger.info("Model loaded successfully")

        return cls(model=model, speakers_config=speakers_config)

    def _get_speaker_config(self, speaker: str) -> Dict[str, Any]:
        """Get configuration for a speaker, with fallback to defaults."""
        if speaker in self.speakers_config:
            return self.speakers_config[speaker]

        # Check if speaker exists in model's supported speakers
        supported = self.model.get_supported_speakers()
        if supported and speaker.lower() in [s.lower() for s in supported]:
            # Return default custom_voice config
            return {
                "model_type": "custom_voice",
                "speaker_id": speaker,
                "language": "Auto",
            }

        raise ValueError(
            f"Unknown speaker: {speaker}. "
            f"Available in config: {list(self.speakers_config.keys())}"
        )

    def _wav_to_bytes(self, wav: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy waveform to WAV bytes."""
        import io
        import wave

        # Convert to int16 based on input dtype
        if wav.dtype in (np.float32, np.float64):
            # Clip to prevent overflow and convert
            wav_clipped = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav_clipped * 32767).astype(np.int16)
        elif wav.dtype == np.int16:
            wav_int16 = wav
        elif wav.dtype == np.int32:
            wav_int16 = (wav // 65536).astype(np.int16)
        else:
            # Best effort conversion
            wav_int16 = wav.astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(wav_int16.tobytes())

        return buffer.getvalue()

    def synthesize(
        self,
        text: str,
        speaker: str,
        speed: float = 1.0,
        language: Optional[str] = None,
        ref_audio: Optional[Union[str, bytes, np.ndarray]] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        text_replacement_by_user: Optional[List[Dict]] = None,
        prioritize_instruction: bool = True,
        **kwargs,
    ) -> TtsOutput:
        """
        Synthesize speech from text.

        This method routes to the appropriate Qwen3 generation method based
        on the speaker configuration (custom_voice, voice_design, base, or hybrid).

        Args:
            text: Text to synthesize
            speaker: LOVO speaker name (must be in speakers_config.json or model)
            speed: Speech speed multiplier (not yet supported by Qwen3, parameter reserved)
            language: Language code (default: from config or "Auto")
            ref_audio: Reference audio for voice cloning (path, bytes, or numpy)
            ref_text: Reference text for ICL voice cloning (higher quality)
            instruct: Style/emotion instruction (e.g., "Speak with excitement")
            text_replacement_by_user: Text replacements (passed through, not applied)
            prioritize_instruction: When both ref_text and instruct are provided,
                                    prioritize instruction over reference style
            **kwargs: Additional generation parameters

        Returns:
            TtsOutput with wav_bytes and metadata
        """
        config = self._get_speaker_config(speaker)
        model_type = config.get("model_type", "custom_voice")
        lang = language or config.get("language", "Auto")

        # Speed control not yet supported by Qwen3
        if speed != 1.0:
            logger.warning(f"speed={speed} requested but Qwen3-TTS does not support speed control yet. Ignoring.")

        # Mark text replacements as not applied (Qwen handles its own text processing)
        if text_replacement_by_user:
            for item in text_replacement_by_user:
                item["isReplaced"] = False

        # Route to appropriate generation method
        if model_type == "hybrid":
            # Hybrid mode: voice cloning + instruction control
            if ref_audio is None:
                raise ValueError("ref_audio is required for hybrid voice cloning")

            voice_instruct = instruct or config.get("instruct")
            wavs, sample_rate = generate_hybrid_voice(
                model=self.model,
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                instruct=voice_instruct,
                language=lang,
                prioritize_instruction=prioritize_instruction,
                **kwargs,
            )

        elif model_type == "base":
            # Voice cloning mode
            if ref_audio is None:
                raise ValueError("ref_audio is required for voice cloning (model_type=base)")

            # If instruct is provided with base model, use hybrid mode instead
            voice_instruct = instruct or config.get("instruct")
            if voice_instruct:
                logger.info("Instruction provided with base model, using hybrid voice generation")
                wavs, sample_rate = generate_hybrid_voice(
                    model=self.model,
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    instruct=voice_instruct,
                    language=lang,
                    prioritize_instruction=prioritize_instruction,
                    **kwargs,
                )
            else:
                x_vector_only = config.get("x_vector_only_mode", False)
                if not x_vector_only and ref_text is None:
                    raise ValueError("ref_text is required for ICL voice cloning (x_vector_only_mode=False)")

                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                    **kwargs,
                )

        elif model_type == "voice_design":
            # Voice design mode (natural language instruction)
            voice_instruct = instruct or config.get("instruct", "")

            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                language=lang,
                instruct=voice_instruct,
                **kwargs,
            )

        else:  # custom_voice (default)
            # Predefined speaker mode
            speaker_id = config.get("speaker_id", speaker)
            voice_instruct = instruct or config.get("instruct")

            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                speaker=speaker_id,
                language=lang,
                instruct=voice_instruct,
                **kwargs,
            )

        self.sample_rate = sample_rate
        wav = wavs[0]  # Take first result (batch size 1)

        # Apply post-processing if configured
        wav = postprocess_audio(
            wav=wav,
            sample_rate=sample_rate,
            loudness_lufs=config.get("loudness_lufs_level"),
            bass_boost=config.get("bass_boost"),
            treble_boost=config.get("treble_boost"),
            pitch_shift=config.get("pitch_shift"),
            fade_in=config.get("fade_in"),
            fade_out=config.get("fade_out"),
        )

        # Convert to bytes
        wav_bytes = self._wav_to_bytes(wav, sample_rate)

        return TtsOutput(
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            text_replacement_by_user=text_replacement_by_user or [],
            skip_post_process=True,  # We already did post-processing
        )

    def get_supported_speakers(self) -> List[str]:
        """Get list of available speakers (from config + model)."""
        config_speakers = [
            k for k in self.speakers_config.keys()
            if not k.startswith("_")
        ]
        model_speakers = self.model.get_supported_speakers() or []
        return list(set(config_speakers + model_speakers))

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.model.get_supported_languages() or ["Auto"]
