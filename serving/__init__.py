"""
LOVO Serving Layer for Qwen3-TTS

This module provides the serving infrastructure to integrate Qwen3-TTS
with LOVO's thirdpartytts service.

Components:
- LovoQwenTTS: Main wrapper class for thirdpartytts integration
- postprocessing: Audio post-processing utilities (loudness, bass, etc.)
- speakers_config.json: LOVO speaker mappings and settings
"""

from .lovo_api import LovoQwenTTS, TtsOutput

__all__ = ["LovoQwenTTS", "TtsOutput"]
