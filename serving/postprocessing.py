"""
Audio Post-processing Utilities for LOVO TTS

Provides audio transformations like loudness normalization, EQ adjustments,
pitch shifting, and fades. These match the post-processing chain used in
tortoise-tts for consistency across LOVO voices.

Optional Dependencies:
    - pyloudnorm: Required for loudness normalization (pip install pyloudnorm)
    - torchaudio: Required for bass/treble boost (usually installed with torch)
    - audiotools: Required for pitch shifting (pip install audiotools)

If a dependency is missing, the corresponding effect will be skipped with a warning.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def normalize_loudness(
    wav: np.ndarray,
    sample_rate: int,
    target_lufs: float = -24.0,
) -> np.ndarray:
    """
    Normalize audio to target loudness level (LUFS).

    Args:
        wav: Audio waveform (float32, mono)
        sample_rate: Sample rate in Hz
        target_lufs: Target loudness in LUFS (default: -24)

    Returns:
        Loudness-normalized waveform
    """
    try:
        import pyloudnorm as pyln

        # Peak normalize first to prevent clipping
        peak_normalized = pyln.normalize.peak(wav, -2.0)

        # Measure loudness
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(peak_normalized)

        # Normalize to target
        if loudness > -70:  # Only normalize if we can measure loudness
            normalized = pyln.normalize.loudness(peak_normalized, loudness, target_lufs)
            return normalized.astype(np.float32)

        return peak_normalized.astype(np.float32)

    except ImportError:
        logger.warning("pyloudnorm not installed, skipping loudness normalization")
        return wav


def boost_bass(
    wav: np.ndarray,
    sample_rate: int,
    gain_db: float,
    central_freq: float = 100.0,
) -> np.ndarray:
    """
    Apply bass boost using a biquad filter.

    Args:
        wav: Audio waveform
        sample_rate: Sample rate in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        central_freq: Center frequency for bass (default: 100 Hz)

    Returns:
        Bass-boosted waveform
    """
    try:
        import torch
        import torchaudio

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()
        boosted = torchaudio.functional.bass_biquad(
            waveform=wav_tensor,
            sample_rate=sample_rate,
            gain=gain_db,
            central_freq=central_freq,
            Q=0.3,
        )
        return boosted.squeeze(0).numpy()

    except ImportError:
        logger.warning("torchaudio not installed, skipping bass boost")
        return wav


def boost_treble(
    wav: np.ndarray,
    sample_rate: int,
    gain_db: float,
) -> np.ndarray:
    """
    Apply treble boost using biquad filters.

    Applies two-pass treble adjustment at 1kHz and 2kHz for smooth response.

    Args:
        wav: Audio waveform
        sample_rate: Sample rate in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)

    Returns:
        Treble-boosted waveform
    """
    try:
        import torch
        import torchaudio

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()

        # First pass at 1kHz
        boosted = torchaudio.functional.treble_biquad(
            waveform=wav_tensor,
            sample_rate=sample_rate,
            gain=gain_db,
            central_freq=1000,
            Q=0.3,
        )

        # Second pass at 2kHz
        boosted = torchaudio.functional.treble_biquad(
            waveform=boosted,
            sample_rate=sample_rate,
            gain=gain_db,
            central_freq=2000,
            Q=0.3,
        )

        # Compensate for gain increase
        boosted = torchaudio.functional.gain(boosted, gain_db=float(-1 * gain_db / 2))

        return boosted.squeeze(0).numpy()

    except ImportError:
        logger.warning("torchaudio not installed, skipping treble boost")
        return wav


def apply_pitch_shift(
    wav: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    """
    Shift pitch by semitones.

    Args:
        wav: Audio waveform
        sample_rate: Sample rate in Hz
        semitones: Number of semitones to shift (positive = higher, negative = lower)

    Returns:
        Pitch-shifted waveform
    """
    try:
        from audiotools import AudioSignal

        signal = AudioSignal(wav.reshape(1, 1, -1), sample_rate=sample_rate)
        shifted = signal.clone().pitch_shift(semitones)
        return shifted.audio_data.squeeze().numpy().astype(np.float32)

    except ImportError:
        logger.warning("audiotools not installed, skipping pitch shift")
        return wav


def apply_fade(
    wav: np.ndarray,
    sample_rate: int,
    fade_in_seconds: Optional[float] = None,
    fade_out_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Apply fade in/out to audio.

    Args:
        wav: Audio waveform
        sample_rate: Sample rate in Hz
        fade_in_seconds: Duration of fade in (None = no fade in)
        fade_out_seconds: Duration of fade out (None = no fade out)

    Returns:
        Audio with fades applied
    """
    result = wav.copy()

    if fade_in_seconds and fade_in_seconds > 0:
        fade_samples = int(fade_in_seconds * sample_rate)
        fade_samples = min(fade_samples, len(result))
        fade_curve = np.linspace(0, 1, fade_samples)
        result[:fade_samples] *= fade_curve

    if fade_out_seconds and fade_out_seconds > 0:
        fade_samples = int(fade_out_seconds * sample_rate)
        fade_samples = min(fade_samples, len(result))
        # Exponential fade out for smoother decay
        fade_curve = np.linspace(1, 0, fade_samples) ** 2
        result[-fade_samples:] *= fade_curve

    return result


def postprocess_audio(
    wav: np.ndarray,
    sample_rate: int,
    loudness_lufs: Optional[float] = None,
    bass_boost: Optional[float] = None,
    treble_boost: Optional[float] = None,
    pitch_shift: Optional[float] = None,
    fade_in: Optional[float] = None,
    fade_out: Optional[float] = None,
) -> np.ndarray:
    """
    Apply full post-processing chain to audio.

    Order of operations:
    1. Bass boost
    2. Treble boost
    3. Pitch shift
    4. Fade in/out
    5. Loudness normalization (last to ensure consistent output level)

    Args:
        wav: Audio waveform (float32, mono)
        sample_rate: Sample rate in Hz
        loudness_lufs: Target loudness in LUFS (None = skip)
        bass_boost: Bass boost in dB (None = skip)
        treble_boost: Treble boost in dB (None = skip)
        pitch_shift: Pitch shift in semitones (None = skip)
        fade_in: Fade in duration in seconds (None = skip)
        fade_out: Fade out duration in seconds (None = skip)

    Returns:
        Processed audio waveform
    """
    # Skip if no processing needed
    if all(x is None for x in [loudness_lufs, bass_boost, treble_boost, pitch_shift, fade_in, fade_out]):
        return wav

    result = wav.astype(np.float32)

    # EQ adjustments
    if bass_boost is not None:
        result = boost_bass(result, sample_rate, bass_boost)

    if treble_boost is not None:
        result = boost_treble(result, sample_rate, treble_boost)

    # Pitch shift
    if pitch_shift is not None:
        result = apply_pitch_shift(result, sample_rate, pitch_shift)

    # Fades
    if fade_in is not None or fade_out is not None:
        result = apply_fade(result, sample_rate, fade_in, fade_out)

    # Loudness normalization (last)
    if loudness_lufs is not None:
        result = normalize_loudness(result, sample_rate, loudness_lufs)

    return result
