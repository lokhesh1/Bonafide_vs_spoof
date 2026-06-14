"""Audio loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np


def load_audio(path: Union[str, Path], sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Load an audio file, resampling to `sr` and downmixing to mono.

    Returns a 1-D float32 waveform.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    return np.ascontiguousarray(y, dtype=np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D waveform to `target_sr` (no-op if rates already match).

    Returns a 1-D float32 waveform.
    """
    y = np.ascontiguousarray(y, dtype=np.float32)
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
