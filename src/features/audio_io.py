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
