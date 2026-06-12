"""Spectral Flatness — one value per frame (Wiener entropy).

Ratio of the geometric mean to the arithmetic mean of the power spectrum per
frame. Near 1.0 => noise-like / flat spectrum; near 0.0 => tonal. Computed for
every frame in the signal, so the output is (n_frames, 1).
"""
from __future__ import annotations

import librosa
import numpy as np

from src.features.base import BaseFeatureExtractor


class SpectralFlatnessExtractor(BaseFeatureExtractor):
    name = "spectral_flatness"
    apply_pre_emphasis = False

    def _extract(self, signal: np.ndarray) -> np.ndarray:
        f = self.frame
        flatness = librosa.feature.spectral_flatness(
            y=signal,
            n_fft=f.n_fft,
            hop_length=f.hop_length,
            win_length=f.win_length,
            window=f.window,
        )  # (1, n_frames)
        return flatness.T  # (n_frames, 1)
