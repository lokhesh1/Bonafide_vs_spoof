"""Spectral Flux — one value per frame.

Measures how quickly the magnitude spectrum changes between consecutive frames:
the norm of the frame-to-frame spectral difference. The first frame is padded
with 0. Output is (n_frames, 1).

Config (`feature_params['spectral_flux']`):
    flux_type : 'l2'        -> L2 norm of the difference (default)
                'rectified' -> sum of positive differences (half-wave rectified)
    normalize : bool        -> per-frame normalise the spectrum first, so flux
                               reflects spectral *shape* change, not loudness.
"""
from __future__ import annotations

import librosa
import numpy as np

from src.features.base import BaseFeatureExtractor


class SpectralFluxExtractor(BaseFeatureExtractor):
    name = "spectral_flux"
    apply_pre_emphasis = False

    def _extract(self, signal: np.ndarray) -> np.ndarray:
        f, p = self.frame, self.params
        mag = np.abs(
            librosa.stft(
                signal,
                n_fft=f.n_fft,
                hop_length=f.hop_length,
                win_length=f.win_length,
                window=f.window,
            )
        )  # (n_freq, n_frames)

        if p.get("normalize", True):
            mag = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-12)

        diff = np.diff(mag, axis=1)  # (n_freq, n_frames - 1)
        if p.get("flux_type", "l2") == "rectified":
            flux = np.sum(np.maximum(diff, 0.0), axis=0)
        else:
            flux = np.sqrt(np.sum(diff ** 2, axis=0))

        flux = np.concatenate([[0.0], flux])  # pad first frame -> length n_frames
        return flux[:, None].astype(np.float32)  # (n_frames, 1)
