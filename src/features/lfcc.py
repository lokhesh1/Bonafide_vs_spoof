"""LFCC — Linear-Frequency Cepstral Coefficients.

Computed directly from the power spectrum with a triangular filterbank whose
centres are spaced **linearly** in Hz (the only difference from MFCC), followed
by log + DCT-II. Implemented here (rather than via spafe) to keep full control
of the framing so it matches the other librosa-based features.
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.fftpack import dct

from src.features.base import BaseFeatureExtractor
from src.features.utils import add_deltas, linear_filterbank


class LFCCExtractor(BaseFeatureExtractor):
    name = "lfcc"
    apply_pre_emphasis = True

    def _extract(self, signal: np.ndarray) -> np.ndarray:
        f, p = self.frame, self.params
        stft = librosa.stft(
            signal,
            n_fft=f.n_fft,
            hop_length=f.hop_length,
            win_length=f.win_length,
            window=f.window,
        )
        power = np.abs(stft) ** 2  # (n_freq, n_frames)

        fb = linear_filterbank(
            n_filters=p.get("n_filters", 70),
            n_fft=f.n_fft,
            sample_rate=f.sample_rate,
            fmin=p.get("fmin", 0),
            fmax=p.get("fmax", None),
        )
        filt_energy = fb @ power                       # (n_filters, n_frames)
        log_energy = np.log(filt_energy + 1e-10)
        cepstra = dct(log_energy, type=2, axis=0, norm="ortho")
        cepstra = cepstra[: p.get("n_ceps", 20)]       # (n_ceps, n_frames)

        feat = cepstra.T  # (n_frames, n_ceps)
        if self.include_deltas:
            feat = add_deltas(feat)
        return feat
