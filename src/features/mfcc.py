"""MFCC — Mel-Frequency Cepstral Coefficients (librosa)."""
from __future__ import annotations

import librosa
import numpy as np

from src.features.base import BaseFeatureExtractor
from src.features.utils import add_deltas


class MFCCExtractor(BaseFeatureExtractor):
    name = "mfcc"
    apply_pre_emphasis = True

    def _extract(self, signal: np.ndarray) -> np.ndarray:
        f, p = self.frame, self.params
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=f.sample_rate,
            n_mfcc=p.get("n_mfcc", 20),
            n_fft=f.n_fft,
            hop_length=f.hop_length,
            win_length=f.win_length,
            window=f.window,
            n_mels=p.get("n_mels", 40),
            fmin=p.get("fmin", 0),
            fmax=p.get("fmax", None),
        )
        feat = mfcc.T  # (n_frames, n_mfcc)
        if self.include_deltas:
            feat = add_deltas(feat)
        return feat
