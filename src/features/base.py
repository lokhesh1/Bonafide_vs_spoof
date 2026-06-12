"""Base class shared by all feature extractors.

Contract
--------
Every extractor's `extract(signal)` returns a 2-D float32 array of shape
**(n_frames, n_coeff)** — time on axis 0, coefficients on axis 1. Scalar-per-frame
features (spectral flatness/flux) use n_coeff == 1. This uniform orientation lets
the orchestrator save frames and compute summary statistics generically.
"""
from __future__ import annotations

import numpy as np

from src.features.utils import pre_emphasis


class BaseFeatureExtractor:
    name: str = "base"
    apply_pre_emphasis: bool = False  # cepstral features override to True

    def __init__(self, frame, include_deltas: bool = False, **params):
        # `frame` is duck-typed: needs sample_rate, n_fft, window,
        # win_length, hop_length, pre_emphasis.
        self.frame = frame
        self.include_deltas = include_deltas
        self.params = params

    def _extract(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extract(self, signal: np.ndarray) -> np.ndarray:
        if self.apply_pre_emphasis:
            signal = pre_emphasis(signal, getattr(self.frame, "pre_emphasis", 0.0))
        feat = self._extract(signal)
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 2:
            raise ValueError(
                f"{self.name}._extract must return a 2-D (n_frames, n_coeff) "
                f"array, got shape {feat.shape}"
            )
        return feat
