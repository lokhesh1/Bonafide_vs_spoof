"""CQCC — Constant-Q Cepstral Coefficients (via the `spafe` library).

CQCC is not available in librosa, so we delegate to spafe. spafe's public API
has changed across releases (notably the windowing argument in 0.3.x), so the
call is wrapped defensively: we try the modern `SlidingWindow` signature first
and fall back to a minimal call. Pin `spafe>=0.3.2` (see requirements.txt).

This module is intentionally a thin adapter — to swap in a custom CQT-based
implementation later, replace `_extract` and keep the same return contract.
"""
from __future__ import annotations

import numpy as np

from src.features.base import BaseFeatureExtractor
from src.features.utils import add_deltas


class CQCCExtractor(BaseFeatureExtractor):
    name = "cqcc"
    apply_pre_emphasis = True

    def _extract(self, signal: np.ndarray) -> np.ndarray:
        from spafe.features.cqcc import cqcc  # imported lazily

        f, p = self.frame, self.params
        num_ceps = p.get("n_ceps", 20)
        kwargs = dict(fs=f.sample_rate, num_ceps=num_ceps, nfft=f.n_fft)
        if p.get("fmin") is not None:
            kwargs["low_freq"] = p["fmin"]
        if p.get("fmax") is not None:
            kwargs["high_freq"] = p["fmax"]

        try:
            from spafe.utils.preprocessing import SlidingWindow

            window = SlidingWindow(
                f.win_length_ms / 1000.0,
                f.hop_length_ms / 1000.0,
                "hamming",
            )
            feats = cqcc(signal, window=window, **kwargs)
        except (ImportError, TypeError):
            # Older spafe without SlidingWindow / different signature.
            feats = cqcc(signal, **kwargs)

        feats = np.asarray(feats, dtype=np.float32)  # (n_frames, num_ceps)
        if feats.ndim != 2:
            raise ValueError(f"Unexpected CQCC shape from spafe: {feats.shape}")
        if self.include_deltas:
            feats = add_deltas(feats)
        return feats
