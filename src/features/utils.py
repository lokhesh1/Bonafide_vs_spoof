"""Shared signal-processing and aggregation helpers for feature extractors."""
from __future__ import annotations

from typing import Dict, Optional

import librosa
import numpy as np
from scipy import stats


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply a first-order pre-emphasis filter: y[n] = x[n] - a*x[n-1]."""
    if coeff <= 0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)


def linear_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Triangular filterbank with centres spaced **linearly** in Hz (for LFCC).

    Returns an array of shape (n_filters, 1 + n_fft // 2).
    """
    fmax = fmax if fmax is not None else sample_rate / 2.0
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, 1 + n_fft // 2)
    edges = np.linspace(fmin, fmax, n_filters + 2)  # linear spacing

    fb = np.zeros((n_filters, fft_freqs.size), dtype=np.float64)
    for m in range(1, n_filters + 1):
        left, center, right = edges[m - 1], edges[m], edges[m + 1]
        left_slope = (fft_freqs - left) / max(center - left, 1e-12)
        right_slope = (right - fft_freqs) / max(right - center, 1e-12)
        fb[m - 1] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    return fb


def _safe_delta(feat: np.ndarray, order: int) -> np.ndarray:
    """librosa delta along the time axis (axis 0), robust to short signals."""
    n_frames = feat.shape[0]
    if n_frames < 3:
        return np.zeros_like(feat)
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if width < 3:
        width = 3
    return librosa.feature.delta(feat, width=width, order=order, axis=0)


def add_deltas(feat: np.ndarray, orders=(1, 2)) -> np.ndarray:
    """Append Δ (and ΔΔ) coefficients along the feature axis.

    `feat` is (n_frames, n_coeff); result is (n_frames, n_coeff * (1 + len(orders))).
    """
    parts = [feat]
    for o in orders:
        parts.append(_safe_delta(feat, o))
    return np.concatenate(parts, axis=1)


# Summary statistics computed per coefficient (i.e. across the time axis).
SUMMARY_STATS = ("mean", "std", "min", "max", "median", "skew", "kurtosis")


def summarize(feat: np.ndarray) -> Dict[str, np.ndarray]:
    """Reduce an (n_frames, n_coeff) matrix to per-coefficient statistics.

    Each value in the returned dict is a length-`n_coeff` vector.
    """
    feat = np.asarray(feat, dtype=np.float64)
    if feat.ndim != 2:
        raise ValueError(f"summarize expects a 2-D array, got shape {feat.shape}")

    out: Dict[str, np.ndarray] = {
        "mean": np.nanmean(feat, axis=0),
        "std": np.nanstd(feat, axis=0),
        "min": np.nanmin(feat, axis=0),
        "max": np.nanmax(feat, axis=0),
        "median": np.nanmedian(feat, axis=0),
    }
    sk = stats.skew(feat, axis=0, bias=False, nan_policy="omit")
    ku = stats.kurtosis(feat, axis=0, bias=False, nan_policy="omit")
    out["skew"] = np.nan_to_num(np.asarray(sk, dtype=np.float64))
    out["kurtosis"] = np.nan_to_num(np.asarray(ku, dtype=np.float64))
    return out
