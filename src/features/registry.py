"""Registry mapping feature names -> extractor classes.

To add a feature: implement a `BaseFeatureExtractor` subclass and register it
here. `build_extractors` then wires it up from config with no other changes.
"""
from __future__ import annotations

from typing import Dict, List

from src.features.base import BaseFeatureExtractor
from src.features.cqcc import CQCCExtractor
from src.features.lfcc import LFCCExtractor
from src.features.mfcc import MFCCExtractor
from src.features.spectral_flatness import SpectralFlatnessExtractor
from src.features.spectral_flux import SpectralFluxExtractor

EXTRACTOR_REGISTRY: Dict[str, type] = {
    "mfcc": MFCCExtractor,
    "lfcc": LFCCExtractor,
    "cqcc": CQCCExtractor,
    "spectral_flatness": SpectralFlatnessExtractor,
    "spectral_flux": SpectralFluxExtractor,
}


def build_extractors(
    enabled_features: List[str],
    feature_params: dict,
    frame,
    include_deltas: bool,
) -> List[BaseFeatureExtractor]:
    extractors: List[BaseFeatureExtractor] = []
    for name in enabled_features:
        if name not in EXTRACTOR_REGISTRY:
            raise KeyError(
                f"Unknown feature '{name}'. Available: {list(EXTRACTOR_REGISTRY)}"
            )
        cls = EXTRACTOR_REGISTRY[name]
        params = dict(feature_params.get(name, {}))
        extractors.append(cls(frame=frame, include_deltas=include_deltas, **params))
    return extractors
