"""Strategies for forming corresponding bona-fide <-> spoof pairs.

The ASVspoof2019 LA CM protocol does NOT contain an explicit bona-fide/spoof
mapping (spoof utterances are TTS/VC-generated). For the initial trial we pair
**1:1 within the same speaker**: each pair shares a speaker, every utterance is
used at most once, and the result is deterministic given `seed`.

Adding a new strategy = add a function + register it in `PAIRING_STRATEGIES`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

Pair = Tuple[str, str]  # (bonafide_utt, spoof_utt)


def pair_by_speaker(
    df: pd.DataFrame,
    n_pairs: Optional[int] = None,
    seed: int = 1234,
    spoof_systems: Optional[List[str]] = None,
) -> List[Pair]:
    """Pair each bona-fide utterance 1:1 with a spoof utterance of the same speaker.

    Within every speaker the bona-fide and spoof pools are shuffled and zipped,
    so each utterance appears in at most one pair. Pairs are then globally
    shuffled and truncated to `n_pairs`.
    """
    rng = np.random.default_rng(seed)

    bona = df[df["label"] == "bonafide"]
    spoof = df[df["label"] == "spoof"]
    if spoof_systems:
        spoof = spoof[spoof["system_id"].isin(spoof_systems)]

    pairs: List[Pair] = []
    for speaker in sorted(df["speaker"].unique()):
        b = bona.loc[bona["speaker"] == speaker, "utt_id"].to_numpy()
        s = spoof.loc[spoof["speaker"] == speaker, "utt_id"].to_numpy()
        if len(b) == 0 or len(s) == 0:
            continue
        b = b[rng.permutation(len(b))]
        s = s[rng.permutation(len(s))]
        k = min(len(b), len(s))
        pairs.extend(zip(b[:k].tolist(), s[:k].tolist()))

    # Global shuffle for an unbiased subsample, then truncate.
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    if n_pairs is not None:
        pairs = pairs[:n_pairs]
    return pairs


def pair_from_file(
    path: str,
    n_pairs: Optional[int] = None,
    **_ignored,
) -> List[Pair]:
    """Load explicit pairs from a CSV with columns `bonafide_utt,spoof_utt`."""
    if path is None:
        raise ValueError("pairing_file must be set in config.py for 'from_file'.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pairing file not found: {p}")
    pf = pd.read_csv(p)
    missing = {"bonafide_utt", "spoof_utt"} - set(pf.columns)
    if missing:
        raise ValueError(f"Pairing file missing columns: {missing}")
    pairs = list(zip(pf["bonafide_utt"].tolist(), pf["spoof_utt"].tolist()))
    if n_pairs is not None:
        pairs = pairs[:n_pairs]
    return pairs


PAIRING_STRATEGIES = {
    "speaker_1to1": pair_by_speaker,
    "from_file": pair_from_file,
}


def build_pairs(df: pd.DataFrame, config) -> List[Pair]:
    """Dispatch to the configured pairing strategy."""
    strategy = config.pairing_strategy
    if strategy not in PAIRING_STRATEGIES:
        raise KeyError(
            f"Unknown pairing_strategy '{strategy}'. "
            f"Available: {list(PAIRING_STRATEGIES)}"
        )
    if strategy == "speaker_1to1":
        return pair_by_speaker(
            df,
            n_pairs=config.n_pairs,
            seed=config.seed,
            spoof_systems=config.spoof_systems,
        )
    if strategy == "from_file":
        return pair_from_file(config.pairing_file, n_pairs=config.n_pairs)
    raise AssertionError("unreachable")
