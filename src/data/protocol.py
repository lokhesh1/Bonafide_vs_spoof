"""Parser for the ASVspoof2019 LA CM protocol files.

Protocol line format (whitespace separated), e.g. the train protocol
`ASVspoof2019.LA.cm.train.trn.txt`:

    LA_0079 LA_T_1138215 - -   bonafide
    LA_0079 LA_T_1271820 - A01 spoof

Columns:
    0: speaker_id      (e.g. LA_0079)
    1: utt_id          (audio file stem, e.g. LA_T_1138215  ->  LA_T_1138215.flac)
    2: unused          ("-")
    3: system_id       ("-" for bona fide, A01..A06 for train spoof)
    4: label           ("bonafide" | "spoof")
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PROTOCOL_COLUMNS = ["speaker", "utt_id", "_unused", "system_id", "label"]


def load_protocol(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CM protocol file into a DataFrame.

    Returns columns: ``speaker, utt_id, system_id, label``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {path}\n"
            "Set DATASET_ROOT / protocol_relpath in config.py."
        )
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PROTOCOL_COLUMNS,
        engine="python",
    )
    df = df.drop(columns=["_unused"])

    # Sanity check on labels.
    labels = set(df["label"].unique())
    unexpected = labels - {"bonafide", "spoof"}
    if unexpected:
        raise ValueError(
            f"Unexpected label(s) {unexpected} in {path}. "
            "Is this an ASVspoof2019 LA CM protocol file?"
        )
    return df
