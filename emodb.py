"""
emodb.py
========

PyTorch ``Dataset`` + ``DataLoader`` utilities for **emotion recognition** on the
EMO-DB dataset (Berlin Database of Emotional Speech).

The task here is EMOTION classification:
    input  -> audio waveform
    target -> emotion id in [0 .. num_classes-1]

Where do the labels come from?
------------------------------
Unlike AudioMNIST (where the label is the folder name), EMO-DB ships its labels
in separate **gold-standard CSV files**, NOT in the directory layout. The dataset
already defines a fixed train / test split via two files::

    emodb/
      wav/                                            ->  03a01Fa.wav, ...
      db.emotion.categories.train.gold_standard.csv   ->  304 clips (train)
      db.emotion.categories.test.gold_standard.csv    ->  231 clips (test)

Each CSV has the columns ``file,emotion,emotion.confidence`` e.g.::

    file,emotion,emotion.confidence
    wav/03a01Fa.wav,happiness,0.9

``file`` is the clip path relative to ``root_dir`` and ``emotion`` is the label.
The two files together cover all 535 clips with **no overlap**, and contain the
full set of 7 emotions: anger, boredom, disgust, fear, happiness, neutral,
sadness.

Split policy
------------
* ``test``  = the predefined ``...test.gold_standard.csv`` (held out, untouched).
* ``train`` = the predefined ``...train.gold_standard.csv`` ...
* ``val``   = ... with a stratified fraction carved out of it (so every emotion
  is represented in validation). This mirrors the ``train/val/test`` signature of
  ``audio_mnist.py``.

Typical usage from another file
-------------------------------
    from emodb import EmoDBDataset, get_emotion_dataloaders

    train_loader, val_loader, test_loader, ds = get_emotion_dataloaders(
        root_dir="emodb",
        batch_size=32,
        num_workers=4,
    )
    num_classes = ds.num_classes            # -> 7
    print(ds.idx_to_emotion)                # {0: 'anger', 1: 'boredom', ...}
    for waveforms, labels in train_loader:  # waveforms: (B, 1, num_samples)
        ...
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio

# Sibling helpers: embedding I/O + librosa-based audio decode (avoids the
# torchaudio.load -> TorchCodec backend, which may not be installed).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_io import embedding_path, load_pooled_embedding
from src.features.audio_io import load_audio


# Default gold-standard CSV filenames shipped with EMO-DB.
TRAIN_CSV = "db.emotion.categories.train.gold_standard.csv"
TEST_CSV = "db.emotion.categories.test.gold_standard.csv"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EmoDBDataset(Dataset):
    """EMO-DB as an emotion-classification dataset.

    Each item is ``(waveform, emotion_label)`` where ``waveform`` is a float
    tensor of shape ``(1, num_samples)`` (mono, resampled, fixed length) and
    ``emotion_label`` is an ``int`` in ``[0, num_classes)``.

    Parameters
    ----------
    root_dir:
        Path to the EMO-DB folder (the one that contains ``wav/`` and the
        ``*.gold_standard.csv`` files). The ``file`` column of the CSV is
        resolved relative to this directory.
    csv_file:
        Which gold-standard CSV to read labels from. May be an absolute path,
        or just a filename (resolved under ``root_dir``). Defaults to the train
        split.
    emotion_to_idx:
        Optional fixed ``{emotion_name -> int}`` mapping. Pass this to keep the
        label ids identical across the train and test datasets (the factory
        below does exactly that). If ``None`` the mapping is built from the
        sorted unique emotions found in ``csv_file``.
    keep_emotions:
        Optional iterable restricting which emotions to keep (e.g.
        ``["anger", "happiness", "sadness", "fear", "neutral"]`` for a 5-class
        setup). Rows with other emotions are dropped. ``None`` keeps all.
    min_confidence:
        Optional float in ``[0, 1]``; rows whose ``emotion.confidence`` is below
        this threshold are dropped. ``None`` keeps all.
    target_sample_rate:
        EMO-DB is recorded at 16 kHz, so the default is a no-op resample.
    num_samples:
        Fixed clip length in samples. Default = 3 s at ``target_sample_rate``
        (clips range ~1.2-9 s, median ~2.6 s, so most are padded, the longest
        are truncated).
    transform:
        Optional callable applied to the fixed-length ``(1, num_samples)``
        waveform (e.g. MelSpectrogram / MFCC). Ignored in ``mode="embedding"``.
    mode:
        ``"audio"`` (default) returns a waveform; ``"embedding"`` returns a
        precomputed embedding loaded from ``embedding_dir`` (mirroring the audio
        folder structure). In embedding mode the index is filtered to clips that
        actually have an embedding file, so it works with a partial extraction.
    embedding_dir:
        Root of the mirrored embedding tree for THIS dataset (e.g.
        ``embeddings/openai__whisper-base/emodb``). Required when
        ``mode="embedding"``.
    layer:
        Which layer's mean-pooled vector to return in embedding mode: an ``int``
        (``-1`` = last layer) -> ``[dim]``, or ``"all"`` -> ``[num_layers, dim]``.
    embedding_ext:
        Embedding file extension, ``"pt"`` (default) or ``"npz"``.
    """

    DEFAULT_SECONDS = 3.0

    def __init__(
        self,
        root_dir: str | Path,
        csv_file: str | Path = TRAIN_CSV,
        emotion_to_idx: Optional[Mapping[str, int]] = None,
        keep_emotions: Optional[Sequence[str]] = None,
        min_confidence: Optional[float] = None,
        target_sample_rate: int = 16_000,
        num_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
        mode: str = "audio",
        embedding_dir: Optional[str | Path] = None,
        layer: Union[int, str] = -1,
        embedding_ext: str = "pt",
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir!r}")

        self.csv_path = self._resolve_csv(csv_file)
        self.target_sample_rate = int(target_sample_rate)
        self.num_samples = (
            int(num_samples)
            if num_samples is not None
            else int(self.target_sample_rate * self.DEFAULT_SECONDS)
        )
        self.transform = transform
        self.keep_emotions = set(keep_emotions) if keep_emotions is not None else None
        self.min_confidence = min_confidence

        if mode not in ("audio", "embedding"):
            raise ValueError(f"mode must be 'audio' or 'embedding', got {mode!r}")
        self.mode = mode
        self.layer = layer
        self.embedding_ext = embedding_ext.lstrip(".")
        self.embedding_dir = Path(embedding_dir) if embedding_dir is not None else None
        self.num_missing_embeddings = 0

        # --- read (file, emotion[, confidence]) rows from the gold standard ---
        rows = self._read_rows(self.csv_path)

        # --- emotion -> contiguous integer label mapping ---
        if emotion_to_idx is not None:
            self.emotion_to_idx: dict[str, int] = dict(emotion_to_idx)
        else:
            emotions = sorted({emotion for _, emotion, _ in rows})
            self.emotion_to_idx = {name: i for i, name in enumerate(emotions)}
        self.idx_to_emotion: dict[int, str] = {
            i: name for name, i in self.emotion_to_idx.items()
        }

        # --- build the (file_path, label) index ---
        self.samples: list[tuple[Path, int]] = []
        for rel_path, emotion, _conf in rows:
            if emotion not in self.emotion_to_idx:
                # emotion filtered out via a fixed mapping that excludes it
                continue
            wav_path = (self.root_dir / rel_path).resolve()
            self.samples.append((wav_path, self.emotion_to_idx[emotion]))

        if not self.samples:
            raise RuntimeError(
                f"No usable rows from {self.csv_path!r} "
                f"(keep_emotions={self.keep_emotions}, min_confidence={self.min_confidence})"
            )

        if self.mode == "embedding":
            if self.embedding_dir is None:
                raise ValueError("embedding_dir is required when mode='embedding'")
            self._filter_to_existing_embeddings()

    # ----- embedding mode helpers ---------------------------------------- #
    def _embedding_path(self, audio_path: Path) -> Path:
        return embedding_path(audio_path, self.root_dir, self.embedding_dir, self.embedding_ext)

    def _filter_to_existing_embeddings(self) -> None:
        """Drop samples whose embedding file is missing (partial extraction)."""
        kept: list[tuple[Path, int]] = []
        for path, label in self.samples:
            if self._embedding_path(path).is_file():
                kept.append((path, label))
            else:
                self.num_missing_embeddings += 1
        if not kept:
            raise RuntimeError(
                f"No embedding files found under {self.embedding_dir!r}; "
                "run extract_dataset_embeddings.py first."
            )
        self.samples = kept

    # ----- public helpers ------------------------------------------------- #
    @property
    def num_classes(self) -> int:
        return len(self.emotion_to_idx)

    @property
    def labels(self) -> list[int]:
        """Label for every sample, in index order (handy for stratified splits)."""
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        if self.mode == "embedding":
            vec = load_pooled_embedding(
                self._embedding_path(path), layer=self.layer, ext=self.embedding_ext
            )
            return vec, label
        wav = load_audio(path, sr=self.target_sample_rate, mono=True)  # 1-D float32 @ target sr
        waveform = torch.from_numpy(wav).unsqueeze(0)  # (1, time)
        waveform = self._fix_length(waveform)          # (1, num_samples)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

    # ----- CSV parsing ---------------------------------------------------- #
    def _resolve_csv(self, csv_file: str | Path) -> Path:
        p = Path(csv_file)
        if not p.is_absolute() and not p.exists():
            p = self.root_dir / csv_file
        if not p.is_file():
            raise FileNotFoundError(f"gold-standard CSV not found: {p!r}")
        return p

    def _read_rows(self, csv_path: Path) -> list[tuple[str, str, Optional[float]]]:
        rows: list[tuple[str, str, Optional[float]]] = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                emotion = row["emotion"].strip()
                if self.keep_emotions is not None and emotion not in self.keep_emotions:
                    continue
                conf_raw = row.get("emotion.confidence")
                conf = float(conf_raw) if conf_raw not in (None, "") else None
                if (
                    self.min_confidence is not None
                    and conf is not None
                    and conf < self.min_confidence
                ):
                    continue
                rows.append((row["file"].strip(), emotion, conf))
        return rows

    # ----- internal audio ops -------------------------------------------- #
    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.target_sample_rate
            )
        return waveform

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[1]
        if length > self.num_samples:                  # truncate
            waveform = waveform[:, : self.num_samples]
        elif length < self.num_samples:                # right-pad with zeros
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - length))
        return waveform


# --------------------------------------------------------------------------- #
# Splitting helper (per-emotion / stratified)
# --------------------------------------------------------------------------- #
def _stratified_split(
    labels: Sequence[int],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Per-emotion split so EVERY emotion appears in both train and val."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(idx)

    train_idx, val_idx = [], []
    for idxs in by_label.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_split))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _random_split(n_items: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idxs = list(range(n_items))
    rng.shuffle(idxs)
    n_val = int(round(n_items * val_split))
    return idxs[n_val:], idxs[:n_val]


def _build_shared_mapping(
    root_dir: Path,
    csv_files: Sequence[str | Path],
    keep_emotions: Optional[Sequence[str]],
) -> dict[str, int]:
    """Union of emotions across the given CSVs -> stable sorted int mapping.

    Building it from every split guarantees train and test share identical
    label ids even if some emotion happened to be missing from one file.
    """
    emotions: set[str] = set()
    keep = set(keep_emotions) if keep_emotions is not None else None
    for csv_file in csv_files:
        p = Path(csv_file)
        if not p.is_absolute() and not p.exists():
            p = root_dir / csv_file
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                emotion = row["emotion"].strip()
                if keep is None or emotion in keep:
                    emotions.add(emotion)
    return {name: i for i, name in enumerate(sorted(emotions))}


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def get_emotion_dataloaders(
    root_dir: str | Path,
    train_csv: str | Path = TRAIN_CSV,
    test_csv: str | Path = TEST_CSV,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    target_sample_rate: int = 16_000,
    num_samples: Optional[int] = None,
    transform: Optional[Callable] = None,
    keep_emotions: Optional[Sequence[str]] = None,
    min_confidence: Optional[float] = None,
    mode: str = "audio",
    embedding_dir: Optional[str | Path] = None,
    layer: Union[int, str] = -1,
    embedding_ext: str = "pt",
    seed: int = 42,
    stratified: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, EmoDBDataset]:
    """Build train/val/test ``DataLoader``s for EMO-DB emotion classification.

    The train/test split is the one EMO-DB ships (``train_csv`` / ``test_csv``);
    ``val_split`` is carved out of the *train* CSV. Set ``val_split=0.0`` to skip
    validation (its loader will be empty).

    Returns
    -------
    (train_loader, val_loader, test_loader, dataset)
        ``dataset`` is the full train ``EmoDBDataset`` (before the val carve-out);
        read ``dataset.num_classes`` and ``dataset.idx_to_emotion`` from it. The
        label ids are shared with the test dataset.
    """
    root_dir = Path(root_dir)

    # Shared label mapping so train & test ids line up (covers all 7 emotions).
    emotion_to_idx = _build_shared_mapping(root_dir, [train_csv, test_csv], keep_emotions)

    common = dict(
        emotion_to_idx=emotion_to_idx,
        keep_emotions=keep_emotions,
        min_confidence=min_confidence,
        target_sample_rate=target_sample_rate,
        num_samples=num_samples,
        transform=transform,
        mode=mode,
        embedding_dir=embedding_dir,
        layer=layer,
        embedding_ext=embedding_ext,
    )
    train_ds = EmoDBDataset(root_dir, csv_file=train_csv, **common)
    test_ds = EmoDBDataset(root_dir, csv_file=test_csv, **common)

    if stratified:
        train_idx, val_idx = _stratified_split(train_ds.labels, val_split, seed)
    else:
        train_idx, val_idx = _random_split(len(train_ds), val_split, seed)

    def _loader(dataset, shuffle: bool, drop: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop,
        )

    train_loader = _loader(Subset(train_ds, train_idx), shuffle=True, drop=drop_last)
    val_loader = _loader(Subset(train_ds, val_idx), shuffle=False, drop=False)
    test_loader = _loader(test_ds, shuffle=False, drop=False)
    return train_loader, val_loader, test_loader, train_ds


# --------------------------------------------------------------------------- #
# Self-test: run `python emodb.py` to verify on synthetic data.
# (Generates a tiny fake EMO-DB layout - wavs + gold-standard CSVs - so you can
#  confirm the pipeline works end-to-end without the real dataset.)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    EMOTIONS = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "emodb"
        (root / "wav").mkdir(parents=True)
        sr = 16_000

        def _make_split(csv_name: str, per_emotion: int, tag: str):
            rows = [("file", "emotion", "emotion.confidence")]
            for emo in EMOTIONS:
                for i in range(per_emotion):
                    fname = f"{tag}_{emo}_{i}.wav"
                    dur = int(sr * (1.0 + 2.0 * random.random()))   # 1-3 s
                    torchaudio.save(str(root / "wav" / fname), torch.randn(1, dur) * 0.1, sr)
                    rows.append((f"wav/{fname}", emo, "0.9"))
            with open(root / csv_name, "w", newline="") as fh:
                csv.writer(fh).writerows(rows)

        _make_split(TRAIN_CSV, per_emotion=6, tag="tr")
        _make_split(TEST_CSV, per_emotion=3, tag="te")

        train, val, test, ds = get_emotion_dataloaders(
            root_dir=root, batch_size=8, num_workers=0
        )
        print(f"num_classes      : {ds.num_classes}")
        print(f"idx_to_emotion   : {ds.idx_to_emotion}")
        print(f"train / val / test clips : "
              f"{len(train.dataset)} / {len(val.dataset)} / {len(test.dataset)}")
        xb, yb = next(iter(train))
        print(f"batch waveforms  : {tuple(xb.shape)}  (B, 1, num_samples)")
        print(f"batch labels     : {yb.tolist()}")
        print("OK -- pipeline runs end-to-end.")
