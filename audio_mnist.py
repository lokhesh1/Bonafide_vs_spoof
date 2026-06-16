"""
audiomnist_dataset.py
=====================

PyTorch ``Dataset`` + ``DataLoader`` utilities for **speaker classification** on the
AudioMNIST dataset (https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist).

The task here is SPEAKER identification, NOT digit recognition:
    input  -> audio waveform
    target -> speaker id in [0 .. num_speakers-1], derived from the folder name.

Expected directory layout (speaker id == folder name)::

    root/
      01/  ->  *.wav
      02/  ->  *.wav
      ...
      60/  ->  *.wav

Typical usage from another file
-------------------------------
    from audiomnist_dataset import AudioMNISTSpeakerDataset, get_speaker_dataloaders

    train_loader, val_loader, test_loader, ds = get_speaker_dataloaders(
        root_dir="data",
        batch_size=64,
        num_workers=4,
    )
    num_classes = ds.num_speakers          # -> 60
    for waveforms, labels in train_loader: # waveforms: (B, 1, num_samples)
        ...
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio

# Sibling helper for the precomputed-embedding mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_io import embedding_path, load_pooled_embedding


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class AudioMNISTSpeakerDataset(Dataset):
    """AudioMNIST as a speaker-classification dataset.

    Each item is ``(waveform, speaker_label)`` where ``waveform`` is a float
    tensor of shape ``(1, num_samples)`` (mono, resampled, fixed length) and
    ``speaker_label`` is an ``int`` in ``[0, num_speakers)``.

    Parameters
    ----------
    root_dir:
        Path to the folder that directly contains the speaker subfolders
        (``01``, ``02``, ...).
    target_sample_rate:
        16 kHz keeps most speaker-relevant detail while staying light.
    num_samples:
        Default = 1 second at ``target_sample_rate``. Most
        AudioMNIST clips are < 1 s, so they get padded rather than cut.
    transform:
        Optional callable applied to the fixed-length ``(1, num_samples)``
        waveform (e.g. MelSpectrogram / MFCC). Ignored in ``mode="embedding"``.
    file_ext:
        Audio file extension to glob for (default ``"wav"``).
    mode:
        ``"audio"`` (default) returns a waveform; ``"embedding"`` returns a
        precomputed embedding loaded from ``embedding_dir`` (mirroring the audio
        folder structure). In embedding mode the index is filtered to clips that
        actually have an embedding file, so it works with a partial extraction.
    embedding_dir:
        Root of the mirrored embedding tree for THIS dataset (e.g.
        ``embeddings/openai__whisper-base/data``). Required when
        ``mode="embedding"``.
    layer:
        Which layer's mean-pooled vector to return in embedding mode: an ``int``
        (negative indexes from the top, ``-1`` = last layer) -> ``[dim]``, or
        ``"all"`` -> ``[num_layers, dim]``.
    embedding_ext:
        Embedding file extension, ``"pt"`` (default) or ``"npz"``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_sample_rate: int = 16_000,
        num_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
        file_ext: str = "wav",
        mode: str = "audio",
        embedding_dir: Optional[str | Path] = None,
        layer: Union[int, str] = -1,
        embedding_ext: str = "pt",
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir!r}")

        self.target_sample_rate = int(target_sample_rate)
        self.num_samples = int(num_samples) if num_samples is not None else self.target_sample_rate
        self.transform = transform
        self.file_ext = file_ext.lstrip(".")

        if mode not in ("audio", "embedding"):
            raise ValueError(f"mode must be 'audio' or 'embedding', got {mode!r}")
        self.mode = mode
        self.layer = layer
        self.embedding_ext = embedding_ext.lstrip(".")
        self.embedding_dir = Path(embedding_dir) if embedding_dir is not None else None
        self.num_missing_embeddings = 0

        # --- discover speaker folders and assign contiguous integer labels ---
        speaker_dirs = sorted(d for d in self.root_dir.iterdir() if d.is_dir())
        if not speaker_dirs:
            raise RuntimeError(f"No speaker subfolders found under {self.root_dir!r}")

        self.speaker_to_idx: dict[str, int] = {d.name: i for i, d in enumerate(speaker_dirs)}
        self.idx_to_speaker: dict[int, str] = {i: name for name, i in self.speaker_to_idx.items()}

        # --- build the (file_path, label) index ---
        self.samples: list[tuple[Path, int]] = []
        for d in speaker_dirs:
            label = self.speaker_to_idx[d.name]
            for f in sorted(d.glob(f"*.{self.file_ext}")):
                self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(
                f"No *.{self.file_ext} files found under {self.root_dir!r}"
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
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)

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
        waveform, sr = torchaudio.load(str(path))      # (channels, time)
        waveform = self._to_mono(waveform)
        waveform = self._resample(waveform, sr)
        waveform = self._fix_length(waveform)          # (1, num_samples)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

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
# Splitting helpers
# --------------------------------------------------------------------------- #
def _stratified_split(
    labels: Sequence[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Per-speaker split so EVERY speaker appears in train/val/test.

    This is the correct setup for closed-set speaker identification: the model
    must see each of the 60 speakers during training and be tested on held-out
    clips from those same speakers.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for idxs in by_label.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        test_idx += idxs[:n_test]
        val_idx += idxs[n_test : n_test + n_val]
        train_idx += idxs[n_test + n_val :]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _random_split(
    n_items: int,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    idxs = list(range(n_items))
    rng.shuffle(idxs)
    n_test = int(round(n_items * test_split))
    n_val = int(round(n_items * val_split))
    return idxs[n_test + n_val :], idxs[n_test : n_test + n_val], idxs[:n_test]


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def get_speaker_dataloaders(
    root_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    target_sample_rate: int = 16_000,
    num_samples: Optional[int] = None,
    transform: Optional[Callable] = None,
    mode: str = "audio",
    embedding_dir: Optional[str | Path] = None,
    layer: Union[int, str] = -1,
    embedding_ext: str = "pt",
    seed: int = 42,
    stratified: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, AudioMNISTSpeakerDataset]:
    """Build train/val/test ``DataLoader``s for speaker classification.

    Returns
    -------
    (train_loader, val_loader, test_loader, dataset)
        The underlying ``dataset`` is returned too so callers can read
        ``dataset.num_speakers`` and ``dataset.idx_to_speaker``.

    Notes
    -----
    * ``val_split`` / ``test_split`` are fractions of the full dataset; the
      remainder is the training set. Set either to ``0.0`` to skip that split
      (the corresponding loader will be empty).
    * With ``stratified=True`` (default) the split is done per speaker, so all
      speakers appear in every split. This is what you want for classifying
      among a fixed set of known speakers.
    """
    dataset = AudioMNISTSpeakerDataset(
        root_dir=root_dir,
        target_sample_rate=target_sample_rate,
        num_samples=num_samples,
        transform=transform,
        mode=mode,
        embedding_dir=embedding_dir,
        layer=layer,
        embedding_ext=embedding_ext,
    )

    if stratified:
        train_idx, val_idx, test_idx = _stratified_split(
            dataset.labels, val_split, test_split, seed
        )
    else:
        train_idx, val_idx, test_idx = _random_split(
            len(dataset), val_split, test_split, seed
        )

    def _loader(indices: list[int], shuffle: bool, drop: bool) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop,
        )

    train_loader = _loader(train_idx, shuffle=True, drop=drop_last)
    val_loader = _loader(val_idx, shuffle=False, drop=False)
    test_loader = _loader(test_idx, shuffle=False, drop=False)
    return train_loader, val_loader, test_loader, dataset


# --------------------------------------------------------------------------- #
# Self-test: run `python audiomnist_dataset.py` to verify on synthetic data.
# (Generates a tiny fake dataset so you can confirm the pipeline works
#  end-to-end without downloading the real one.)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        sr = 48_000
        for spk in ("01", "02", "03"):                 # 3 fake speakers
            (root / spk).mkdir(parents=True)
            for i in range(8):                          # 8 clips each
                dur = int(sr * (0.4 + 0.4 * random.random()))   # variable length
                wav = torch.randn(1, dur) * 0.1
                torchaudio.save(str(root / spk / f"{i}_{spk}_0.wav"), wav, sr)

        train, val, test, ds = get_speaker_dataloaders(
            root_dir=root, batch_size=4, num_workers=0
        )
        print(f"num_speakers      : {ds.num_speakers}")
        print(f"idx_to_speaker    : {ds.idx_to_speaker}")
        print(f"total / train / val / test : "
              f"{len(ds)} / {len(train.dataset)} / {len(val.dataset)} / {len(test.dataset)}")
        xb, yb = next(iter(train))
        print(f"batch waveforms   : {tuple(xb.shape)}  (B, 1, num_samples)")
        print(f"batch labels      : {yb.tolist()}")
        print("OK -- pipeline runs end-to-end.")