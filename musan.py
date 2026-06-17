"""
musan.py
========

PyTorch ``Dataset`` + ``DataLoader`` utilities for the **MUSAN** corpus
(MUsic, SPeech And Noise).

This loader supports **three classification tasks**, selected via the ``task``
config argument (this is the knob the caller flips):

* ``task="source"``      -> classify **music vs. speech vs. noise** (3 classes).
                            Labels come from the top-level folder name.
* ``task="music_type"``  -> classify the **type of music** (5 classes), using the
                            sub-folders under ``music/`` as labels:
                            ``fma, fma-western-art, hd-classical, jamendo, rfm``.
* ``task="music_genre"`` -> classify the **musical genre**, with labels read from
                            the per-source ``ANNOTATIONS`` files (e.g. ``pop``,
                            ``jazz``, ``westernart``, ``blues``, ...).

The first two derive labels purely from folder names; ``music_genre`` instead
parses each ``music/<source>/ANNOTATIONS`` file (one ``<clip-id> <genre> ...``
line per clip, ``<clip-id>`` == wav filename stem). Annotation tags can list
several genres (e.g. ``pop,electronica``); ``genre_from="primary"`` (default)
keeps the first, ``genre_from="full"`` keeps the whole tag as one class. Music
clips lacking an annotation entry are skipped.

Expected directory layout::

    musan/
      music/
        fma/             ->  *.wav  + ANNOTATIONS
        fma-western-art/ ->  *.wav  + ANNOTATIONS
        hd-classical/    ->  *.wav  + ANNOTATIONS
        jamendo/         ->  *.wav  + ANNOTATIONS
        rfm/             ->  *.wav  + ANNOTATIONS
      speech/
        librivox/        ->  *.wav
        us-gov/          ->  *.wav
      noise/
        free-sound/      ->  *.wav
        sound-bible/     ->  *.wav

A note on clip length
----------------------
MUSAN clips are long (music/speech often run for *minutes*). Truncating every
file from the start would throw away almost all the audio, so this loader takes
a fixed-length **window** out of each file: a *random* window for the training
split (data augmentation / coverage) and a deterministic *centre* window for the
val/test splits (reproducibility). Window length is set by ``num_samples``
(default 4 s).

Typical usage from another file
-------------------------------
    from musan import MusanDataset, get_musan_dataloaders

    # Task 1: music / speech / noise
    tr, va, te, ds = get_musan_dataloaders(root_dir="musan", task="source")

    # Task 2: type of music
    tr, va, te, ds = get_musan_dataloaders(root_dir="musan", task="music_type")

    # Task 3: musical genre (labels from ANNOTATIONS files)
    tr, va, te, ds = get_musan_dataloaders(root_dir="musan", task="music_genre")

    num_classes = ds.num_classes            # 3 / 5 / #genres depending on task
    print(ds.idx_to_class)
    for waveforms, labels in tr:            # waveforms: (B, 1, num_samples)
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

# Sibling helpers: embedding I/O + librosa-based audio decode (avoids the
# torchaudio.load -> TorchCodec backend, which may not be installed).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_io import embedding_path, load_pooled_embedding
from src.features.audio_io import load_audio


# Recognised values for the ``task`` config argument.
TASK_SOURCE = "source"          # music vs speech vs noise
TASK_MUSIC_TYPE = "music_type"  # type of music (music/ sub-folders)
TASK_MUSIC_GENRE = "music_genre"  # musical genre (from music/*/ANNOTATIONS)
VALID_TASKS = (TASK_SOURCE, TASK_MUSIC_TYPE, TASK_MUSIC_GENRE)

# Top-level source folders used by the "source" task.
SOURCE_DIRS = ("music", "noise", "speech")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MusanDataset(Dataset):
    """MUSAN as a classification dataset, configurable via ``task``.

    Each item is ``(waveform, label)`` where ``waveform`` is a float tensor of
    shape ``(1, num_samples)`` (mono, resampled, fixed-length window) and
    ``label`` is an ``int`` in ``[0, num_classes)``.

    Parameters
    ----------
    root_dir:
        Path to the MUSAN folder (the one containing ``music/``, ``speech/``,
        ``noise/``).
    task:
        ``"source"`` (music/speech/noise), ``"music_type"`` (music sub-folders),
        or ``"music_genre"`` (musical genre read from the per-source
        ``ANNOTATIONS`` files).
    genre_from:
        Only used when ``task="music_genre"``. ``"primary"`` (default) labels a
        clip by the first genre tag (annotations may list several, e.g.
        ``"pop,electronica"`` -> ``pop``), giving a compact single-label set.
        ``"full"`` uses the whole comma-joined tag as one class (many sparse
        classes). Music clips with no annotation entry are skipped.
    target_sample_rate:
        MUSAN is 16 kHz, so the default is a no-op resample.
    num_samples:
        Length of the window taken from each clip, in samples. Default = 4 s at
        ``target_sample_rate``. Clips shorter than this are zero-padded.
    random_crop:
        If ``True``, take a *random* window from each clip on every access, so a
        clip contributes different segments across epochs (use for the training
        split). If ``False``, take a deterministic centre window (use for
        val/test). The factory sets this per split. Reproducibility comes from
        PyTorch seeding each DataLoader worker's RNG; set a global seed via
        ``torch.manual_seed`` for fully repeatable runs.
    transform:
        Optional callable applied to the fixed-length ``(1, num_samples)``
        waveform (e.g. MelSpectrogram / MFCC). Ignored in ``mode="embedding"``.
    file_ext:
        Audio file extension to glob for (default ``"wav"``).
    mode:
        ``"audio"`` (default) returns a waveform; ``"embedding"`` returns a
        precomputed embedding loaded from ``embedding_dir`` (mirroring the audio
        folder structure). In embedding mode the index is filtered to clips that
        actually have an embedding file, so it works with a partial extraction
        (and ``random_crop`` is irrelevant, the window was fixed at extraction).
    embedding_dir:
        Root of the mirrored embedding tree for THIS dataset (e.g.
        ``embeddings/openai__whisper-base/musan``). Required when
        ``mode="embedding"``.
    layer:
        Which layer's mean-pooled vector to return in embedding mode: an ``int``
        (``-1`` = last layer) -> ``[dim]``, or ``"all"`` -> ``[num_layers, dim]``.
    embedding_ext:
        Embedding file extension, ``"pt"`` (default) or ``"npz"``.
    """

    DEFAULT_SECONDS = 4.0

    def __init__(
        self,
        root_dir: str | Path,
        task: str = TASK_SOURCE,
        genre_from: str = "primary",
        target_sample_rate: int = 16_000,
        num_samples: Optional[int] = None,
        random_crop: bool = False,
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

        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
        self.task = task

        if genre_from not in ("primary", "full"):
            raise ValueError(f"genre_from must be 'primary' or 'full', got {genre_from!r}")
        self.genre_from = genre_from
        self.num_unlabeled = 0  # music clips skipped for lacking an annotation

        self.target_sample_rate = int(target_sample_rate)
        self.num_samples = (
            int(num_samples)
            if num_samples is not None
            else int(self.target_sample_rate * self.DEFAULT_SECONDS)
        )
        self.random_crop = random_crop
        self.transform = transform
        self.file_ext = file_ext.lstrip(".")

        if mode not in ("audio", "embedding"):
            raise ValueError(f"mode must be 'audio' or 'embedding', got {mode!r}")
        self.mode = mode
        self.layer = layer
        self.embedding_ext = embedding_ext.lstrip(".")
        self.embedding_dir = Path(embedding_dir) if embedding_dir is not None else None
        self.num_missing_embeddings = 0

        # --- build class mapping + (file_path, label) index for the task -----
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self.samples: list[tuple[Path, int]] = []
        if self.task == TASK_MUSIC_GENRE:
            self._build_genre_index()
        else:
            self._build_folder_index()

        if not self.samples:
            raise RuntimeError(
                f"No *.{self.file_ext} files found for task={self.task!r} under {self.root_dir!r}"
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
        return len(self.class_to_idx)

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

    # ----- task -> (file, label) index ------------------------------------ #
    def _build_folder_index(self) -> None:
        """Label = folder name. Used by ``source`` and ``music_type`` tasks."""
        class_dirs = self._discover_class_dirs()
        if not class_dirs:
            raise RuntimeError(
                f"No class folders found for task={self.task!r} under {self.root_dir!r}"
            )
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        for d in class_dirs:
            label = self.class_to_idx[d.name]
            for f in sorted(d.rglob(f"*.{self.file_ext}")):  # recurse: sources nest
                self.samples.append((f, label))

    def _discover_class_dirs(self) -> list[Path]:
        if self.task == TASK_SOURCE:
            dirs = [self.root_dir / name for name in SOURCE_DIRS]
            return sorted(d for d in dirs if d.is_dir())
        # TASK_MUSIC_TYPE: sub-folders directly under music/
        music_dir = self.root_dir / "music"
        if not music_dir.is_dir():
            raise FileNotFoundError(f"music/ folder not found under {self.root_dir!r}")
        return sorted(d for d in music_dir.iterdir() if d.is_dir())

    def _build_genre_index(self) -> None:
        """Label = musical genre read from the ``music/*/ANNOTATIONS`` files.

        Each annotation line is ``<clip-id> <genre[,genre...]> <vocals> ...`` and
        ``<clip-id>`` matches the wav filename stem. Clips without an annotation
        entry are skipped (counted in ``self.num_unlabeled``).
        """
        music_dir = self.root_dir / "music"
        if not music_dir.is_dir():
            raise FileNotFoundError(f"music/ folder not found under {self.root_dir!r}")

        # clip-stem -> genre tag, gathered from every ANNOTATIONS file under music/
        stem_to_genre: dict[str, str] = {}
        for ann in sorted(music_dir.rglob("ANNOTATIONS")):
            stem_to_genre.update(self._parse_annotations(ann))

        # resolve each wav to a genre label, skipping unannotated clips
        pairs: list[tuple[Path, str]] = []
        for f in sorted(music_dir.rglob(f"*.{self.file_ext}")):
            genre = stem_to_genre.get(f.stem)
            if genre is None:
                self.num_unlabeled += 1
                continue
            name = genre.split(",")[0] if self.genre_from == "primary" else genre
            pairs.append((f, name))

        if not pairs:
            raise RuntimeError(
                f"No annotated music clips found under {music_dir!r}; "
                f"are the ANNOTATIONS files present?"
            )

        classes = sorted({name for _, name in pairs})
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        self.samples = [(f, self.class_to_idx[name]) for f, name in pairs]

    @staticmethod
    def _parse_annotations(path: Path) -> dict[str, str]:
        """Return ``{clip-stem: genre-tag}`` from one ANNOTATIONS file."""
        out: dict[str, str] = {}
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0]] = parts[1]
        return out

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
        """Take a ``num_samples`` window; random for train, centre otherwise."""
        length = waveform.shape[1]
        if length < self.num_samples:                  # right-pad short clips
            return torch.nn.functional.pad(waveform, (0, self.num_samples - length))
        if length == self.num_samples:
            return waveform

        max_start = length - self.num_samples
        if self.random_crop:
            start = random.randint(0, max_start)        # varies per access/epoch
        else:
            start = max_start // 2                      # centre window
        return waveform[:, start : start + self.num_samples]


# --------------------------------------------------------------------------- #
# Splitting helpers
# --------------------------------------------------------------------------- #
def _stratified_split(
    labels: Sequence[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Per-class split so EVERY class appears in train/val/test."""
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
def get_musan_dataloaders(
    root_dir: str | Path,
    task: str = TASK_SOURCE,
    genre_from: str = "primary",
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
) -> tuple[DataLoader, DataLoader, DataLoader, MusanDataset]:
    """Build train/val/test ``DataLoader``s for a MUSAN classification task.

    Parameters
    ----------
    task:
        ``"source"`` -> music/speech/noise (3 classes); ``"music_type"`` ->
        type of music from the ``music/`` sub-folders (5 classes);
        ``"music_genre"`` -> musical genre from the ``ANNOTATIONS`` files. This
        is the config switch that picks the task.
    genre_from:
        For ``task="music_genre"`` only: ``"primary"`` (first tag) or ``"full"``
        (whole comma-joined tag). See :class:`MusanDataset`.
    val_split / test_split:
        Fractions of the full dataset; the remainder is training data. Set
        either to ``0.0`` to skip that split. With ``stratified=True`` (default)
        the split is per class, so every class appears in every split.

    Returns
    -------
    (train_loader, val_loader, test_loader, dataset)
        ``dataset`` (eval-mode, centre-crop) is returned so callers can read
        ``dataset.num_classes`` and ``dataset.idx_to_class``.

    Notes
    -----
    The training split uses random cropping; val/test use a deterministic centre
    crop. Both share the same deterministic file ordering, so the index split is
    consistent between them.
    """
    common = dict(
        root_dir=root_dir,
        task=task,
        genre_from=genre_from,
        target_sample_rate=target_sample_rate,
        num_samples=num_samples,
        transform=transform,
        mode=mode,
        embedding_dir=embedding_dir,
        layer=layer,
        embedding_ext=embedding_ext,
    )
    # Two views over the SAME (sorted, deterministic) file list: one that random-
    # crops for training, one that centre-crops for evaluation.
    train_ds = MusanDataset(**common, random_crop=True)
    eval_ds = MusanDataset(**common, random_crop=False)

    if stratified:
        train_idx, val_idx, test_idx = _stratified_split(
            eval_ds.labels, val_split, test_split, seed
        )
    else:
        train_idx, val_idx, test_idx = _random_split(
            len(eval_ds), val_split, test_split, seed
        )

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
    val_loader = _loader(Subset(eval_ds, val_idx), shuffle=False, drop=False)
    test_loader = _loader(Subset(eval_ds, test_idx), shuffle=False, drop=False)
    return train_loader, val_loader, test_loader, eval_ds


# --------------------------------------------------------------------------- #
# Self-test: run `python musan.py` to verify both tasks on synthetic data.
# (Generates a tiny fake MUSAN layout so you can confirm the pipeline works
#  end-to-end without the real corpus.)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    LAYOUT = {
        "music": ["fma", "fma-western-art", "hd-classical", "jamendo", "rfm"],
        "speech": ["librivox", "us-gov"],
        "noise": ["free-sound", "sound-bible"],
    }

    # fake multi-genre tags to exercise the ANNOTATIONS path (primary genre wins)
    FAKE_GENRES = ["pop,electronica", "jazz", "westernart,baroque", "blues", "rock,folk", "hiphop,rap"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "musan"
        sr = 16_000
        for source, subdirs in LAYOUT.items():
            for sub in subdirs:
                d = root / source / sub
                d.mkdir(parents=True)
                ann_lines = []
                for i in range(6):
                    stem = f"{source}-{sub}-{i}"
                    dur = int(sr * (2.0 + 6.0 * random.random()))   # 2-8 s
                    torchaudio.save(str(d / f"{stem}.wav"), torch.randn(1, dur) * 0.1, sr)
                    if source == "music" and i < 5:                 # leave 1 clip unannotated
                        ann_lines.append(f"{stem} {FAKE_GENRES[i % len(FAKE_GENRES)]} N Artist")
                if source == "music":
                    (d / "ANNOTATIONS").write_text("\n".join(ann_lines) + "\n")

        for task in (TASK_SOURCE, TASK_MUSIC_TYPE, TASK_MUSIC_GENRE):
            print(f"\n=== task = {task!r} ===")
            train, val, test, ds = get_musan_dataloaders(
                root_dir=root, task=task, batch_size=8, num_workers=0
            )
            print(f"num_classes   : {ds.num_classes}")
            print(f"idx_to_class  : {ds.idx_to_class}")
            if task == TASK_MUSIC_GENRE:
                print(f"skipped (unannotated): {ds.num_unlabeled}")
            print(f"train / val / test : "
                  f"{len(train.dataset)} / {len(val.dataset)} / {len(test.dataset)}")
            xb, yb = next(iter(train))
            print(f"batch waveforms : {tuple(xb.shape)}  (B, 1, num_samples)")
            print(f"batch labels    : {yb.tolist()}")
        print("\nOK -- pipeline runs end-to-end for all tasks.")
