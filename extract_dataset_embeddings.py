"""
extract_dataset_embeddings.py
=============================

Drive :class:`SpeechEmbeddingExtractor` over a whole dataset (AudioMNIST /
EMO-DB / MUSAN), for a chosen **percentage** of the clips, and save one
mean-pooled per-layer embedding per clip into a folder tree that **mirrors the
source audio**.

On-disk layout::

    <out>/<model-slug>/<dataset-root-name>/<same sub-path as the .wav>.pt

e.g. ``data/01/0_01_0.wav``  ->  ``embeddings/openai__whisper-base/data/01/0_01_0.pt``

The dataset loaders read these back in ``mode="embedding"`` by mirroring the
audio path the same way (see ``embedding_io.embedding_path``); point their
``embedding_dir`` at ``<out>/<model-slug>/<dataset-root-name>``.

Percentage sampling is **stratified per class** (every class keeps at least one
clip even at tiny percentages) and seed-controlled. EMO-DB ships a fixed
train/test split (two CSVs), so it takes **separate** ``--train-percent`` /
``--test-percent`` knobs; both splits are written into the same mirrored
``wav/`` tree (the split itself stays defined by the CSVs at probe time).

Windowing reuses each loader's own fixed-length pipeline, so the embedding for a
clip is computed from exactly the same audio the audio-mode loader would yield.
MUSAN clips run for minutes, so they default to a single **25 s centre window**
(one embedding per file, 1:1 with the source); override with ``--seconds``.

Examples
--------
    # 10% of AudioMNIST with Whisper
    python extract_dataset_embeddings.py --dataset audio_mnist --root data \
        --model openai/whisper-base --percent 10 --out embeddings

    # EMO-DB: 80% of train, 100% of test
    python extract_dataset_embeddings.py --dataset emodb --root emodb \
        --model openai/whisper-base --train-percent 80 --test-percent 100 \
        --out embeddings

    # MUSAN source task, 5%, 25 s window (default) with XLS-R
    python extract_dataset_embeddings.py --dataset musan --root musan \
        --model facebook/wav2vec2-xls-r-300m --task source --percent 5 \
        --out embeddings

Dependencies (for an actual run): torch, transformers, librosa  (plus this
repo's dataset loaders). The module itself imports torch lazily, so
``--help`` and the pure helpers work without it.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# Make sibling modules importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_io import embedding_path

VALID_DATASETS = ("audio_mnist", "emodb", "musan")
SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- #
# Pure helpers (no torch needed)
# --------------------------------------------------------------------------- #
def model_slug(model_name: str) -> str:
    """Filesystem-safe folder name for a HF model id (``a/b`` -> ``a__b``)."""
    return model_name.replace("/", "__").replace(" ", "_")


def stratified_subset(labels: Sequence[int], percent: float, seed: int) -> List[int]:
    """Indices for a per-class ``percent``% sample (>=1 per class), sorted.

    ``percent >= 100`` keeps everything. Sorting keeps a deterministic file
    order so re-runs (and resume/skip) line up.
    """
    n = len(labels)
    if percent >= 100:
        return list(range(n))
    if percent <= 0:
        return []
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    keep: list[int] = []
    for idxs in by_label.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = max(1, round(len(idxs) * percent / 100.0))
        keep.extend(idxs[:k])
    keep.sort()
    return keep


def parse_layers(value: Optional[str]):
    """``"all"``/``None`` -> ``"all"``; else a list of ints from a CSV string."""
    if value is None or value.lower() == "all":
        return "all"
    return [int(x) for x in value.split(",")]


def num_samples_for(dataset: str, seconds: Optional[float]) -> Optional[int]:
    """Window length in samples. ``None`` -> use the loader's own default.

    MUSAN clips are minutes long, so default to a 25 s window when unspecified;
    AudioMNIST (~1 s) and EMO-DB (~3 s) fall back to their loader defaults.
    """
    if seconds is None:
        seconds = 25.0 if dataset == "musan" else None
    return int(seconds * SAMPLE_RATE) if seconds is not None else None


# --------------------------------------------------------------------------- #
# Dataset construction (audio mode) -> list of (dataset, indices, out_root)
# --------------------------------------------------------------------------- #
def _build_jobs(args, dataset_out: Path):
    """Return a list of (audio_dataset, selected_indices, split_name) jobs.

    EMO-DB yields two jobs (train/test, separate percentages); the others one.
    """
    num_samples = num_samples_for(args.dataset, args.seconds)

    if args.dataset == "audio_mnist":
        from audio_mnist import AudioMNISTSpeakerDataset

        ds = AudioMNISTSpeakerDataset(
            root_dir=args.root, target_sample_rate=SAMPLE_RATE, num_samples=num_samples
        )
        idx = stratified_subset(ds.labels, args.percent, args.seed)
        return [(ds, idx, "all")]

    if args.dataset == "emodb":
        from emodb import EmoDBDataset, TRAIN_CSV, TEST_CSV, _build_shared_mapping

        root = Path(args.root)
        mapping = _build_shared_mapping(root, [TRAIN_CSV, TEST_CSV], None)
        common = dict(
            emotion_to_idx=mapping,
            target_sample_rate=SAMPLE_RATE,
            num_samples=num_samples,
        )
        train_ds = EmoDBDataset(root, csv_file=TRAIN_CSV, **common)
        test_ds = EmoDBDataset(root, csv_file=TEST_CSV, **common)
        train_idx = stratified_subset(train_ds.labels, args.train_percent, args.seed)
        test_idx = stratified_subset(test_ds.labels, args.test_percent, args.seed)
        return [(train_ds, train_idx, "train"), (test_ds, test_idx, "test")]

    # musan
    from musan import MusanDataset

    ds = MusanDataset(
        root_dir=args.root,
        task=args.task,
        genre_from=args.genre_from,
        target_sample_rate=SAMPLE_RATE,
        num_samples=num_samples,
        random_crop=False,  # deterministic centre window for extraction
    )
    idx = stratified_subset(ds.labels, args.percent, args.seed)
    return [(ds, idx, args.task)]


# --------------------------------------------------------------------------- #
# Main extraction loop
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"--root not found: {root!r}")

    dataset_out = Path(args.out) / model_slug(args.model) / root.name
    layers = parse_layers(args.layers)

    # Heavy deps imported only now (so --help / tests work without torch).
    from speech_embedding_extractor import SpeechEmbeddingExtractor

    print(f"loading model: {args.model}")
    extractor = SpeechEmbeddingExtractor(
        args.model, device=args.device, trust_remote_code=args.trust_remote_code
    )

    jobs = _build_jobs(args, dataset_out)
    total = sum(len(idx) for _, idx, _ in jobs)
    print(
        f"dataset={args.dataset} root={root} -> out={dataset_out}\n"
        f"selecting {total} clip(s); window="
        f"{'loader-default' if args.seconds is None and args.dataset != 'musan' else (args.seconds or 25.0)} s; "
        f"layers={layers}; format={args.format}"
    )

    done = skipped = failed = 0
    for ds, indices, split in jobs:
        if not indices:
            print(f"  [{split}] nothing selected, skipping")
            continue
        print(f"  [{split}] {len(indices)} clip(s)")
        for n, di in enumerate(indices, 1):
            audio_path = Path(ds.samples[di][0])
            save_path = embedding_path(audio_path, ds.root_dir, dataset_out, args.format)
            if save_path.exists() and not args.overwrite:
                skipped += 1
            else:
                try:
                    waveform, _label = ds[di]  # (1, num_samples) tensor, windowed
                    extractor.extract(
                        waveform,
                        layers=layers,
                        sampling_rate=SAMPLE_RATE,
                        pooling="mean",
                        include_raw=False,  # store only the pooled [dim] vectors
                        save_path=save_path,
                        save_format=args.format,
                    )
                    done += 1
                except Exception as exc:  # keep going; report at the end
                    failed += 1
                    print(f"    FAIL {audio_path.name}: {type(exc).__name__}: {exc}")
            if n % 50 == 0 or n == len(indices):
                print(f"    [{split}] {n}/{len(indices)} "
                      f"(saved {done}, skipped {skipped}, failed {failed})")

    print(f"\nDONE: saved {done}, skipped(existing) {skipped}, failed {failed} "
          f"-> {dataset_out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract mean-pooled per-layer embeddings for a dataset, "
                    "mirroring the source folder structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    p.add_argument("--root", required=True, help="Dataset root dir (data / emodb / musan)")
    p.add_argument("--model", required=True, help="HF model id, e.g. openai/whisper-base")
    p.add_argument("--out", default="embeddings", help="Embeddings root dir")
    p.add_argument("--percent", type=float, default=100.0,
                   help="%% of data to extract (AudioMNIST / MUSAN)")
    p.add_argument("--train-percent", type=float, default=100.0,
                   help="%% of EMO-DB train split (EMO-DB only)")
    p.add_argument("--test-percent", type=float, default=100.0,
                   help="%% of EMO-DB test split (EMO-DB only)")
    p.add_argument("--seconds", type=float, default=None,
                   help="Window length per clip; default: loader default "
                        "(MUSAN -> 25 s)")
    p.add_argument("--layers", default="all",
                   help='Comma-separated layer indices or "all" (0 = embedding layer)')
    p.add_argument("--format", default="pt", choices=["pt", "npz"])
    p.add_argument("--task", default="source",
                   choices=["source", "music_type", "music_genre"],
                   help="MUSAN task (MUSAN only)")
    p.add_argument("--genre-from", default="primary", choices=["primary", "full"],
                   help="MUSAN music_genre label granularity")
    p.add_argument("--device", default=None, help='"cuda", "cpu" or auto')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract clips whose embedding file already exists")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
