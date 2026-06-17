"""
run_probe.py
============

Driver that linear-probes precomputed speech embeddings, one probe **per layer**,
for any of the three datasets in this repo (AudioMNIST / EMO-DB / MUSAN).

It glues together:

  * the dataset loaders (``audio_mnist`` / ``emodb`` / ``musan``) in
    ``mode="embedding"``, which read the per-clip ``.pt`` files, and
  * ``linear_probe.probe_layers``, which fits a logistic-regression probe on each
    layer and reports where the target is most linearly decodable.

Why load with ``layer="all"``?
------------------------------
Every clip's ``.pt`` file stores **all** layers (``pooled = {layer: [dim]}``). If
we asked the loader for one layer at a time we'd re-open every file once per layer
(7x for whisper-base). Instead we materialize each split **once** as
``X[N, num_layers, dim]`` and then slice ``X[:, L, :]`` per layer in memory.

Fit / eval policy
-----------------
* AudioMNIST / MUSAN: the loaders give a stratified train/val/test carve-out.
  We fit on **train + val** and evaluate on the held-out **test** split.
* EMO-DB: the loaders give EMO-DB's *predefined* train/test split (val carved
  from train). Same policy: fit on **train + val** (= the full train CSV),
  evaluate on the predefined **test** CSV.

Usage
-----
    # AudioMNIST speaker id
    python run_probe.py --dataset audio_mnist

    # EMO-DB emotion
    python run_probe.py --dataset emodb

    # MUSAN, pick the task
    python run_probe.py --dataset musan --task source
    python run_probe.py --dataset musan --task music_type
    python run_probe.py --dataset musan --task music_genre

    # different model tree / output path
    python run_probe.py --dataset emodb --model openai__whisper-base \
        --out probe_results/emodb_whisper-base.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from linear_probe import probe_layers


# Per-dataset wiring: the audio root and the sub-folder name used under the
# embeddings tree (embeddings/<model>/<subdir>). Both default to the same name
# except AudioMNIST, whose audio lives under "data/".
DATASETS = {
    "audio_mnist": {"root": "data", "subdir": "data"},
    "emodb": {"root": "emodb", "subdir": "emodb"},
    "musan": {"root": "musan", "subdir": "musan"},
}


# --------------------------------------------------------------------------- #
# Loader dispatch
# --------------------------------------------------------------------------- #
def _build_loaders(dataset: str, task: str, root: str, embedding_dir: str,
                   batch_size: int, num_workers: int):
    """Return ``(train, val, test, ds)`` loaders in embedding mode, layer='all'."""
    common = dict(
        root_dir=root,
        mode="embedding",
        embedding_dir=embedding_dir,
        layer="all",                 # one read per clip -> [num_layers, dim]
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if dataset == "audio_mnist":
        from audio_mnist import get_speaker_dataloaders
        return get_speaker_dataloaders(**common)
    if dataset == "emodb":
        from emodb import get_emotion_dataloaders
        return get_emotion_dataloaders(**common)
    if dataset == "musan":
        from musan import get_musan_dataloaders
        return get_musan_dataloaders(task=task, **common)
    raise ValueError(f"unknown dataset: {dataset!r}")


def _num_classes(ds) -> int:
    """AudioMNIST exposes ``num_speakers``; EMO-DB / MUSAN expose ``num_classes``."""
    return int(getattr(ds, "num_classes", None) or ds.num_speakers)


# --------------------------------------------------------------------------- #
# Materialization
# --------------------------------------------------------------------------- #
def _materialize(loader) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Drain a loader into ``(X[N, num_layers, dim], y[N])`` numpy arrays.

    Returns ``(None, None)`` if the loader is empty (e.g. ``val_split=0``).
    """
    xs, ys = [], []
    for xb, yb in loader:
        x = xb.detach().cpu().numpy() if hasattr(xb, "detach") else np.asarray(xb)
        y = yb.detach().cpu().numpy() if hasattr(yb, "detach") else np.asarray(yb)
        xs.append(x)
        ys.append(np.ravel(y))
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _concat(a, b):
    """Concatenate two (X, y) pairs, tolerating an empty second pair."""
    Xa, ya = a
    Xb, yb = b
    if Xb is None:
        return Xa, ya
    return np.concatenate([Xa, Xb], axis=0), np.concatenate([ya, yb], axis=0)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(dataset: str, task: str, model: str, embeddings_root: str,
        root: Optional[str], out: Optional[str], batch_size: int,
        num_workers: int, C: float, max_iter: int, seed: int,
        standardize: bool) -> dict:
    cfg = DATASETS[dataset]
    root = root or cfg["root"]
    embedding_dir = str(Path(embeddings_root) / model / cfg["subdir"])

    if not Path(embedding_dir).is_dir():
        raise FileNotFoundError(
            f"embedding_dir not found: {embedding_dir!r}\n"
            f"  (expected embeddings/<model>/<subdir>; check --model / --embeddings-root)"
        )

    print(f"dataset       : {dataset}" + (f" (task={task})" if dataset == "musan" else ""))
    print(f"audio root    : {root}")
    print(f"embedding dir : {embedding_dir}")

    train, val, test, ds = _build_loaders(
        dataset, task, root, embedding_dir, batch_size, num_workers
    )
    n_classes = _num_classes(ds)

    # Materialize each split ONCE as [N, num_layers, dim].
    fit = _concat(_materialize(train), _materialize(val))   # train + val
    Xte, yte = _materialize(test)
    Xfit, yfit = fit
    if Xfit is None or Xte is None:
        raise RuntimeError("empty train or test split after materialization "
                           "(no embeddings extracted for this dataset?)")

    num_layers = Xfit.shape[1]
    print(f"classes       : {n_classes}")
    print(f"fit / test    : {len(yfit)} / {len(yte)} clips   "
          f"(features {Xfit.shape[1]}x{Xfit.shape[2]} = num_layers x dim)")

    # Slice per layer in memory -> per-layer (features, labels) sources.
    layer_sources = {L: (Xfit[:, L, :], yfit) for L in range(num_layers)}
    test_sources = {L: (Xte[:, L, :], yte) for L in range(num_layers)}

    if out is None:
        # NOTE: keep this OUT of any dataset root (e.g. data/). AudioMNIST treats
        # every sub-folder of its root as a speaker class, so a results folder
        # under data/ would be picked up as a phantom 61st speaker.
        tag = f"{dataset}_{task}" if dataset == "musan" else dataset
        out = f"probe_results/{tag}_{model}.json"

    report = probe_layers(
        layer_sources, test_sources, save_path=out,
        C=C, max_iter=max_iter, seed=seed, standardize=standardize,
    )

    # Per-layer accuracy table.
    print("\nper-layer accuracy:")
    per = report["per_layer"]
    for L in sorted(per, key=int):
        r = per[L]
        star = "  <- best" if L == report["best_layer"] else ""
        print(f"  layer {int(L):>2}: acc={r['accuracy']:.3f}  "
              f"macro_f1={r['macro_f1']:.3f}  chance={r['chance']:.3f}{star}")
    print(f"\nbest layer = {report['best_layer']} "
          f"(acc={report['best_accuracy']:.3f})")
    print(f"wrote {out}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Per-layer linear probing of speech embeddings.")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                   help="which dataset/loader to probe")
    p.add_argument("--task", default="source",
                   choices=("source", "music_type", "music_genre"),
                   help="MUSAN task (ignored for other datasets)")
    p.add_argument("--model", default="openai__whisper-base",
                   help="model sub-folder under the embeddings root")
    p.add_argument("--embeddings-root", default="embeddings",
                   help="root of the embeddings tree (default: embeddings)")
    p.add_argument("--root", default=None,
                   help="audio/label root (default: per-dataset; emodb needs its CSVs here)")
    p.add_argument("--out", default=None, help="output JSON path (default: probe_results/...)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--C", type=float, default=0.1, help="logistic-regression inverse reg strength")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-standardize", action="store_true", help="skip feature standardization")
    args = p.parse_args()

    run(
        dataset=args.dataset, task=args.task, model=args.model,
        embeddings_root=args.embeddings_root, root=args.root, out=args.out,
        batch_size=args.batch_size, num_workers=args.num_workers,
        C=args.C, max_iter=args.max_iter, seed=args.seed,
        standardize=not args.no_standardize,
    )


if __name__ == "__main__":
    main()
