"""
linear_probe.py
===============

Layer-wise linear probing of speech embeddings.

A *linear probe* trains a simple linear classifier on **frozen** features and
measures how much task-relevant information is linearly decodable from them.
Run one probe per layer to see *where* in a network a given attribute (speaker
identity, spoken digit, gender, bonafide-vs-spoof, ...) becomes most accessible.

This module pairs with ``speech_embedding_extractor.py``: that script saves a
per-utterance dict with mean-pooled, per-layer embeddings (``pooled``) and the
raw ``[time, dim]`` states (``raw``). Here we:

  1. Load many such saved files and attach a label to each (via a *labeler*).
  2. Assemble, for every layer, a feature matrix ``X = [N, dim]`` + labels ``y``.
  3. Train + evaluate one ``LinearProbe`` per layer.
  4. Report per-layer accuracy / macro-F1 and the best layer.

The classification *target* is decoupled from the probe via a ``labeler``
callable ``path -> label`` (default: AudioMNIST speaker id). Swap the labeler to
probe digits, gender, accent, spoof, etc. — the rest of the pipeline is unchanged.

Smoke test (no audio, no torch required)::

    python linear_probe.py --smoke-test

Dependencies:
    pip install numpy scikit-learn      # core probing
    pip install torch                   # only to load .pt embedding files
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PathLike = Union[str, Path]
Labeler = Callable[[Path], Optional[object]]

# Layer -> feature matrix [N, dim]; every layer shares the same row order / labels.
FeaturesByLayer = Dict[int, np.ndarray]


# ---------------------------------------------------------------------- #
# Core probe
# ---------------------------------------------------------------------- #
class LinearProbe:
    """A single linear classifier (standardize -> multinomial logistic reg).

    Kept deliberately thin so it is easy to swap the estimator (e.g. a linear
    SVM) without touching the layer-wise driver below.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        }


@dataclass
class LayerResult:
    layer: int
    accuracy: float
    macro_f1: float
    chance: float
    n_train: int
    n_test: int
    n_classes: int


class ProbeReport:
    """Container for per-layer results with small reporting helpers."""

    def __init__(self, results: List[LayerResult], target: str = "label") -> None:
        self.results = sorted(results, key=lambda r: r.layer)
        self.target = target

    def best(self) -> LayerResult:
        return max(self.results, key=lambda r: r.accuracy)

    def to_table(self) -> str:
        head = f"{'layer':>5} {'acc':>8} {'macro_f1':>9} {'chance':>8}  n_tr/n_te  classes"
        lines = [f"Layer-wise linear probe  (target = {self.target})", head, "-" * len(head)]
        best_layer = self.best().layer
        for r in self.results:
            mark = "  <- best" if r.layer == best_layer else ""
            lines.append(
                f"{r.layer:>5} {r.accuracy:>8.4f} {r.macro_f1:>9.4f} "
                f"{r.chance:>8.4f}  {r.n_train}/{r.n_test}   {r.n_classes}{mark}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"target": self.target, "results": [asdict(r) for r in self.results]}

    def save_json(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def save_csv(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["layer", "accuracy", "macro_f1", "chance", "n_train", "n_test", "n_classes"]
        rows = [",".join(cols)]
        for r in self.results:
            d = asdict(r)
            rows.append(",".join(str(d[c]) for c in cols))
        path.write_text("\n".join(rows) + "\n")


def run_layerwise_probe(
    features_by_layer: FeaturesByLayer,
    labels: Sequence,
    *,
    groups: Optional[Sequence] = None,
    test_size: float = 0.25,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
    target: str = "label",
    verbose: bool = True,
) -> ProbeReport:
    """Train + evaluate one :class:`LinearProbe` per layer.

    Parameters
    ----------
    features_by_layer:
        ``{layer_index: X}`` with ``X`` shaped ``[N, dim]``. Every layer must
        share the same ``N`` and row order (row i is utterance i).
    labels:
        Length-``N`` sequence of targets (e.g. speaker ids). Any hashable type.
    groups:
        Optional length-``N`` group ids for a *group-disjoint* split (e.g. split
        by speaker when probing a content attribute so the test set has unseen
        speakers). ``None`` -> a label-stratified random split. Do **not** pass
        groups when the label *is* the group (e.g. speaker-id probing).
    test_size, C, max_iter, seed:
        Split fraction, logistic-regression inverse-reg strength, solver
        iterations, RNG seed.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        raise ValueError("No samples to probe.")

    classes, counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError(f"Need >= 2 classes to probe, got {n_classes}.")
    # Majority-class baseline is a fairer "chance" line than 1/n_classes.
    chance = float(counts.max() / n)

    train_idx, test_idx = _split_indices(
        labels, groups=groups, test_size=test_size, seed=seed
    )

    results: List[LayerResult] = []
    for layer in sorted(features_by_layer):
        X = np.asarray(features_by_layer[layer], dtype=np.float32)
        if X.shape[0] != n:
            raise ValueError(
                f"Layer {layer}: {X.shape[0]} rows but {n} labels — row counts must match."
            )
        probe = LinearProbe(C=C, max_iter=max_iter, seed=seed)
        probe.fit(X[train_idx], labels[train_idx])
        metrics = probe.evaluate(X[test_idx], labels[test_idx])
        results.append(
            LayerResult(
                layer=layer,
                accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"],
                chance=chance,
                n_train=len(train_idx),
                n_test=len(test_idx),
                n_classes=n_classes,
            )
        )
        if verbose:
            print(
                f"  layer {layer:>2}: acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

    report = ProbeReport(results, target=target)
    if verbose:
        print(report.to_table())
    return report


def _split_indices(
    labels: np.ndarray,
    groups: Optional[Sequence],
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train/test row indices: group-disjoint if ``groups`` given, else stratified."""
    idx = np.arange(len(labels))
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(idx, labels, groups=np.asarray(groups)))
        return train_idx, test_idx

    # Stratify only when every class has >= 2 samples; otherwise fall back.
    _, counts = np.unique(labels, return_counts=True)
    stratify = labels if counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=stratify
    )
    return train_idx, test_idx


# ---------------------------------------------------------------------- #
# Loading embeddings produced by speech_embedding_extractor.py
# ---------------------------------------------------------------------- #
def load_embeddings(
    paths: Sequence[PathLike],
    labeler: Labeler,
    *,
    source: str = "pooled",
    layers: Optional[Sequence[int]] = None,
    group_fn: Optional[Labeler] = None,
) -> Tuple[FeaturesByLayer, List, Optional[List]]:
    """Load saved per-utterance embeddings into per-layer matrices + labels.

    Each file is a ``.pt`` (``torch.save`` of the extractor's result dict) or a
    ``.npz`` (flat ``pooled_<i>`` / ``raw_<i>`` arrays).

    Parameters
    ----------
    paths:
        Embedding files. The labeler receives each ``Path``.
    labeler:
        ``path -> label`` (return ``None`` to skip a file).
    source:
        ``"pooled"`` uses the stored mean-pooled vector; ``"raw"`` mean-pools the
        ``[time, dim]`` states on the fly. Both yield one ``[dim]`` vector/layer.
    layers:
        Restrict to these layer indices; ``None`` keeps the intersection of
        layers present in every file.
    group_fn:
        Optional ``path -> group`` for a group-disjoint split downstream.

    Returns
    -------
    ``(features_by_layer, labels, groups_or_None)``.
    """
    if source not in ("pooled", "raw"):
        raise ValueError(f"source must be 'pooled' or 'raw', got {source!r}")

    records: List[Tuple[Dict[int, np.ndarray], object, Optional[object]]] = []
    common: Optional[set] = None
    for p in paths:
        path = Path(p)
        label = labeler(path)
        if label is None:
            continue
        per_layer = _load_one(path, source)
        if not per_layer:
            continue
        common = set(per_layer) if common is None else (common & set(per_layer))
        group = group_fn(path) if group_fn is not None else None
        records.append((per_layer, label, group))

    if not records or not common:
        raise RuntimeError("No usable embeddings found (no files, or no shared layers).")

    keep = sorted(common if layers is None else (common & set(layers)))
    if not keep:
        raise RuntimeError(f"Requested layers {layers} not present in all files.")

    features_by_layer: FeaturesByLayer = {layer: [] for layer in keep}
    labels: List = []
    groups: List = []
    for per_layer, label, group in records:
        for layer in keep:
            features_by_layer[layer].append(np.asarray(per_layer[layer], dtype=np.float32))
        labels.append(label)
        groups.append(group)

    stacked = {layer: np.vstack(rows) for layer, rows in features_by_layer.items()}
    groups_out = groups if group_fn is not None else None
    return stacked, labels, groups_out


def _load_one(path: Path, source: str) -> Dict[int, np.ndarray]:
    """Return ``{layer_index: 1-D vector}`` for one saved utterance."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        prefix = "pooled_" if source == "pooled" else "raw_"
        out: Dict[int, np.ndarray] = {}
        for key in data.files:
            if key.startswith(prefix):
                idx = int(key[len(prefix):])
                arr = np.asarray(data[key], dtype=np.float32)
                out[idx] = arr if source == "pooled" else arr.mean(axis=0)
        return out
    if suffix == ".pt":
        import torch  # lazy: only needed for torch-saved files

        rec = torch.load(path, map_location="cpu")
        store = rec["pooled"] if source == "pooled" else rec["raw"]
        out = {}
        for idx, t in store.items():
            arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
            out[int(idx)] = arr.astype(np.float32) if source == "pooled" else arr.mean(axis=0)
        return out
    raise ValueError(f"Unsupported embedding file {path.name} (expected .pt or .npz)")


# ---------------------------------------------------------------------- #
# Labelers (modular targets) — swap these to probe a different attribute
# ---------------------------------------------------------------------- #
def audiomnist_field(index: int) -> Labeler:
    """Labeler from the AudioMNIST stem ``{digit}_{speaker}_{rep}``.

    ``index=0`` -> digit, ``index=1`` -> speaker id, ``index=2`` -> repetition.
    """

    def _labeler(path: Path) -> Optional[str]:
        parts = path.stem.split("_")
        return parts[index] if len(parts) > index else None

    return _labeler


# Ready-made targets.
speaker_labeler = audiomnist_field(1)  # default target: speaker identification
digit_labeler = audiomnist_field(0)


def metadata_labeler(meta_path: PathLike, field: str) -> Labeler:
    """Labeler that maps an utterance -> speaker -> metadata ``field``.

    Reads ``audioMNIST_meta.txt`` (JSON keyed by zero-padded speaker id) so you
    can probe ``"gender"``, ``"accent"``, ``"age"``, etc. with the same driver.
    """
    meta = json.loads(Path(meta_path).read_text())

    def _labeler(path: Path) -> Optional[object]:
        speaker = path.stem.split("_")[1]
        entry = meta.get(speaker) or meta.get(speaker.zfill(2))
        return None if entry is None else entry.get(field)

    return _labeler


_LABELERS: Dict[str, Labeler] = {
    "speaker": speaker_labeler,
    "digit": digit_labeler,
}


# ---------------------------------------------------------------------- #
# Smoke test (synthetic data; no audio / no torch needed)
# ---------------------------------------------------------------------- #
def make_dummy_data(
    n_classes: int = 40,
    n_per_class: int = 12,
    dim: int = 64,
    n_layers: int = 6,
    informative_layers: Sequence[int] = (3, 4),
    seed: int = 0,
) -> Tuple[FeaturesByLayer, np.ndarray]:
    """Synthetic per-layer features with a known, layer-dependent signal.

    Each class gets a random centroid. ``informative_layers`` place samples near
    their class centroid (linearly separable); all other layers are pure noise.
    A correct probe should score high on informative layers and ~chance elsewhere.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    n = len(labels)
    centroids = rng.normal(scale=5.0, size=(n_classes, dim))

    features_by_layer: FeaturesByLayer = {}
    for layer in range(n_layers):
        if layer in informative_layers:
            X = centroids[labels] + rng.normal(scale=1.0, size=(n, dim))
        else:
            X = rng.normal(scale=1.0, size=(n, dim))
        features_by_layer[layer] = X.astype(np.float32)
    return features_by_layer, labels


def smoke_test() -> bool:
    """Build dummy data, run the layer-wise probe, and assert it behaves."""
    print("== linear_probe smoke test ==")
    informative = (3, 4)
    features_by_layer, labels = make_dummy_data(
        n_classes=40, n_per_class=12, dim=64, n_layers=6, informative_layers=informative
    )
    print(f"dummy: {len(labels)} samples, 40 classes, layers {sorted(features_by_layer)}, "
          f"informative={informative}")

    report = run_layerwise_probe(
        features_by_layer, labels, test_size=0.25, seed=0,
        target="speaker(dummy)", verbose=False,
    )
    print(report.to_table())

    by_layer = {r.layer: r for r in report.results}
    chance = report.results[0].chance
    ok = True
    for layer, r in by_layer.items():
        expect_high = layer in informative
        good = r.accuracy > 0.80 if expect_high else r.accuracy < (chance + 0.15)
        status = "OK" if good else "FAIL"
        kind = "informative" if expect_high else "noise"
        print(f"  layer {layer} ({kind:>11}): acc={r.accuracy:.3f} -> {status}")
        ok = ok and good

    best = report.best()
    if best.layer not in informative:
        print(f"  best layer {best.layer} not in informative set -> FAIL")
        ok = False

    # Exercise the round-trip serialization helpers too.
    out = Path("data/probe_results/smoke_test.json")
    report.save_json(out)
    report.save_csv(out.with_suffix(".csv"))
    print(f"  wrote {out} and {out.with_suffix('.csv')}")

    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return ok


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Layer-wise linear probing of speech embeddings.")
    p.add_argument("--smoke-test", action="store_true", help="Run a synthetic-data self-check and exit.")
    p.add_argument("--emb-glob", help="Glob for saved embedding files, e.g. 'data/emb/*.pt'")
    p.add_argument("--target", default="speaker", choices=sorted(_LABELERS),
                   help="Built-in labeler / classification target (default: speaker).")
    p.add_argument("--source", default="pooled", choices=["pooled", "raw"],
                   help="Probe the mean-pooled vector or mean-pooled raw states.")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices to probe (default: all shared).")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional output path stem for .json/.csv results.")
    args = p.parse_args()

    if args.smoke_test:
        raise SystemExit(0 if smoke_test() else 1)

    if not args.emb_glob:
        p.error("provide --emb-glob (or use --smoke-test)")

    paths = sorted(_glob.glob(args.emb_glob))
    if not paths:
        p.error(f"no files matched {args.emb_glob!r}")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    features_by_layer, labels, _ = load_embeddings(
        paths, _LABELERS[args.target], source=args.source, layers=layers
    )
    print(f"Loaded {len(labels)} utterances, layers {sorted(features_by_layer)}")
    report = run_layerwise_probe(
        features_by_layer, labels, test_size=args.test_size, C=args.C,
        seed=args.seed, target=args.target, verbose=True,
    )
    best = report.best()
    print(f"\nBest layer for '{args.target}': {best.layer} (acc={best.accuracy:.4f})")
    if args.out:
        report.save_json(Path(args.out).with_suffix(".json"))
        report.save_csv(Path(args.out).with_suffix(".csv"))
        print(f"Saved results to {args.out}.json / .csv")


if __name__ == "__main__":
    main()
