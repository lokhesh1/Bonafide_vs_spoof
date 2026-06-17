"""
linear_probe.py
===============

Linear probing of (frozen) speech embeddings.

A *linear probe* trains a simple linear classifier on fixed features and reports
how accurately a target attribute (speaker id, digit, spoof, ...) can be linearly
decoded from them. Run one probe per layer to see *where* in a network the target
is most accessible.

This module is deliberately I/O-agnostic. It accepts features either as:

  * a **DataLoader** (any iterable yielding ``(features, labels)`` batches —
    e.g. a ``torch.utils.data.DataLoader``), or
  * in-memory **data + labels** arrays/tensors.

It standardizes, fits a multinomial logistic regression, evaluates accuracy, and
saves the result to a JSON file.

API
---
``linear_probe(...)``       -> probe one feature set, return + save accuracy.
``probe_layers(...)``       -> run ``linear_probe`` per layer, save a combined JSON.

Self-check (synthetic data, no torch needed)::

    python linear_probe.py --smoke-test

Dependencies:
    pip install numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PathLike = Union[str, Path]
# A per-layer data source is either an (features, labels) pair or a DataLoader.
Source = object


# ---------------------------------------------------------------------- #
# Input handling
# ---------------------------------------------------------------------- #
def _to_numpy(x) -> Optional[np.ndarray]:
    """Convert a tensor / array / list to a numpy array (None passes through)."""
    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _materialize(data, labels, dataloader) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve one data source to ``(X[N, dim], y[N])`` numpy arrays.

    Pass exactly one of: ``(data, labels)`` or ``dataloader``.
    """
    if dataloader is not None:
        if data is not None or labels is not None:
            raise ValueError("Pass either (data, labels) or dataloader, not both.")
        xs, ys = [], []
        for batch in dataloader:
            xb, yb = batch  # each batch is (features, labels)
            xs.append(_to_numpy(xb))
            ys.append(np.ravel(_to_numpy(yb)))
        if not xs:
            raise ValueError("dataloader yielded no batches.")
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    if data is None or labels is None:
        raise ValueError("Provide both data and labels, or a dataloader.")
    X = _to_numpy(data)
    y = np.ravel(_to_numpy(labels))
    if X.ndim != 2:
        X = X.reshape(len(y), -1)
    return X, y


# ---------------------------------------------------------------------- #
# Core probe
# ---------------------------------------------------------------------- #
def linear_probe(
    data=None,
    labels=None,
    dataloader=None,
    *,
    test_data=None,
    test_labels=None,
    test_dataloader=None,
    test_size: float = 0.25,
    standardize: bool = True,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
    save_path: Optional[PathLike] = None,
) -> Dict[str, float]:
    """Fit + evaluate a single linear probe and (optionally) save accuracy.

    Parameters
    ----------
    data, labels:
        In-memory features ``[N, dim]`` and targets ``[N]`` (arrays or tensors).
    dataloader:
        Alternative to ``data``/``labels``: an iterable of ``(features, labels)``
        batches. Provide this *or* ``data``/``labels``, not both.
    test_data / test_labels / test_dataloader:
        Optional held-out set. If omitted, the training data is split using
        ``test_size`` (label-stratified when possible).
    standardize, C, max_iter, seed:
        Standardize features (recommended), logistic-regression inverse-reg
        strength, solver iterations, RNG seed.
    save_path:
        If given, write the result dict to this JSON file.

    Returns
    -------
    dict with ``accuracy``, ``macro_f1``, ``chance``, ``n_train``, ``n_test``,
    ``n_classes``.
    """
    X, y = _materialize(data, labels, dataloader)

    if test_dataloader is not None or test_data is not None:
        X_train, y_train = X, y
        X_test, y_test = _materialize(test_data, test_labels, test_dataloader)
    else:
        _, counts = np.unique(y, return_counts=True)
        stratify = y if counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )

    n_classes = int(len(np.unique(y_train)))
    if n_classes < 2:
        raise ValueError(f"Need >= 2 classes to probe, got {n_classes}.")

    steps = []
    if standardize:
        steps.append(StandardScaler())
    steps.append(LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    model = make_pipeline(*steps)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)



    _, te_counts = np.unique(y_test, return_counts=True)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "chance": float(te_counts.max() / len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": n_classes,
    }
    if save_path is not None:
        _save_json(result, save_path)
    return result


def probe_layers(
    layer_sources: Mapping[object, Source],
    test_sources: Optional[Mapping[object, Source]] = None,
    *,
    save_path: Optional[PathLike] = None,
    **probe_kwargs,
) -> Dict[str, object]:
    """Run :func:`linear_probe` once per layer and save a combined accuracy JSON.

    Parameters
    ----------
    layer_sources:
        ``{layer: source}`` where each ``source`` is an ``(features, labels)``
        pair or a DataLoader.
    test_sources:
        Optional ``{layer: source}`` of held-out sets, same form as above.
    save_path:
        If given, write the combined result to this JSON file.

    Returns
    -------
    dict with ``per_layer`` (``{layer: accuracy-dict}``), ``best_layer`` and
    ``best_accuracy``.
    """
    per_layer: Dict[str, Dict[str, float]] = {}
    for layer, src in layer_sources.items():
        test_src = test_sources.get(layer) if test_sources else None
        per_layer[str(layer)] = _probe_source(src, test_src, probe_kwargs)

    best_layer = max(per_layer, key=lambda k: per_layer[k]["accuracy"])
    out: Dict[str, object] = {
        "per_layer": per_layer,
        "best_layer": best_layer,
        "best_accuracy": per_layer[best_layer]["accuracy"],
    }
    if save_path is not None:
        _save_json(out, save_path)
    return out


def _probe_source(src: Source, test_src: Optional[Source], kwargs: dict) -> Dict[str, float]:
    """Dispatch a (pair | dataloader) source into :func:`linear_probe`."""
    data, labels, loader = _unpack_source(src)
    t_data, t_labels, t_loader = _unpack_source(test_src) if test_src is not None else (None, None, None)
    return linear_probe(
        data=data, labels=labels, dataloader=loader,
        test_data=t_data, test_labels=t_labels, test_dataloader=t_loader,
        save_path=None, **kwargs,
    )


def _unpack_source(src: Source):
    """An ``(features, labels)`` tuple -> (data, labels, None); else (None, None, loader)."""
    if isinstance(src, tuple) and len(src) == 2:
        return src[0], src[1], None
    return None, None, src


def _save_json(obj: dict, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------- #
# Smoke test (synthetic data; no torch / no audio needed)
# ---------------------------------------------------------------------- #
def _make_dummy(
    n_classes: int = 40,
    n_per_class: int = 12,
    dim: int = 64,
    n_layers: int = 6,
    informative=(3, 4),
    seed: int = 0,
):
    """Per-layer features with signal only in ``informative`` layers; rest noise."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    n = len(labels)
    centroids = rng.normal(scale=5.0, size=(n_classes, dim))
    layers = {}
    for layer in range(n_layers):
        if layer in informative:
            layers[layer] = (centroids[labels] + rng.normal(size=(n, dim))).astype(np.float32)
        else:
            layers[layer] = rng.normal(size=(n, dim)).astype(np.float32)
    return layers, labels


def _chunk_loader(X, y, batch_size=32):
    """Minimal DataLoader stand-in: yield ``(features, labels)`` numpy batches."""
    for i in range(0, len(y), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


def smoke_test() -> bool:
    print("== linear_probe smoke test ==")
    informative = (3, 4)
    layers, labels = _make_dummy(informative=informative)
    n_classes = len(np.unique(labels))
    print(f"dummy: {len(labels)} samples, {n_classes} classes, layers {sorted(layers)}, "
          f"informative={informative}")

    ok = True

    # 1) array path: informative layer should be learnable, noise layer ~chance.
    inf = linear_probe(layers[informative[0]], labels, seed=0)
    noise = linear_probe(layers[0], labels, seed=0)
    print(f"array  informative acc={inf['accuracy']:.3f} (chance {inf['chance']:.3f})")
    print(f"array  noise       acc={noise['accuracy']:.3f} (chance {noise['chance']:.3f})")
    ok &= inf["accuracy"] > 0.80
    ok &= noise["accuracy"] < noise["chance"] + 0.15

    # 2) dataloader path must match the array path exactly (same order + seed).
    via_loader = linear_probe(
        dataloader=_chunk_loader(layers[informative[0]], labels), seed=0
    )
    print(f"loader informative acc={via_loader['accuracy']:.3f} "
          f"(matches array: {np.isclose(via_loader['accuracy'], inf['accuracy'])})")
    ok &= np.isclose(via_loader["accuracy"], inf["accuracy"])

    # 3) per-layer driver + JSON save; best layer must be an informative one.
    out_path = Path("dataa/probe_results/smoke_test.json")
    report = probe_layers(
        {layer: (X, labels) for layer, X in layers.items()},
        save_path=out_path, seed=0,
    )
    print("per-layer accuracy:")
    for layer in sorted(report["per_layer"], key=int):
        acc = report["per_layer"][layer]["accuracy"]
        tag = "informative" if int(layer) in informative else "noise"
        print(f"  layer {layer:>2} ({tag:>11}): acc={acc:.3f}")
    print(f"best layer = {report['best_layer']} (acc={report['best_accuracy']:.3f})")
    ok &= int(report["best_layer"]) in informative
    ok &= out_path.exists()
    print(f"wrote {out_path}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return bool(ok)


def main() -> None:
    p = argparse.ArgumentParser(description="Linear probing of speech embeddings.")
    p.add_argument("--smoke-test", action="store_true", help="Run a synthetic-data self-check and exit.")
    args = p.parse_args()
    if args.smoke_test:
        raise SystemExit(0 if smoke_test() else 1)
    p.error("nothing to do: import linear_probe / probe_layers, or pass --smoke-test")


if __name__ == "__main__":
    main()
