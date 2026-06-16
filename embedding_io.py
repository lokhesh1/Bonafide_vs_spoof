"""
embedding_io.py
===============

Shared helpers for the *embedding* half of the pipeline: locating a clip's
precomputed embedding file (mirroring the audio folder structure) and loading
its mean-pooled per-layer vectors back.

Both ends go through ``embedding_path`` so the on-disk layout is guaranteed to
line up:

  * the extraction driver (``extract_dataset_embeddings.py``) *writes* the files;
  * the dataset loaders (``audio_mnist.py`` / ``emodb.py`` / ``musan.py`` in
    ``mode="embedding"``) *read* them.

An embedding file is whatever ``SpeechEmbeddingExtractor._save`` produces:
  * ``.pt``  -> a dict with ``pooled = {layer_idx: tensor[dim]}`` (+ raw, meta).
  * ``.npz`` -> flat arrays ``pooled_<idx>`` / ``raw_<idx>`` (+ ``__model__``).

This module has **no torch/transformers import at module load**; torch/numpy are
imported lazily inside :func:`load_pooled_embedding` only when a file is read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
LayerSpec = Union[int, str, None]


def embedding_path(
    audio_path: PathLike,
    dataset_root: PathLike,
    embedding_dir: PathLike,
    ext: str = "pt",
) -> Path:
    """Mirror an audio file's location under ``embedding_dir``.

    The embedding for ``<dataset_root>/<rel>.<audio-ext>`` is placed at
    ``<embedding_dir>/<rel>.<ext>`` — the sub-folder structure under the dataset
    root is preserved exactly. ``embedding_dir`` is therefore the per-dataset,
    per-model root, e.g. ``embeddings/openai__whisper-base/emodb``.
    """
    audio_path = Path(audio_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    rel = audio_path.relative_to(dataset_root)
    return Path(embedding_dir) / rel.with_suffix(f".{ext.lstrip('.')}")


def _sorted_layer_keys(pooled) -> list:
    """Layer keys in ascending numeric order (keys may be int or str)."""
    return sorted(pooled.keys(), key=lambda k: int(k))


def load_pooled_embedding(path: PathLike, layer: LayerSpec = "all", ext: str = "pt"):
    """Load mean-pooled embeddings saved by the extractor.

    Parameters
    ----------
    layer:
        ``"all"`` / ``None`` -> stacked ``[num_layers, dim]`` in ascending layer
        order; an ``int`` -> that single layer's ``[dim]`` vector (negative
        indices count from the last layer, so ``-1`` is the top layer).

    Returns a ``torch.Tensor`` for ``.pt`` files, a ``numpy.ndarray`` for
    ``.npz`` files.
    """
    path = Path(path)
    ext = ext.lstrip(".")

    if ext == "pt":
        import torch

        try:
            result = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # older torch without the weights_only kwarg
            result = torch.load(path, map_location="cpu")
        pooled = result["pooled"]
        keys = _sorted_layer_keys(pooled)
        if layer in ("all", None):
            return torch.stack([pooled[k].float() for k in keys], dim=0)
        return pooled[keys[int(layer)]].float()

    if ext == "npz":
        import numpy as np

        data = np.load(path)
        keys = sorted(
            (k for k in data.files if k.startswith("pooled_")),
            key=lambda k: int(k.split("_", 1)[1]),
        )
        if not keys:
            raise KeyError(f"no pooled_* arrays in {path!r}")
        if layer in ("all", None):
            return np.stack([data[k] for k in keys], axis=0)
        return np.asarray(data[keys[int(layer)]])

    raise ValueError(f"unknown embedding ext: {ext!r} (use 'pt' or 'npz')")
