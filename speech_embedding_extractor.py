"""
speech_embedding_extractor.py
=============================

Modular layer-wise speech-embedding extractor for Hugging Face audio models
(ASR / SSL encoders / speech-to-speech systems).

Given a model name, it:
  1. Loads the matching processor/feature-extractor and model from the Hub.
  2. Accepts speech input (file path, numpy array, or torch tensor).
  3. Runs a forward pass with `output_hidden_states=True`.
  4. Returns per-layer embeddings, both raw `[time, dim]` and mean-pooled `[dim]`.
     Layer index 0 is the *embedding layer* output (input to the first
     transformer block); indices 1..N are the transformer layer outputs.
  5. Optionally saves everything to disk (.pt or .npz).

Supported (tested-by-design) families:
  - Whisper                       (encoder-decoder ASR)
  - Wav2Vec2 / HuBERT / WavLM     (encoder-only SSL)
  - SpeechT5 / Seamless (M4T)     (encoder-decoder speech-to-speech)
  - Anything else via the generic AutoModel + output_hidden_states path
    (best-effort; override `encoder_attr` if auto-detection picks the wrong
    encoder, e.g. a text encoder on a multimodal model).

Dependencies:
    pip install torch transformers librosa numpy

Author: generated for the Bonafide_vs_spoof comparative-analysis project.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from transformers import AutoConfig, AutoModel, AutoProcessor

# Make `src` importable regardless of the current working directory, then reuse
# the project's shared audio I/O helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features.audio_io import load_audio, resample


AudioInput = Union[str, Path, np.ndarray, torch.Tensor]
LayerSpec = Union[str, int, Sequence[int], None]


class SpeechEmbeddingExtractor:
    """Load a HF audio model once and extract layer-wise embeddings repeatedly.

    Parameters
    ----------
    model_name:
        Hugging Face model id, e.g. ``"openai/whisper-base"``,
        ``"facebook/wav2vec2-base"``, ``"microsoft/wavlm-base"``.
    device:
        ``"cuda"``, ``"cpu"`` or ``None`` (auto-detect).
    dtype:
        Torch dtype for the model weights (e.g. ``torch.float16`` on GPU).
        Defaults to float32.
    encoder_attr:
        For encoder-decoder / multimodal models, the attribute or method that
        returns the speech encoder. Auto-detected when ``None`` (tries
        ``get_encoder()``, then ``.encoder``, then ``.speech_encoder``).
        Override only if the wrong sub-module is picked.
    trust_remote_code:
        Pass through to ``from_pretrained`` for models with custom code.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        encoder_attr: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype or torch.float32
        self.encoder_attr = encoder_attr

        self.config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        self.is_encoder_decoder = bool(getattr(self.config, "is_encoder_decoder", False))
        self.is_whisper = getattr(self.config, "model_type", "") == "whisper"
        self.sampling_rate = self._resolve_sampling_rate()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract(
        self,
        audio: AudioInput,
        layers: LayerSpec = "all",
        sampling_rate: Optional[int] = None,
        pooling: str = "mean",
        save_path: Optional[Union[str, Path]] = None,
        save_format: str = "pt",
    ) -> Dict[str, object]:
        """Extract layer-wise embeddings for a single utterance.

        Parameters
        ----------
        audio:
            Path to an audio file, a 1-D numpy array, or a 1-D torch tensor.
        layers:
            Which hidden-state indices to keep:
              * ``"all"`` / ``None`` -> every layer (0 = embedding layer).
              * ``int``              -> a single layer.
              * list/tuple of ints   -> those layers (negative indexing ok).
        sampling_rate:
            Required when ``audio`` is an array/tensor and not already at the
            model's expected rate. Ignored for file paths (read from file).
        pooling:
            Time-pooling for the pooled output: ``"mean"`` (mask-aware),
            ``"last"``, or ``"max"``.
        save_path:
            If given, write the result to this path.
        save_format:
            ``"pt"`` (torch.save) or ``"npz"`` (numpy, raw+pooled as arrays).

        Returns
        -------
        dict with keys:
            ``model``         -> model name
            ``num_layers``    -> total hidden-state count returned by the model
            ``layer_indices`` -> the indices actually returned
            ``raw``           -> {index: tensor [time, dim]}
            ``pooled``        -> {index: tensor [dim]}
            ``embedding_layer`` -> index 0 raw tensor (the embedding layer), or None
        """
        waveform, sr = self._load_audio(audio, sampling_rate)
        num_samples = len(waveform)
        inputs = self._preprocess(waveform, sr)
        hidden_states, attention_mask = self._forward(inputs)

        # Whisper pads/truncates every clip to 30s and returns no usable
        # attention mask, so its hidden states contain padded frames. Compute
        # the number of frames that correspond to real audio and drop the rest,
        # making raw + pooled outputs comparable to encoder-only models that
        # already exclude padding.
        valid_frames = self._whisper_valid_frames(num_samples) if self.is_whisper else None

        selected = self._select_layers(layers, len(hidden_states))
        raw: Dict[int, torch.Tensor] = {}
        pooled: Dict[int, torch.Tensor] = {}
        for idx in selected:
            hs = hidden_states[idx][0]  # drop batch dim -> [time, dim]
            if valid_frames is not None:
                hs = hs[:valid_frames]  # slicing auto-clamps if estimate is high
            raw[idx] = hs.float().cpu()
            pooled[idx] = self._pool(hs, attention_mask, pooling).float().cpu()

        result = {
            "model": self.model_name,
            "num_layers": len(hidden_states),
            "layer_indices": selected,
            "raw": raw,
            "pooled": pooled,
            # hidden_states[0] is the embedding-layer output when present.
            "embedding_layer": raw.get(0),
        }

        if save_path is not None:
            self._save(result, save_path, save_format)

        return result

    def extract_batch(
        self,
        audios: Sequence[AudioInput],
        save_dir: Optional[Union[str, Path]] = None,
        save_format: str = "pt",
        **kwargs,
    ) -> List[Dict[str, object]]:
        """Convenience loop over many utterances.

        When ``save_dir`` is set, each result is saved as
        ``<stem>.<ext>`` (file inputs) or ``utt_<i>.<ext>`` (array inputs).
        """
        results = []
        for i, audio in enumerate(audios):
            save_path = None
            if save_dir is not None:
                ext = "pt" if save_format == "pt" else "npz"
                stem = Path(audio).stem if isinstance(audio, (str, Path)) else f"utt_{i}"
                save_path = Path(save_dir) / f"{stem}.{ext}"
            results.append(
                self.extract(audio, save_path=save_path, save_format=save_format, **kwargs)
            )
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_sampling_rate(self) -> int:
        """Pull the expected input sampling rate from the feature extractor."""
        fe = getattr(self.processor, "feature_extractor", self.processor)
        return int(getattr(fe, "sampling_rate", 16_000))

    def _whisper_valid_frames(self, num_samples: int) -> int:
        """Encoder frames that correspond to real (pre-pad) audio.

        mel frames = num_samples // hop_length; the Whisper encoder conv stack
        then downsamples the time axis by a factor of 2.
        """
        fe = getattr(self.processor, "feature_extractor", self.processor)
        hop = int(getattr(fe, "hop_length", 160))
        mel_frames = num_samples // hop
        return max(1, mel_frames // 2)

    def _load_audio(
        self, audio: AudioInput, sampling_rate: Optional[int]
    ) -> tuple[np.ndarray, int]:
        if isinstance(audio, (str, Path)):
            # load_audio resamples to the model's expected rate and mixes to mono.
            wav = load_audio(audio, sr=self.sampling_rate, mono=True)
            return wav, self.sampling_rate

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # mix down to mono

        if sampling_rate is None:
            raise ValueError(
                "sampling_rate must be provided when passing a raw array/tensor."
            )
        audio = resample(audio, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        return audio, self.sampling_rate

    def _preprocess(self, waveform: np.ndarray, sr: int) -> dict:
        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        return {
            k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

    def _get_encoder(self):
        """Best-effort resolution of the speech encoder for enc-dec models."""
        if self.encoder_attr is not None:
            attr = getattr(self.model, self.encoder_attr)
            return attr() if callable(attr) else attr
        if hasattr(self.model, "get_encoder"):
            return self.model.get_encoder()
        for name in ("encoder", "speech_encoder"):
            if hasattr(self.model, name):
                return getattr(self.model, name)
        raise AttributeError(
            f"Could not locate a speech encoder on {type(self.model).__name__}. "
            "Pass encoder_attr=... explicitly."
        )

    def _forward(self, inputs: dict):
        """Run the model and return (hidden_states tuple, attention_mask)."""
        attention_mask = inputs.get("attention_mask")
        if self.is_encoder_decoder:
            encoder = self._get_encoder()
            outputs = encoder(**inputs, output_hidden_states=True)
        else:
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = getattr(outputs, "encoder_hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                f"{self.model_name} did not return hidden_states. "
                "It may need a custom forward path or encoder_attr override."
            )
        return hidden_states, attention_mask

    @staticmethod
    def _select_layers(layers: LayerSpec, n: int) -> List[int]:
        if layers is None or layers == "all":
            return list(range(n))
        if isinstance(layers, int):
            layers = [layers]
        out = []
        for idx in layers:
            real = idx if idx >= 0 else n + idx
            if not 0 <= real < n:
                raise IndexError(f"layer {idx} out of range for {n} hidden states")
            out.append(real)
        return out

    @staticmethod
    def _pool(hs: torch.Tensor, attention_mask, mode: str) -> torch.Tensor:
        """Collapse the time axis of a [time, dim] tensor to [dim]."""
        if mode == "last":
            return hs[-1]
        if mode == "max":
            return hs.max(dim=0).values
        if mode == "mean":
            # Use the mask only if its length matches the (possibly downsampled)
            # time axis; otherwise fall back to a plain mean.
            if attention_mask is not None:
                mask = attention_mask[0]
                if mask.shape[-1] == hs.shape[0]:
                    m = mask.unsqueeze(-1).to(hs.dtype)
                    return (hs * m).sum(0) / m.sum().clamp(min=1)
            return hs.mean(dim=0)
        raise ValueError(f"unknown pooling mode: {mode!r}")

    @staticmethod
    def _save(result: dict, save_path: Union[str, Path], save_format: str) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_format == "pt":
            torch.save(result, save_path)
        elif save_format == "npz":
            flat = {"__model__": np.array(result["model"])}
            for idx, t in result["raw"].items():
                flat[f"raw_{idx}"] = t.numpy()
            for idx, t in result["pooled"].items():
                flat[f"pooled_{idx}"] = t.numpy()
            np.savez_compressed(save_path, **flat)
        else:
            raise ValueError(f"unknown save_format: {save_format!r} (use 'pt' or 'npz')")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _parse_layers(value: Optional[str]) -> LayerSpec:
    if value is None or value.lower() == "all":
        return "all"
    return [int(x) for x in value.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract layer-wise speech embeddings from a HF model.")
    p.add_argument("--model", required=True, help="HF model id, e.g. openai/whisper-base")
    p.add_argument("--audio", required=True, nargs="+", help="One or more audio file paths")
    p.add_argument("--layers", default="all", help='Comma-separated indices or "all" (0 = embedding layer)')
    p.add_argument("--pooling", default="mean", choices=["mean", "last", "max"])
    p.add_argument("--out", default=None, help="Output dir (batch) or file (single)")
    p.add_argument("--format", default="pt", choices=["pt", "npz"])
    p.add_argument("--device", default=None)
    p.add_argument("--encoder-attr", default=None, help="Override speech-encoder attribute")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    extractor = SpeechEmbeddingExtractor(
        args.model,
        device=args.device,
        encoder_attr=args.encoder_attr,
        trust_remote_code=args.trust_remote_code,
    )
    layers = _parse_layers(args.layers)

    if len(args.audio) == 1 and args.out and not os.path.isdir(args.out):
        res = extractor.extract(
            args.audio[0], layers=layers, pooling=args.pooling,
            save_path=args.out, save_format=args.format,
        )
        print(f"{args.model}: {res['num_layers']} hidden states, kept {res['layer_indices']}")
    else:
        results = extractor.extract_batch(
            args.audio, save_dir=args.out, save_format=args.format,
            layers=layers, pooling=args.pooling,
        )
        for path, r in zip(args.audio, results):
            print(f"{path}: kept layers {r['layer_indices']} -> dim {next(iter(r['pooled'].values())).shape[-1]}")


if __name__ == "__main__":
    main()
