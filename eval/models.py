"""
models.py

Model-loading utilities for the embedding evaluation pipeline.

Every model is exposed behind a uniform `EmbeddingModel` interface with a
`.embed(lon, lat) -> np.ndarray` method, so eval_embeddings.py never needs to
know what's actually inside a given model.

Two ways to register a model:
  1. `make_satclip_loader(...)` — for SatCLIP checkpoints, using the same
     `load.get_satclip` call shown in the SatCLIP demo notebook.
  2. `make_custom_loader(...)` — wrap literally anything else (your Global
     Location Encoder, a competitor model, a random baseline, ...) as long
     as you can write a function `(lon, lat) -> embeddings`.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

sys.path.append(r"D:\Code\satclip\satclip") 
from positional_encoding import SphericalHarmonics


@dataclass
class EmbeddingModel:
    """Wraps a location encoder behind a uniform `.embed()` interface."""

    name: str
    embed_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def embed(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return self.embed_fn(lon, lat)


def make_satclip_loader(
    name: str,
    ckpt_path: str,
    satclip_repo_path: str = "./satclip",
    device: Optional[str] = None,
    batch_size: int = 4096,
    spherical_harmonics: int = 0, # [0: no, 1: only sh, 2: sh + model]
) -> EmbeddingModel:
    """Build an EmbeddingModel around a SatCLIP checkpoint.

    Mirrors the demo notebook:
        model = get_satclip(ckpt_path, device=device)
        emb = model(coords.double().to(device))
    where `coords` is an (N, 2) tensor of (lon, lat) -- note lon first, then
    lat, matching SatCLIP's convention.
    """
    if satclip_repo_path not in sys.path:
        sys.path.append(satclip_repo_path)
    from load import get_satclip  # requires the cloned satclip repo on path

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_satclip(ckpt_path, device=device)
    model.eval()

    def embed_fn(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        if spherical_harmonics>0:
            sh = SphericalHarmonics(legendre_polys=10, harmonics_calculation="analytic")
            sh.to(device)
            coords = np.stack([lon, lat], axis=1)
            with torch.no_grad():
                t = torch.from_numpy(coords).double().to(device)
                sh_emb = sh(t).detach().cpu().numpy()
            if spherical_harmonics==1:
                return sh_emb
            else:
                coords = np.stack([lon, lat], axis=1)
                chunks = []
                with torch.no_grad():
                    for start in range(0, len(coords), batch_size):
                        chunk = coords[start : start + batch_size]
                        t = torch.from_numpy(chunk).double().to(device)
                        emb = model(t).detach().cpu().numpy()
                        chunks.append(np.concatenate([sh_emb[start : start + batch_size], emb], axis=1))
                return np.concatenate(chunks, axis=0)
        coords = np.stack([lon, lat], axis=1)
        chunks = []

        with torch.no_grad():
            for start in range(0, len(coords), batch_size):
                chunk = coords[start : start + batch_size]
                t = torch.from_numpy(chunk).double().to(device)
                emb = model(t).detach().cpu().numpy()
                chunks.append(emb)
        return np.concatenate(chunks, axis=0)

    return EmbeddingModel(name=name, embed_fn=embed_fn)


def make_custom_loader(
    name: str, embed_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> EmbeddingModel:
    """Wrap any other encoder as an EmbeddingModel.

    `embed_fn` must accept 1-D numpy arrays `(lon, lat)` and return an
    `(N, D)` numpy array of embeddings, in the same row order as the input.
    Use this for e.g. your Global Location Encoder, or any sklearn-style
    model you already have loaded in memory.
    """
    return EmbeddingModel(name=name, embed_fn=embed_fn)