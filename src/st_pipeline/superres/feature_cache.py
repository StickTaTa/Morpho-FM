from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from .preprocess import prepare_slide_cache
from .feature_extract import extract_embeddings


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_cache_root(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else get_project_root()
    return root / "data" / "superres_cache"


def get_common_genes(slide_ids, root: Path | None = None):
    cache_root = get_cache_root(root)
    base = None
    for sid in slide_ids:
        genes_path = cache_root / sid / "gene-names.txt"
        genes = genes_path.read_text(encoding="utf-8").splitlines()
        if base is None:
            base = genes
        else:
            gene_set = set(genes)
            base = [g for g in base if g in gene_set]
    return base or []


def load_cache_slide(slide_id: str, root: Path | None = None, genes=None):
    cache_root = get_cache_root(root)
    slide_dir = cache_root / slide_id

    # counts
    cnts_df = pd.read_csv(slide_dir / "cnts.tsv", sep="	", index_col=0)
    gene_names = list(cnts_df.columns)
    if genes is not None:
        cnts_df = cnts_df[genes]
        gene_names = list(cnts_df.columns)
    cnts = cnts_df.to_numpy().astype(np.float32)

    # locs
    locs_df = pd.read_csv(slide_dir / "locs.tsv", sep="	")
    spot_ids = locs_df["SpotID"].astype(str).to_list()
    locs = np.stack([locs_df["y"].to_numpy(), locs_df["x"].to_numpy()], axis=-1)

    # radius
    radius = float((slide_dir / "radius.txt").read_text(encoding="utf-8").strip())

    # embeddings
    with open(slide_dir / "embeddings-hist.pickle", "rb") as f:
        embs = pickle.load(f)
    embs = np.concatenate([embs["cls"], embs["sub"], embs["rgb"]])
    embs = embs.transpose(1, 2, 0)

    return embs, cnts, locs, radius, gene_names, spot_ids


def normalize_slide(embs: np.ndarray, cnts: np.ndarray):
    embs_mean = np.nanmean(embs, axis=(0, 1))
    embs_std = np.nanstd(embs, axis=(0, 1))
    embs_norm = (embs - embs_mean) / (embs_std + 1e-12)

    cnts_min = cnts.min(axis=0)
    cnts_max = cnts.max(axis=0)
    cnts_norm = (cnts - cnts_min) / (cnts_max - cnts_min + 1e-12)

    stats = {
        "embs_mean": embs_mean,
        "embs_std": embs_std,
        "cnts_min": cnts_min,
        "cnts_max": cnts_max,
    }
    return embs_norm, cnts_norm, stats


def ensure_feature_cache(
    slide_ids,
    root: Path | None = None,
    force: bool = False,
    device: str = "cuda",
    max_dim: int = 16000,
    use_tissue_seg: bool = True,
):
    root = Path(root) if root is not None else get_project_root()
    cache_root = get_cache_root(root)
    hest_dir = root / "data" / "hest_data"
    spatial_dir = root / "data" / "spatial_data"
    model_dir = root / "checkpoints" / "hipt_backbone"

    for sid in slide_ids:
        slide_dir = prepare_slide_cache(
            slide_id=sid,
            cache_dir=cache_root,
            hest_dir=hest_dir,
            spatial_dir=spatial_dir,
            max_dim=max_dim,
            use_tissue_seg=use_tissue_seg,
            force=force,
        )

        emb_path = slide_dir / "embeddings-hist.pickle"
        extract_embeddings(
            he_path=slide_dir / "he.jpg",
            out_path=emb_path,
            device=device,
            model_dir=model_dir,
            smoothen=True,
            use_shift=False,
            force=force,
        )

    return cache_root
