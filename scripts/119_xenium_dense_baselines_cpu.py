from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULT_ROOT = ROOT / "results" / "st_superres"
OUT_ROOT = RESULT_ROOT / "xenium_extended" / "dense_baseline_cpu"
PANEL29_ROOT = OUT_ROOT / "panel29"
BOUNDARY_ROOT = OUT_ROOT / "boundary_roi_xen1"
FIG_ROOT = OUT_ROOT / "figures"
PANEL29_ROOT.mkdir(parents=True, exist_ok=True)
BOUNDARY_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

GRID_FACTOR = 16
BASELINES = ("gaussian_smooth", "idw_interpolation")
METHOD_LABELS = {
    "morphofm_dense": "Morpho-FM dense",
    "gaussian_smooth": "Spot + Gaussian",
    "idw_interpolation": "Spot + IDW",
}
METHOD_COLORS = {
    "morphofm_dense": "#2878b5",
    "gaussian_smooth": "#c75d2c",
    "idw_interpolation": "#3a924a",
}
PANEL_BY_CATEGORY: dict[str, list[str]] = {
    "tumour_luminal": ["ERBB2", "ESR1", "PGR", "FOXA1", "GATA3"],
    "epithelial": ["EPCAM", "KRT7", "KRT8", "TACSTD2", "CEACAM6"],
    "proliferation": ["MKI67", "TOP2A", "CCND1", "CENPF"],
    "stromal": ["LUM", "POSTN", "MMP2", "LRRC15", "SFRP1"],
    "vascular": ["PECAM1", "VWF", "RAMP2", "AQP1", "CAV1"],
    "immune": ["PTPRC", "CD3D", "CD4", "CD8A", "C1QA"],
}
SIGNATURES = {
    "tumour_luminal": ["ERBB2", "FOXA1", "GATA3"],
    "epithelial": ["EPCAM", "KRT7", "TACSTD2", "CEACAM6"],
    "proliferation": ["CCND1"],
    "stromal_vascular": ["ACTA2", "CAV1", "AQP1"],
    "immune": ["C1QA"],
}
COARSE_MAP = {
    "Invasive_Tumor": "Tumor",
    "Prolif_Invasive_Tumor": "Tumor",
    "T_Cell_&_Tumor_Hybrid": "Tumor_immune_interface",
    "DCIS 1": "DCIS_myoepithelial",
    "DCIS 2": "DCIS_myoepithelial",
    "Myoepi_ACTA2+": "DCIS_myoepithelial",
    "Myoepi_KRT15+": "DCIS_myoepithelial",
    "Stromal": "Stromal_vascular",
    "Endothelial": "Stromal_vascular",
    "Perivascular-Like": "Stromal_vascular",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "Stromal_&_T_Cell_Hybrid": "Immune_stromal_interface",
    "Unlabeled": "Unlabeled",
}


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    setting: str
    section: str
    train_section: str
    mode: str
    result_dir: Path
    ckpt_dir: Path


RUNS = [
    RunConfig(
        run_name="XEN1_within",
        setting="XEN1 train -> XEN1 predict",
        section="XEN1",
        train_section="XEN1",
        mode="within",
        result_dir=RESULT_ROOT / "xenium_cellfm_1" / "xenium_XEN1",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
    ),
    RunConfig(
        run_name="XEN2_to_XEN1",
        setting="XEN2 train -> XEN1 predict",
        section="XEN1",
        train_section="XEN2",
        mode="cross",
        result_dir=RESULT_ROOT / "xenium_cellfm_2" / "xenium_XEN1",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
    ),
    RunConfig(
        run_name="XEN2_within",
        setting="XEN2 train -> XEN2 predict",
        section="XEN2",
        train_section="XEN2",
        mode="within",
        result_dir=RESULT_ROOT / "xenium_cellfm_2" / "xenium_XEN2",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
    ),
    RunConfig(
        run_name="XEN1_to_XEN2",
        setting="XEN1 train -> XEN2 predict",
        section="XEN2",
        train_section="XEN1",
        mode="cross",
        result_dir=RESULT_ROOT / "xenium_cellfm_1" / "xenium_XEN2",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
    ),
]

SECTION_REFERENCE_MAP = {
    "XEN1": RESULT_ROOT / "xenium_extended" / "whole_slide" / "XEN1_within" / "pred_superres_balanced_panel.npy",
    "XEN2": RESULT_ROOT / "xenium_extended" / "whole_slide" / "XEN2_within" / "pred_superres_balanced_panel.npy",
}
SECTION_BOUNDARY_DENSE_SOURCES = {
    "within_dense": RESULT_ROOT / "xenium_quant_summary" / "XEN1_within" / "pred_spot_from_superres_selected.npy",
    "cross_dense": RESULT_ROOT / "xenium_quant_summary" / "XEN1_cross" / "pred_spot_from_superres_selected.npy",
}
SECTION_META_CACHE: dict[str, dict[str, object]] = {}


def panel_records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rank = 0
    for category, genes in PANEL_BY_CATEGORY.items():
        for gene in genes:
            rows.append({"gene": gene, "category": category, "panel_rank": rank})
            rank += 1
    return pd.DataFrame(rows)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def zscore_columns(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.shape[1]):
        x = arr[:, i]
        mask = np.isfinite(x)
        if mask.sum() < 3:
            out[:, i] = np.nan
            continue
        mean = x[mask].mean()
        std = x[mask].std(ddof=0)
        if std == 0 or not np.isfinite(std):
            out[:, i] = np.nan
        else:
            out[:, i] = (x - mean) / std
    return out


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


def get_valid_coords(section: str) -> np.ndarray:
    cache_dir = ROOT / "data" / "superres_cache" / section
    locs_df = pd.read_csv(cache_dir / "locs.tsv", sep="\t")
    locs_px = np.stack([locs_df["y"].to_numpy(), locs_df["x"].to_numpy()], axis=-1).astype(np.float32)

    with open(cache_dir / "embeddings-hist.pickle", "rb") as handle:
        embs = pickle.load(handle)
    embs = np.concatenate([embs["cls"], embs["sub"], embs["rgb"]]).transpose(1, 2, 0)

    radius_px = float((cache_dir / "radius.txt").read_text(encoding="utf-8").strip())
    radius_grid = radius_px / GRID_FACTOR
    mask = get_disk_mask(radius_grid)
    shape = np.array(mask.shape)
    center = shape // 2
    offsets = np.stack([-center, shape - center], -1)
    r0, r1 = offsets[0], offsets[1]

    locs_grid = np.round(locs_px / float(GRID_FACTOR)).astype(np.int64)
    h, w = embs.shape[:2]
    valid = (
        (locs_grid[:, 0] + r0[0] >= 0)
        & (locs_grid[:, 0] + r0[1] <= h)
        & (locs_grid[:, 1] + r1[0] >= 0)
        & (locs_grid[:, 1] + r1[1] <= w)
    )
    valid &= np.isfinite(embs[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
    return locs_px[valid]


def load_section_meta(section: str) -> dict[str, object]:
    if section in SECTION_META_CACHE:
        return SECTION_META_CACHE[section]

    locs_px = get_valid_coords(section)
    locs_grid = np.round(locs_px / float(GRID_FACTOR)).astype(np.int64)
    cache_dir = ROOT / "data" / "superres_cache" / section
    radius_px = float((cache_dir / "radius.txt").read_text(encoding="utf-8").strip())
    radius_grid = radius_px / GRID_FACTOR
    disk_mask = get_disk_mask(radius_grid)

    reference_map = np.load(SECTION_REFERENCE_MAP[section], mmap_mode="r")
    valid_mask = np.isfinite(reference_map[..., 0])
    h, w = valid_mask.shape
    valid_points = np.argwhere(valid_mask)

    flat = locs_grid[:, 0] * w + locs_grid[:, 1]
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    unique_coords = np.column_stack(np.divmod(unique_flat, w)).astype(np.int64)

    meta = {
        "locs_px": locs_px,
        "locs_grid": locs_grid,
        "radius_px": float(radius_px),
        "radius_grid": float(radius_grid),
        "disk_mask": disk_mask,
        "valid_mask": valid_mask,
        "valid_points": valid_points,
        "height": int(h),
        "width": int(w),
        "unique_coords": unique_coords,
        "inverse": inverse.astype(np.int64),
    }
    SECTION_META_CACHE[section] = meta
    return meta


def aggregate_to_unique(values: np.ndarray, inverse: np.ndarray, n_unique: int) -> np.ndarray:
    sums = np.zeros((n_unique, values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values.astype(np.float64))
    counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)[:, None]
    return (sums / np.maximum(counts, 1.0)).astype(np.float32)


def build_gaussian_map(
    unique_coords: np.ndarray,
    unique_values: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    h, w = valid_mask.shape
    n_genes = unique_values.shape[1]
    value_cube = np.zeros((h, w, n_genes), dtype=np.float32)
    value_cube[unique_coords[:, 0], unique_coords[:, 1], :] = unique_values

    weight_map = np.zeros((h, w), dtype=np.float32)
    weight_map[unique_coords[:, 0], unique_coords[:, 1]] = 1.0
    smooth_values = gaussian_filter(value_cube, sigma=(sigma, sigma, 0.0), mode="nearest")
    smooth_weights = gaussian_filter(weight_map, sigma=sigma, mode="nearest")[..., None]

    dense = np.full((h, w, n_genes), np.nan, dtype=np.float32)
    finite = smooth_weights[..., 0] > 1e-6
    dense[finite] = (smooth_values[finite] / smooth_weights[finite]).astype(np.float32)
    dense[~valid_mask] = np.nan
    return dense


def build_idw_map(
    unique_coords: np.ndarray,
    unique_values: np.ndarray,
    valid_points: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
    power: float,
    chunk_size: int,
) -> np.ndarray:
    h, w = valid_mask.shape
    n_genes = unique_values.shape[1]
    tree = cKDTree(unique_coords.astype(np.float64))
    k_eff = int(min(k, len(unique_coords)))
    dense = np.full((h, w, n_genes), np.nan, dtype=np.float32)

    for start in range(0, len(valid_points), chunk_size):
        end = min(start + chunk_size, len(valid_points))
        pts = valid_points[start:end].astype(np.float64)
        dists, idx = tree.query(pts, k=k_eff, workers=-1)
        if k_eff == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        weights = 1.0 / np.maximum(dists, 1e-6) ** power
        exact = dists < 1e-6
        if np.any(exact):
            rows = np.where(exact.any(axis=1))[0]
            for row in rows:
                mask = exact[row]
                weights[row] = mask.astype(np.float64)
        weights /= weights.sum(axis=1, keepdims=True)
        pred = np.sum(unique_values[idx] * weights[..., None], axis=1).astype(np.float32)
        dense[pts[:, 0].astype(int), pts[:, 1].astype(int), :] = pred
        print(f"[idw] filled {end}/{len(valid_points)} grid points")

    dense[~valid_mask] = np.nan
    return dense


def reaggregate_dense_map(
    dense_map: np.ndarray,
    locs_grid: np.ndarray,
    disk_mask: np.ndarray,
) -> np.ndarray:
    n_obs = len(locs_grid)
    n_genes = dense_map.shape[-1]
    out = np.full((n_obs, n_genes), np.nan, dtype=np.float32)
    if int(disk_mask.sum()) == 1:
        out[:] = dense_map[locs_grid[:, 0], locs_grid[:, 1], :]
        return out

    mask_h, mask_w = disk_mask.shape
    cy, cx = mask_h // 2, mask_w // 2
    h, w = dense_map.shape[:2]
    for i, (yy, xx) in enumerate(locs_grid):
        y0, y1 = yy - cy, yy - cy + mask_h
        x0, x1 = xx - cx, xx - cx + mask_w
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        patch = dense_map[y0:y1, x0:x1]
        vals = patch[disk_mask]
        out[i] = np.nanmean(vals, axis=0)
    return out


def load_run_arrays(run: RunConfig, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    common_genes = json.loads((run.ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    idx = [gene_to_idx[gene] for gene in genes]
    pred = np.load(run.result_dir / "pred_spot.npy", mmap_mode="r")[:, idx].astype(np.float32)
    true = np.load(run.result_dir / "true_spot.npy", mmap_mode="r")[:, idx].astype(np.float32)
    return pred, true


def build_baseline_maps_for_run(
    run: RunConfig,
    genes: list[str],
    sigma: float,
    idw_k: int,
    idw_power: float,
    chunk_size: int,
    force: bool,
    out_root: Path,
) -> dict[str, object]:
    meta = load_section_meta(run.section)
    run_dir = out_root / run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_spot, true_spot = load_run_arrays(run, genes)
    unique_values = aggregate_to_unique(
        pred_spot,
        inverse=np.asarray(meta["inverse"], dtype=np.int64),
        n_unique=len(np.asarray(meta["unique_coords"])),
    )

    rows: list[dict[str, object]] = []
    for baseline in BASELINES:
        dense_path = run_dir / f"{baseline}_whole_slide.npy"
        backproj_path = run_dir / f"{baseline}_backproj.npy"
        if not force and dense_path.exists() and backproj_path.exists():
            dense_map = np.load(dense_path, mmap_mode="r")
            backproj = np.load(backproj_path, mmap_mode="r")
        else:
            if baseline == "gaussian_smooth":
                dense_map = build_gaussian_map(
                    unique_coords=np.asarray(meta["unique_coords"], dtype=np.int64),
                    unique_values=unique_values,
                    valid_mask=np.asarray(meta["valid_mask"], dtype=bool),
                    sigma=sigma,
                )
            elif baseline == "idw_interpolation":
                dense_map = build_idw_map(
                    unique_coords=np.asarray(meta["unique_coords"], dtype=np.int64),
                    unique_values=unique_values,
                    valid_points=np.asarray(meta["valid_points"], dtype=np.int64),
                    valid_mask=np.asarray(meta["valid_mask"], dtype=bool),
                    k=idw_k,
                    power=idw_power,
                    chunk_size=chunk_size,
                )
            else:
                raise ValueError(baseline)
            backproj = reaggregate_dense_map(
                dense_map=dense_map,
                locs_grid=np.asarray(meta["locs_grid"], dtype=np.int64),
                disk_mask=np.asarray(meta["disk_mask"], dtype=bool),
            )
            np.save(dense_path, dense_map)
            np.save(backproj_path, backproj)
            print(f"[saved] {dense_path}")
            print(f"[saved] {backproj_path}")

        for gi, gene in enumerate(genes):
            rows.append(
                {
                    "run_name": run.run_name,
                    "setting": run.setting,
                    "section": run.section,
                    "train_section": run.train_section,
                    "mode": run.mode,
                    "method": baseline,
                    "gene": gene,
                    "pearson": pearson_corr(backproj[:, gi], true_spot[:, gi]),
                }
            )
    return {
        "rows": rows,
        "true_spot": true_spot,
    }


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def plot_panel29_distribution(detail_df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    setting_order = [run.setting for run in RUNS]
    method_order = ["morphofm_dense", "gaussian_smooth", "idw_interpolation"]
    fig, axes = plt.subplots(1, len(setting_order), figsize=(4.6 * len(setting_order), 4.8), sharey=True)
    if len(setting_order) == 1:
        axes = [axes]
    for ax, setting in zip(axes, setting_order):
        sub = detail_df[detail_df["setting"] == setting]
        pos_base = np.arange(len(method_order)) + 1
        data = [
            sub[sub["method"] == method]["pearson"].to_numpy(dtype=float)
            for method in method_order
        ]
        bp = ax.boxplot(
            data,
            positions=pos_base,
            widths=0.65,
            patch_artist=True,
            showfliers=False,
        )
        for patch, method in zip(bp["boxes"], method_order):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        ax.set_xticks(pos_base)
        ax.set_xticklabels([METHOD_LABELS[m] for m in method_order], rotation=25, ha="right", fontsize=9)
        ax.set_title(setting, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("29-gene backprojected Pearson")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_signed_tumour_distance(ann: pd.DataFrame, coords: np.ndarray) -> np.ndarray:
    tumour = ann["coarse_compartment"].to_numpy() == "Tumor"
    non_tumour = ~tumour
    valid = np.isfinite(coords).all(axis=1)
    tumour &= valid
    non_tumour &= valid
    tumour_tree = cKDTree(coords[tumour])
    non_tumour_tree = cKDTree(coords[non_tumour])
    dist = np.full(len(coords), np.nan, dtype=np.float64)
    dist[tumour] = non_tumour_tree.query(coords[tumour], k=1)[0]
    dist[non_tumour] = -tumour_tree.query(coords[non_tumour], k=1)[0]
    return dist


def source_signature_scores(sources: dict[str, np.ndarray], genes: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    gene_pos = {gene: i for i, gene in enumerate(genes)}
    rows = []
    score_tables: dict[str, pd.DataFrame] = {}
    for source_name, values in sources.items():
        z = zscore_columns(values)
        sig_cols = {}
        for signature, sig_genes in SIGNATURES.items():
            available = [gene for gene in sig_genes if gene in gene_pos]
            if not available:
                continue
            idx = [gene_pos[gene] for gene in available]
            sig_cols[signature] = np.nanmean(z[:, idx], axis=1)
            rows.append({"source": source_name, "signature": signature, "genes": ";".join(available), "n_genes": len(available)})
        score_tables[source_name] = pd.DataFrame(sig_cols)
    return score_tables, pd.DataFrame(rows)


def boundary_gradient(signed_distance: np.ndarray, scores: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.array([-400, -250, -150, -100, -50, -25, 0, 25, 50, 100, 150, 250, 400], dtype=float)
    bin_ids = pd.cut(signed_distance, bins=bins, include_lowest=True)
    gradient_rows = []
    for source, df in scores.items():
        for signature in df.columns:
            vals = df[signature].to_numpy()
            for interval in bin_ids.categories:
                mask = np.asarray(bin_ids == interval) & np.isfinite(vals)
                if mask.sum() == 0:
                    mean = sem = np.nan
                else:
                    mean = float(vals[mask].mean())
                    sem = float(vals[mask].std(ddof=0) / np.sqrt(mask.sum()))
                gradient_rows.append(
                    {
                        "source": source,
                        "signature": signature,
                        "distance_bin": str(interval),
                        "distance_mid_um": float((interval.left + interval.right) / 2),
                        "n_cells": int(mask.sum()),
                        "mean_zscore": mean,
                        "sem_zscore": sem,
                    }
                )
    gradient_df = pd.DataFrame(gradient_rows)
    agreement_rows = []
    measured = gradient_df[gradient_df["source"] == "measured"]
    for source in [name for name in scores if name != "measured"]:
        pred = gradient_df[gradient_df["source"] == source]
        for signature in measured["signature"].unique():
            m = measured[measured["signature"] == signature].sort_values("distance_mid_um")
            p = pred[pred["signature"] == signature].sort_values("distance_mid_um")
            merged = m[["distance_mid_um", "mean_zscore"]].merge(
                p[["distance_mid_um", "mean_zscore"]],
                on="distance_mid_um",
                suffixes=("_measured", "_predicted"),
            )
            agreement_rows.append(
                {
                    "source": source,
                    "signature": signature,
                    "gradient_pearson": pearson_corr(
                        merged["mean_zscore_measured"].to_numpy(),
                        merged["mean_zscore_predicted"].to_numpy(),
                    ),
                    "gradient_mae": float(
                        np.nanmean(np.abs(merged["mean_zscore_measured"] - merged["mean_zscore_predicted"]))
                    ),
                }
            )
    return gradient_df, pd.DataFrame(agreement_rows)


def make_roi_masks(ann: pd.DataFrame, signed_distance: np.ndarray, measured: np.ndarray, genes: list[str]) -> dict[str, np.ndarray]:
    coarse = ann["coarse_compartment"].to_numpy()
    erbb2_idx = genes.index("ERBB2") if "ERBB2" in genes else None
    roi_masks: dict[str, np.ndarray] = {
        "tumour_core": (coarse == "Tumor") & (signed_distance >= 100),
        "tumour_boundary_band": np.abs(signed_distance) <= 50,
        "adjacent_stroma": (coarse == "Stromal_vascular") & (signed_distance >= -150) & (signed_distance < 0),
        "dcis_myoepithelial": coarse == "DCIS_myoepithelial",
        "stromal_vascular": coarse == "Stromal_vascular",
        "immune_enriched": coarse == "Immune",
    }
    if erbb2_idx is not None:
        measured_erbb2 = measured[:, erbb2_idx]
        tumour = coarse == "Tumor"
        cutoff = np.nanpercentile(measured_erbb2[tumour], 75)
        roi_masks["erbb2_high_tumour"] = tumour & (measured_erbb2 >= cutoff)

    try:
        roi_df = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name="Fig. 4b-d ")
        barcode_to_index = pd.Series(np.arange(len(ann)), index=ann["Barcode"].astype(int)).to_dict()
        for roi_name, sub in roi_df.groupby("ROI_Membership"):
            if pd.isna(roi_name):
                continue
            idx = [barcode_to_index.get(int(bc)) for bc in sub["Barcode"].dropna()]
            idx = np.array([i for i in idx if i is not None], dtype=int)
            if idx.size < 50:
                continue
            mask = np.zeros(len(ann), dtype=bool)
            mask[idx] = True
            roi_masks[f"named_{str(roi_name).replace(' ', '_')}"] = mask
    except Exception as exc:
        print(f"[warn] could not load named ROIs: {exc}")
    return roi_masks


def roi_enrichment(scores: dict[str, pd.DataFrame], roi_masks: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for roi_name, roi_mask in roi_masks.items():
        if roi_mask.sum() < 20:
            continue
        bg_mask = ~roi_mask
        for source, df in scores.items():
            for signature in df.columns:
                vals = df[signature].to_numpy()
                roi_vals = vals[roi_mask & np.isfinite(vals)]
                bg_vals = vals[bg_mask & np.isfinite(vals)]
                if roi_vals.size == 0 or bg_vals.size == 0:
                    continue
                rows.append(
                    {
                        "roi": roi_name,
                        "source": source,
                        "signature": signature,
                        "n_roi_cells": int(roi_vals.size),
                        "roi_mean_zscore": float(roi_vals.mean()),
                        "background_mean_zscore": float(bg_vals.mean()),
                        "enrichment_delta": float(roi_vals.mean() - bg_vals.mean()),
                    }
                )
    enrich_df = pd.DataFrame(rows)
    corr_rows = []
    measured = enrich_df[enrich_df["source"] == "measured"]
    for source in sorted(set(enrich_df["source"]) - {"measured"}):
        pred = enrich_df[enrich_df["source"] == source]
        merged = measured[["roi", "signature", "enrichment_delta"]].merge(
            pred[["roi", "signature", "enrichment_delta"]],
            on=["roi", "signature"],
            suffixes=("_measured", "_predicted"),
        )
        corr_rows.append(
            {
                "source": source,
                "n_roi_signature_pairs": int(len(merged)),
                "roi_enrichment_pearson": pearson_corr(
                    merged["enrichment_delta_measured"].to_numpy(),
                    merged["enrichment_delta_predicted"].to_numpy(),
                ),
                "roi_enrichment_mae": float(
                    np.nanmean(np.abs(merged["enrichment_delta_measured"] - merged["enrichment_delta_predicted"]))
                ),
            }
        )
    return enrich_df, pd.DataFrame(corr_rows)


def plot_boundary_roi_summary(
    gradient_agree: pd.DataFrame,
    roi_agree: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    summary = (
        gradient_agree.assign(mode=lambda df: df["source"].str.split("_").str[0], method=lambda df: df["source"].str.split("_", n=1).str[1])
        .groupby(["mode", "method"], as_index=False)[["gradient_pearson", "gradient_mae"]]
        .mean()
    )
    roi_summary = roi_agree.assign(mode=lambda df: df["source"].str.split("_").str[0], method=lambda df: df["source"].str.split("_", n=1).str[1])

    mode_order = ["within", "cross"]
    method_order = ["dense", "gaussian_smooth", "idw_interpolation"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    width = 0.24
    x = np.arange(len(mode_order))

    ax = axes[0]
    for idx, method in enumerate(method_order):
        vals = []
        for mode in mode_order:
            row = summary[(summary["mode"] == mode) & (summary["method"] == method)]
            vals.append(float(row["gradient_pearson"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 1) * width, vals, width=width, color=METHOD_COLORS["morphofm_dense" if method == "dense" else method], label=METHOD_LABELS["morphofm_dense" if method == "dense" else method])
    ax.set_xticks(x)
    ax.set_xticklabels(mode_order)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean boundary-gradient Pearson")
    ax.set_title("Boundary gradient agreement")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for idx, method in enumerate(method_order):
        vals = []
        for mode in mode_order:
            row = roi_summary[(roi_summary["mode"] == mode) & (roi_summary["method"] == method)]
            vals.append(float(row["roi_enrichment_pearson"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 1) * width, vals, width=width, color=METHOD_COLORS["morphofm_dense" if method == "dense" else method], label=METHOD_LABELS["morphofm_dense" if method == "dense" else method])
    ax.set_xticks(x)
    ax.set_xticklabels(mode_order)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ROI-enrichment Pearson")
    ax.set_title("ROI enrichment agreement")
    ax.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_panel29_baselines(args: argparse.Namespace) -> None:
    panel_df = panel_records()
    panel_df.to_csv(PANEL29_ROOT / "xenium_balanced_panel_genes.csv", index=False)
    genes = panel_df["gene"].tolist()
    category_by_gene = dict(zip(panel_df["gene"], panel_df["category"]))

    ours = pd.read_csv(RESULT_ROOT / "xenium_extended" / "panel" / "xenium_balanced_panel_metrics_detail.csv")
    ours = ours[["run_name", "setting", "section", "train_section", "mode", "gene", "category", "dense_backproj_pearson"]].rename(
        columns={"dense_backproj_pearson": "pearson"}
    )
    ours["method"] = "morphofm_dense"

    baseline_rows: list[dict[str, object]] = []
    for run in RUNS:
        result = build_baseline_maps_for_run(
            run=run,
            genes=genes,
            sigma=args.gaussian_sigma,
            idw_k=args.idw_k,
            idw_power=args.idw_power,
            chunk_size=args.idw_chunk_size,
            force=args.force,
            out_root=PANEL29_ROOT,
        )
        run_rows = result["rows"]
        for row in run_rows:
            row["category"] = category_by_gene[str(row["gene"])]
        baseline_rows.extend(run_rows)

    detail_df = pd.concat([ours, pd.DataFrame(baseline_rows)], ignore_index=True)
    detail_df.to_csv(PANEL29_ROOT / "xenium_baseline_29gene_backproj_detail.csv", index=False)

    summary_rows = []
    for (run_name, setting, section, train_section, mode, method), sub in detail_df.groupby(
        ["run_name", "setting", "section", "train_section", "mode", "method"], sort=False
    ):
        summary_rows.append(
            {
                "run_name": run_name,
                "setting": setting,
                "section": section,
                "train_section": train_section,
                "mode": mode,
                "method": method,
                **summarize_distribution(sub["pearson"].to_numpy()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(PANEL29_ROOT / "xenium_baseline_29gene_backproj_setting_summary.csv", index=False)

    cat_rows = []
    for (run_name, setting, section, train_section, mode, method, category), sub in detail_df.groupby(
        ["run_name", "setting", "section", "train_section", "mode", "method", "category"], sort=False
    ):
        cat_rows.append(
            {
                "run_name": run_name,
                "setting": setting,
                "section": section,
                "train_section": train_section,
                "mode": mode,
                "method": method,
                "category": category,
                **summarize_distribution(sub["pearson"].to_numpy()),
            }
        )
    pd.DataFrame(cat_rows).to_csv(PANEL29_ROOT / "xenium_baseline_29gene_backproj_category_summary.csv", index=False)

    plot_panel29_distribution(
        detail_df=detail_df,
        out_pdf=FIG_ROOT / "xenium_baseline_29gene_backproj_distribution.pdf",
        out_png=FIG_ROOT / "xenium_baseline_29gene_backproj_distribution.png",
    )


def build_boundary_sources(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], list[str]]:
    ann = pd.read_excel(ROOT / "notebooks" / "breast_cancer_info.xlsx", sheet_name="Fig. 3e-j Xenium")
    ann["fine_label"] = ann["Cluster"].astype(str)
    ann["coarse_compartment"] = ann["fine_label"].map(COARSE_MAP).fillna("Other")
    coords = ann[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)

    selected_genes = json.loads((RESULT_ROOT / "xenium_quant_summary" / "XEN1_within" / "selected_genes.json").read_text(encoding="utf-8"))
    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8"))
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    meta = load_section_meta("XEN1")

    true_all = np.load(RESULT_ROOT / "xenium_cellfm_1" / "xenium_XEN1" / "true_spot.npy", mmap_mode="r")
    measured = np.column_stack([true_all[:, gene_to_idx[gene]] for gene in selected_genes]).astype(np.float32)
    sources: dict[str, np.ndarray] = {"measured": measured}
    sources["within_dense"] = np.load(SECTION_BOUNDARY_DENSE_SOURCES["within_dense"], mmap_mode="r").astype(np.float32)
    sources["cross_dense"] = np.load(SECTION_BOUNDARY_DENSE_SOURCES["cross_dense"], mmap_mode="r").astype(np.float32)

    boundary_runs = {
        "within": RUNS[0],
        "cross": RUNS[1],
    }
    for mode, run in boundary_runs.items():
        run_dir = BOUNDARY_ROOT / run.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_spot, _ = load_run_arrays(run, selected_genes)
        unique_values = aggregate_to_unique(
            pred_spot,
            inverse=np.asarray(meta["inverse"], dtype=np.int64),
            n_unique=len(np.asarray(meta["unique_coords"])),
        )
        for baseline in BASELINES:
            dense_path = run_dir / f"{baseline}_whole_slide_selected.npy"
            backproj_path = run_dir / f"{baseline}_backproj_selected.npy"
            if not args.force and dense_path.exists() and backproj_path.exists():
                backproj = np.load(backproj_path, mmap_mode="r")
            else:
                if baseline == "gaussian_smooth":
                    dense_map = build_gaussian_map(
                        unique_coords=np.asarray(meta["unique_coords"], dtype=np.int64),
                        unique_values=unique_values,
                        valid_mask=np.asarray(meta["valid_mask"], dtype=bool),
                        sigma=args.gaussian_sigma,
                    )
                else:
                    dense_map = build_idw_map(
                        unique_coords=np.asarray(meta["unique_coords"], dtype=np.int64),
                        unique_values=unique_values,
                        valid_points=np.asarray(meta["valid_points"], dtype=np.int64),
                        valid_mask=np.asarray(meta["valid_mask"], dtype=bool),
                        k=args.idw_k,
                        power=args.idw_power,
                        chunk_size=args.idw_chunk_size,
                    )
                backproj = reaggregate_dense_map(
                    dense_map=dense_map,
                    locs_grid=np.asarray(meta["locs_grid"], dtype=np.int64),
                    disk_mask=np.asarray(meta["disk_mask"], dtype=bool),
                )
                np.save(dense_path, dense_map)
                np.save(backproj_path, backproj)
                print(f"[saved] {dense_path}")
                print(f"[saved] {backproj_path}")
            sources[f"{mode}_{baseline}"] = np.asarray(backproj, dtype=np.float32)
    return ann, coords, sources, selected_genes


def run_boundary_roi_baselines(args: argparse.Namespace) -> None:
    ann, coords, sources, genes = build_boundary_sources(args)
    signed_distance = compute_signed_tumour_distance(ann, coords)
    ann_out = ann[["Barcode", "fine_label", "coarse_compartment", "x_centroid", "y_centroid"]].copy()
    ann_out["signed_distance_to_tumour_edge_um"] = signed_distance
    ann_out.to_csv(BOUNDARY_ROOT / "xenium_xen1_annotation_with_boundary_distance.csv", index=False)

    scores, signature_manifest = source_signature_scores(sources, genes)
    signature_manifest.to_csv(BOUNDARY_ROOT / "xenium_xen1_signature_manifest.csv", index=False)

    gradient_df, gradient_agree = boundary_gradient(signed_distance, scores)
    gradient_df.to_csv(BOUNDARY_ROOT / "xenium_xen1_boundary_gradient_bins.csv", index=False)
    gradient_agree.to_csv(BOUNDARY_ROOT / "xenium_xen1_boundary_gradient_agreement.csv", index=False)

    roi_masks = make_roi_masks(ann, signed_distance, sources["measured"], genes)
    roi_summary = pd.DataFrame([{"roi": name, "n_cells": int(mask.sum())} for name, mask in roi_masks.items()]).sort_values("roi")
    roi_summary.to_csv(BOUNDARY_ROOT / "xenium_xen1_roi_definitions.csv", index=False)

    enrich_df, enrich_agree = roi_enrichment(scores, roi_masks)
    enrich_df.to_csv(BOUNDARY_ROOT / "xenium_xen1_roi_signature_enrichment.csv", index=False)
    enrich_agree.to_csv(BOUNDARY_ROOT / "xenium_xen1_roi_signature_enrichment_agreement.csv", index=False)

    plot_boundary_roi_summary(
        gradient_agree=gradient_agree,
        roi_agree=enrich_agree,
        out_pdf=FIG_ROOT / "xenium_xen1_boundary_roi_baseline_summary.pdf",
        out_png=FIG_ROOT / "xenium_xen1_boundary_roi_baseline_summary.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute minimal Xenium dense reconstruction baselines on CPU.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian sigma in small-grid pixels.")
    parser.add_argument("--idw-k", type=int, default=8, help="Number of nearest neighbours for IDW interpolation.")
    parser.add_argument("--idw-power", type=float, default=2.0, help="Distance power for IDW interpolation.")
    parser.add_argument("--idw-chunk-size", type=int, default=50000, help="Chunk size for IDW grid filling.")
    parser.add_argument("--force", action="store_true", help="Recompute baseline maps even if cached outputs exist.")
    parser.add_argument("--skip-panel29", action="store_true", help="Skip the four-setting 29-gene comparison.")
    parser.add_argument("--skip-boundary-roi", action="store_true", help="Skip XEN1 boundary/ROI baseline comparison.")
    return parser.parse_args()


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    args = parse_args()
    print("ROOT:", ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print(f"[config] gaussian_sigma={args.gaussian_sigma} | idw_k={args.idw_k} | idw_power={args.idw_power}")
    if not args.skip_panel29:
        run_panel29_baselines(args)
    if not args.skip_boundary_roi:
        run_boundary_roi_baselines(args)
    print("[done] Xenium dense reconstruction baselines written to", OUT_ROOT)


if __name__ == "__main__":
    main()
