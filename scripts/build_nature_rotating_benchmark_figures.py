"""Build Nature-style rotating single-slide benchmark figures.

This script upgrades the original one-split benchmark displays to the
rotating single-slide setting while preserving the old notebook evidence
structure: per-gene violin, threshold curve, HVG dense-K trajectory and
representative spatial reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
DEFAULT_METRICS = ROOT / "results" / "benchmark_audit" / "benchmark_unified_metrics_with_morphofm.csv"
DEFAULT_REPO_OUT = ROOT / "results" / "paper_figures" / "rotating_single_benchmark_nature"
DEFAULT_PAPER_OUT = PAPER_DIR / "Figure" / "benchmark_rebuild" / "rotating_single_benchmark_nature"

NPG10 = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

METHOD_ORDER = ["Morpho-FM", "iStar", "sCellST", "HiST", "mclSTExp", "HisToGene"]
METHOD_COLORS = dict(zip(METHOD_ORDER, NPG10[: len(METHOD_ORDER)]))
PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
KIDNEY_SLIDES = ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"]
REPRESENTATIVE = {
    "Prostate": {"train": "INT25", "test": "INT28", "gene": "KLK3", "best_baseline": "iStar"},
    "Kidney": {"train": "INT13", "test": "INT24", "gene": "B2M", "best_baseline": "iStar"},
}
TEXT = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--repo-out", type=Path, default=DEFAULT_REPO_OUT)
    parser.add_argument("--paper-out", type=Path, default=DEFAULT_PAPER_OUT)
    parser.add_argument("--supplement-prostate", action="store_true", default=True)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def save_fig(fig: plt.Figure, outdirs: list[Path], name: str) -> None:
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(outdir / f"{name}.{suffix}", bbox_inches="tight")


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    for col in ["Mean_Pearson", "Median_Pearson", "HVG1000_Mean_Pearson", "MAE", "RMSE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["cohort", "method", "train_slide", "test_slide"])


def common_genes() -> list[str]:
    return (ROOT / "data" / "spatial_data" / "common_genes.txt").read_text(encoding="utf-8").splitlines()


def morphofm_gene_list(cohort: str, n_genes: int) -> list[str] | None:
    if cohort == "Prostate":
        slides = PROSTATE_SLIDES
        ckpt = ROOT / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json"
    elif cohort == "Kidney":
        slides = KIDNEY_SLIDES
        ckpt = ROOT / "checkpoints" / "kidney_hest" / "st_superres" / "cellfm_single_INT18_to_INT21" / "common_genes.json"
    else:
        return None

    if ckpt.exists():
        genes = json.loads(ckpt.read_text(encoding="utf-8"))
        if len(genes) == n_genes:
            return genes

    cache_root = ROOT / "data" / "superres_cache"
    base = None
    for slide in slides:
        path = cache_root / slide / "gene-names.txt"
        if not path.exists():
            return None
        genes = path.read_text(encoding="utf-8").splitlines()
        if base is None:
            base = genes
        else:
            gene_set = set(genes)
            base = [gene for gene in base if gene in gene_set]
    if base is None:
        return None

    vocab_path = ROOT / "assets" / "cellfm" / "expand_gene_info.csv"
    if not vocab_path.exists():
        return None
    vocab = set(pd.read_csv(vocab_path, index_col=0).index.astype(str))
    genes = [gene for gene in base if gene in vocab]
    return genes if len(genes) == n_genes else None


def as_result_path(row: pd.Series) -> Path:
    return Path(str(row["result_dir"]))


def split_rows(df: pd.DataFrame, cohort: str, train: str, test: str) -> pd.DataFrame:
    sub = df[
        df["cohort"].astype(str).eq(cohort)
        & df["train_slide"].astype(str).eq(train)
        & df["test_slide"].astype(str).eq(test)
    ].copy()
    return sub.sort_values("method")


def load_corr_array(result_dir: Path) -> np.ndarray:
    for name in ["gene_corr_all.npy", "unified_gene_corr_all.npy"]:
        path = result_dir / name
        if path.exists():
            arr = np.load(path)
            return arr.astype(float, copy=False)
    raise FileNotFoundError(f"No per-gene correlation array in {result_dir}")


def load_hvg_indices(result_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = [
        ("hvg_idx_true.npy", "hvg_idx_pred.npy"),
        ("unified_hvg_idx_true.npy", "unified_hvg_idx_pred.npy"),
    ]
    for true_name, pred_name in candidates:
        true_path = result_dir / true_name
        pred_path = result_dir / pred_name
        if true_path.exists() and pred_path.exists():
            return np.load(true_path), np.load(pred_path)
    return None


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    a = a[valid]
    b = b[valid]
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def protocol_matrix(df: pd.DataFrame, cohort: str, slides: list[str]) -> pd.DataFrame:
    sub = df[(df["cohort"].astype(str).eq(cohort)) & (df["method"].astype(str).eq("Morpho-FM"))]
    mat = pd.DataFrame(np.nan, index=slides, columns=slides)
    for _, row in sub.iterrows():
        train = str(row["train_slide"])
        test = str(row["test_slide"])
        if train in mat.index and test in mat.columns:
            mat.loc[train, test] = row["Mean_Pearson"]
    return mat


def draw_protocol_heatmap(ax: plt.Axes, df: pd.DataFrame, cohort: str, slides: list[str], letter: str) -> None:
    mat = protocol_matrix(df, cohort, slides)
    arr = mat.to_numpy(dtype=float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("coverage", ["#F4F4F4", "#B9D7D5", "#2B7C8F", "#155A6A"])
    cmap.set_bad("#EFEFEF")
    im = ax.imshow(np.ma.masked_invalid(arr), vmin=0.08, vmax=0.38, cmap=cmap)
    ax.set_xticks(range(len(slides)))
    ax.set_yticks(range(len(slides)))
    ax.set_xticklabels(slides, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(slides, fontsize=6.5)
    ax.set_xlabel("Test slide", fontsize=6.8)
    ax.set_ylabel("Training slide", fontsize=6.8)
    ax.set_title(f"{letter}  {cohort} train-test matrix", loc="left", fontsize=8.2, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(slides), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(slides), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(slides)):
        for j in range(len(slides)):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color=("white" if value > 0.24 else TEXT))
            elif i == j:
                ax.text(j, i, "train", ha="center", va="center", fontsize=5.4, color="#777777")
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5, width=0.4)
    cbar.set_label("r", fontsize=6.0)


def draw_direction_summary(ax: plt.Axes, df: pd.DataFrame, cohort: str, letter: str) -> None:
    sub = df[df["cohort"].astype(str).eq(cohort)]
    rng = np.random.default_rng(12)
    for i, method in enumerate(METHOD_ORDER):
        values = sub[sub["method"].astype(str).eq(method)]["Mean_Pearson"].dropna().to_numpy()
        if values.size == 0:
            continue
        color = METHOD_COLORS[method]
        ax.boxplot(
            values,
            positions=[i],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            boxprops={"facecolor": color, "alpha": 0.18, "edgecolor": color, "linewidth": 0.8},
            medianprops={"color": TEXT, "linewidth": 0.9},
            whiskerprops={"color": color, "linewidth": 0.65},
            capprops={"color": color, "linewidth": 0.65},
        )
        x = i + rng.normal(0, 0.055, size=values.size)
        ax.scatter(x, values, s=9 if values.size > 20 else 16, color=color, alpha=0.65, linewidths=0, zorder=3)
        ax.scatter([i], [values.mean()], s=30, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_title(f"{letter}  Across held-out directions", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=33, ha="right", fontsize=6.5)
    ax.set_ylabel("Mean per-gene Pearson r", fontsize=7)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)
    n = sub.groupby("method", observed=True).size()
    ax.text(
        0.0,
        -0.38,
        "directions: " + ", ".join(f"{m}={int(n.get(m, 0))}" for m in METHOD_ORDER),
        transform=ax.transAxes,
        fontsize=5.4,
        color="#666666",
        ha="left",
    )


def rep_corr_payload(df: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rep = REPRESENTATIVE[cohort]
    sub = split_rows(df, cohort, rep["train"], rep["test"])
    arrays = {}
    for _, row in sub.iterrows():
        method = str(row["method"])
        arrays[method] = load_corr_array(as_result_path(row))
    return sub, arrays


def draw_violin(ax: plt.Axes, corr_arrays: dict[str, np.ndarray], cohort: str, letter: str) -> None:
    data = []
    labels = []
    colors = []
    for method in METHOD_ORDER:
        if method not in corr_arrays:
            continue
        arr = corr_arrays[method]
        arr = arr[np.isfinite(arr)]
        data.append(arr)
        labels.append(method)
        colors.append(METHOD_COLORS[method])
    parts = ax.violinplot(data, positions=np.arange(len(data)), widths=0.78, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(0.8)
    for i, arr in enumerate(data):
        q1, med, q3 = np.nanquantile(arr, [0.25, 0.5, 0.75])
        ax.plot([i - 0.22, i + 0.22], [med, med], color=TEXT, lw=0.8)
        ax.plot([i, i], [q1, q3], color=TEXT, lw=1.6, solid_capstyle="round")
    rep = REPRESENTATIVE[cohort]
    ax.set_title(f"{letter}  Per-gene Pearson ({rep['train']}->{rep['test']})", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=33, ha="right", fontsize=6.4)
    ax.set_ylabel("Pearson r", fontsize=7)
    ax.set_ylim(-0.25, 1.0)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)


def draw_ecdf(ax: plt.Axes, corr_arrays: dict[str, np.ndarray], cohort: str, letter: str) -> None:
    thresholds = np.linspace(-0.2, 1.0, 251)
    for method in METHOD_ORDER:
        if method not in corr_arrays:
            continue
        arr = corr_arrays[method]
        arr = arr[np.isfinite(arr)]
        y = np.array([(arr >= t).mean() for t in thresholds])
        ax.plot(thresholds, y, color=METHOD_COLORS[method], lw=1.35, label=method)
    ax.set_title(f"{letter}  Fraction of genes above threshold", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("Pearson r threshold", fontsize=7)
    ax.set_ylabel("Fraction of genes", fontsize=7)
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(color="#E4E4E4", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.5)


def gene_var_rank(matrix: np.ndarray, use_log1p: bool = True) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if use_log1p:
        arr = np.log1p(np.clip(arr, a_min=0, a_max=None))
    var = np.var(arr, axis=0)
    return np.argsort(var)[::-1]


def k_to_size(k: int, k_cap: int = 2000, s_min: float = 8, s_max: float = 62, power: float = 0.7) -> float:
    frac = min(max(k / float(k_cap), 0.0), 1.0)
    return float(s_min + (s_max - s_min) * (frac**power))


def dense_k_from_matrices(row: pd.Series, corr: np.ndarray) -> pd.DataFrame:
    result_dir = as_result_path(row)
    pred = np.load(result_dir / "pred_bag.npy", mmap_mode="r")
    true = np.load(result_dir / "true_bag.npy", mmap_mode="r")
    n_genes = min(pred.shape[1], true.shape[1], len(corr))
    pred_arr = np.asarray(pred[:, :n_genes], dtype=np.float32)
    true_arr = np.asarray(true[:, :n_genes], dtype=np.float32)
    pred_rank = gene_var_rank(pred_arr)
    true_rank = gene_var_rank(true_arr)
    max_k = min(2000, n_genes)
    rows = []
    for k in np.arange(50, max_k + 1, 50, dtype=int):
        top_pred = pred_rank[:k]
        top_true = set(true_rank[:k])
        overlap = sum(int(x in top_true) for x in top_pred)
        vals = corr[top_pred]
        rows.append(
            {
                "method": str(row["method"]),
                "K": int(k),
                "Recall@K": overlap / float(k),
                "MeanPearson_on_HVGpred": float(np.nanmean(vals)),
                "Overlap": int(overlap),
            }
        )
    return pd.DataFrame(rows)


def dense_k_from_saved_indices(row: pd.Series, corr: np.ndarray) -> pd.DataFrame | None:
    result_dir = as_result_path(row)
    idx = load_hvg_indices(result_dir)
    if idx is None:
        return None
    true_idx, pred_idx = idx
    max_k = min(1000, len(true_idx), len(pred_idx), len(corr))
    rows = []
    for k in [50, 200, 500, 1000]:
        if k > max_k:
            continue
        top_pred = pred_idx[:k]
        top_true = set(int(x) for x in true_idx[:k])
        overlap = sum(int(int(x) in top_true) for x in top_pred)
        vals = corr[top_pred[top_pred < len(corr)]]
        rows.append(
            {
                "method": str(row["method"]),
                "K": int(k),
                "Recall@K": overlap / float(k),
                "MeanPearson_on_HVGpred": float(np.nanmean(vals)),
                "Overlap": int(overlap),
            }
        )
    return pd.DataFrame(rows)


def draw_hvg(ax: plt.Axes, sub: pd.DataFrame, corr_arrays: dict[str, np.ndarray], cohort: str, letter: str) -> pd.DataFrame:
    frames = []
    for _, row in sub.iterrows():
        method = str(row["method"])
        if method not in corr_arrays:
            continue
        frames.append(dense_k_from_matrices(row, corr_arrays[method]))
    hvg = pd.concat(frames, ignore_index=True)
    highlight = [50, 200, 500, 1000, 2000]
    for method in METHOD_ORDER:
        d = hvg[hvg["method"].eq(method)].sort_values("K")
        if d.empty:
            continue
        color = METHOD_COLORS[method]
        ax.plot(d["Recall@K"], d["MeanPearson_on_HVGpred"], color=color, lw=1.45, alpha=0.9)
        sizes = [k_to_size(int(k)) for k in d["K"]]
        ax.scatter(d["Recall@K"], d["MeanPearson_on_HVGpred"], s=sizes, color=color, alpha=0.28, edgecolor="none")
        dh = d[d["K"].isin(highlight)]
        ax.scatter(
            dh["Recall@K"],
            dh["MeanPearson_on_HVGpred"],
            s=[k_to_size(int(k)) * 1.08 for k in dh["K"]],
            color=color,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
    ax.set_title(f"{letter}  HVG recovery vs prediction quality", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("HVG overlap (Recall@K)", fontsize=7)
    ax.set_ylabel("Mean Pearson on HVG_pred(topK)", fontsize=7)
    ax.grid(color="#E4E4E4", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    method_handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.8, label=m) for m in METHOD_ORDER]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="none", markerfacecolor="#AAAAAA", markeredgecolor="black", markersize=math.sqrt(k_to_size(k)), label=f"K={k}")
        for k in [50, 500, 2000]
    ]
    ax.legend(handles=method_handles + size_handles, frameon=False, fontsize=5.6, loc="lower right")
    return hvg


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def load_h5ad_spatial(slide: str) -> dict[str, object]:
    with h5py.File(ROOT / "data" / "spatial_data" / f"{slide}.h5ad", "r") as handle:
        obs = decode(handle["obs/_index"][:])
        image = handle["uns/spatial/ST/images/downscaled_fullres"][:]
        scale = float(handle["uns/spatial/ST/scalefactors/tissue_downscaled_fullres_scalef"][()])
        coords = handle["obsm/spatial"][:].astype(float) * scale
    return {"obs": obs, "image": image, "coords": coords}


def parse_yaml_genes(path: Path) -> list[str]:
    genes = []
    in_genes = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip() == "gene_names:":
            in_genes = True
            continue
        if in_genes:
            if line.startswith("- "):
                genes.append(line[2:].strip())
            elif line and not line.startswith(" "):
                break
    return genes


def scellst_gene_list(cohort: str, train: str, test: str, n_genes: int) -> list[str] | None:
    root = ROOT / "results" / f"{cohort.lower()}_hest" / "scellst" / "single"
    candidates = [
        root / f"{train}_to_{test}" / "logs" / "version_0" / "hparams.yaml",
        root / f"{train}_to_{test}" / "logs" / "version_1" / "hparams.yaml",
    ]
    candidates.extend(root.glob(f"{train}_to_*{test}*/logs/version_*/hparams.yaml"))
    for path in candidates:
        if path.exists():
            genes = parse_yaml_genes(path)
            if len(genes) == n_genes:
                return genes
    return None


def method_gene_list(method: str, cohort: str, train: str, test: str, n_genes: int) -> list[str]:
    if method == "Morpho-FM":
        genes = morphofm_gene_list(cohort, n_genes)
        if genes is not None:
            return genes
    if method == "sCellST":
        genes = scellst_gene_list(cohort, train, test, n_genes)
        if genes is not None:
            return genes
    genes = common_genes()
    return genes[:n_genes]


def load_spot_ids(result_dir: Path, n: int, fallback_obs: list[str]) -> list[str]:
    txt = result_dir / "spot_ids.txt"
    npy = result_dir / "pred_spot_ids.npy"
    if txt.exists():
        return txt.read_text(encoding="utf-8").splitlines()[:n]
    if npy.exists():
        values = np.load(npy, allow_pickle=True)
        return decode(values)[:n]
    return fallback_obs[:n]


def match_spot_id(spot: str, obs_to_idx: dict[str, int], test_slide: str) -> str | None:
    if spot in obs_to_idx:
        return spot
    slide_suffix = f"_{test_slide}"
    if spot.endswith(slide_suffix):
        base = spot[: -len(slide_suffix)]
        if base in obs_to_idx:
            return base
    base = re.sub(r"_INT\d+$", "", spot)
    if base in obs_to_idx:
        return base
    return None


def load_gene_vector(row: pd.Series, gene: str, spatial: dict[str, object], field: str = "pred") -> dict[str, object] | None:
    result_dir = as_result_path(row)
    matrix = np.load(result_dir / ("pred_bag.npy" if field == "pred" else "true_bag.npy"), mmap_mode="r")
    genes = method_gene_list(str(row["method"]), str(row["cohort"]), str(row["train_slide"]), str(row["test_slide"]), matrix.shape[1])
    if gene not in genes:
        return None
    gi = genes.index(gene)
    values = np.asarray(matrix[:, gi], dtype=float)
    spot_ids = load_spot_ids(result_dir, len(values), spatial["obs"])
    obs_to_idx = {spot: i for i, spot in enumerate(spatial["obs"])}
    matched: list[tuple[int, str]] = []
    test_slide = str(row["test_slide"])
    for i, spot in enumerate(spot_ids):
        obs_spot = match_spot_id(str(spot), obs_to_idx, test_slide)
        if obs_spot is not None:
            matched.append((i, obs_spot))
    if not matched:
        return None
    keep = [i for i, _ in matched]
    matched_spots = [spot for _, spot in matched]
    coord_idx = [obs_to_idx[spot] for spot in matched_spots]
    return {"values": values[keep], "coords": spatial["coords"][coord_idx], "spot_ids": matched_spots}


def robust_scale(values: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(finite, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def draw_spatial_panel(
    ax: plt.Axes,
    image: np.ndarray,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    spot_size: float = 4.2,
    spot_alpha: float = 0.88,
) -> mpl.cm.ScalarMappable:
    lo, hi = robust_scale(values)
    ax.imshow(image)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=spot_size,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        linewidths=0,
        alpha=spot_alpha,
        rasterized=True,
    )
    ax.set_title(title, fontsize=6.6, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def draw_main_spatial(ax: plt.Axes, df: pd.DataFrame, cohort: str, letter: str) -> None:
    rep = REPRESENTATIVE[cohort]
    sub = split_rows(df, cohort, rep["train"], rep["test"])
    spatial = load_h5ad_spatial(rep["test"])
    grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(), wspace=0.08)
    ax.remove()
    truth_row = sub[sub["method"].astype(str).eq("Morpho-FM")].iloc[0]
    truth = load_gene_vector(truth_row, rep["gene"], spatial, "true")
    assert truth is not None
    morpho_row = sub[sub["method"].astype(str).eq("Morpho-FM")].iloc[0]
    morpho_pred = load_gene_vector(morpho_row, rep["gene"], spatial, "pred")
    morpho_true = load_gene_vector(morpho_row, rep["gene"], spatial, "true")
    assert morpho_pred is not None and morpho_true is not None
    r = safe_pearson(morpho_pred["values"], morpho_true["values"])
    error = np.abs(morpho_pred["values"] - morpho_true["values"])
    panels = [
        ("Measured", truth["values"], truth["coords"], float("nan"), "viridis"),
        ("Morpho-FM", morpho_pred["values"], morpho_pred["coords"], r, "viridis"),
        ("Absolute error", error, morpho_pred["coords"], float("nan"), "magma"),
    ]
    fig = plt.gcf()
    for i, (label, values, coords, r, cmap) in enumerate(panels):
        child = fig.add_subplot(grid[0, i])
        title = label if not np.isfinite(r) else f"{label} (r={r:.3f})"
        if i == 0:
            title = f"{letter}  {rep['gene']} {title}"
        draw_spatial_panel(child, spatial["image"], coords, values, title, cmap=cmap)


def draw_full_spatial_2x4(
    fig: plt.Figure,
    sub_spec,
    df: pd.DataFrame,
    cohort: str,
    title_prefix: str,
    spot_size: float = 4.2,
    spot_alpha: float = 0.88,
) -> None:
    rep = REPRESENTATIVE[cohort]
    sub = split_rows(df, cohort, rep["train"], rep["test"])
    spatial = load_h5ad_spatial(rep["test"])
    truth_row = sub[sub["method"].astype(str).eq("Morpho-FM")].iloc[0]
    truth = load_gene_vector(truth_row, rep["gene"], spatial, "true")
    assert truth is not None
    # 2 x 5 layout: one measured panel plus every method in METHOD_ORDER, ranked
    # high to low. The previous 2 x 4 grid held seven panels and carried a
    # hard-coded six-method list, which silently dropped any method added later.
    _n_panels = 1 + len(METHOD_ORDER)
    if _n_panels > 10:
        raise ValueError(
            f"draw_full_spatial_2x4 is a 2x5 grid and holds 10 panels; "
            f"got 1 measured + {len(METHOD_ORDER)} methods = {_n_panels}."
        )
    grid = GridSpecFromSubplotSpec(2, 5, subplot_spec=sub_spec, wspace=0.14, hspace=0.24)
    panels = [("True", truth["values"], truth["coords"], float("nan"))]
    for method in METHOD_ORDER:
        row = sub[sub["method"].astype(str).eq(method)].iloc[0]
        pred = load_gene_vector(row, rep["gene"], spatial, "pred")
        true_for_r = load_gene_vector(row, rep["gene"], spatial, "true")
        if pred is None or true_for_r is None:
            continue
        r = safe_pearson(pred["values"], true_for_r["values"])
        panels.append((method, pred["values"], pred["coords"], r))
    positions = [(r, c) for r in range(2) for c in range(5)]
    for (row_i, col_i), (label, values, coords, r) in zip(positions, panels):
        ax = fig.add_subplot(grid[row_i, col_i])
        panel_title = label if not np.isfinite(r) else f"{label} (r={r:.3f})"
        draw_spatial_panel(ax, spatial["image"], coords, values, panel_title, spot_size=spot_size, spot_alpha=spot_alpha)
        if label == "True":
            ax.text(0.0, 1.11, title_prefix, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold")


def build_main_figure(df: pd.DataFrame, cohort: str, outdirs: list[Path]) -> None:
    slides = PROSTATE_SLIDES if cohort == "Prostate" else KIDNEY_SLIDES
    _, corr_arrays = rep_corr_payload(df, cohort)
    sub, _ = rep_corr_payload(df, cohort)
    fig = plt.figure(figsize=(185 / 25.4, 164 / 25.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.10], width_ratios=[1.02, 1.0, 1.05], wspace=0.54, hspace=0.64)
    draw_protocol_heatmap(fig.add_subplot(gs[0, 0]), df, cohort, slides, "a")
    draw_direction_summary(fig.add_subplot(gs[0, 1:]), df, cohort, "b")
    draw_violin(fig.add_subplot(gs[1, 0]), corr_arrays, cohort, "c")
    draw_ecdf(fig.add_subplot(gs[1, 1]), corr_arrays, cohort, "d")
    hvg = draw_hvg(fig.add_subplot(gs[1, 2]), sub, corr_arrays, cohort, "e")
    draw_main_spatial(fig.add_subplot(gs[2, :]), df, cohort, "f")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.055)
    name = f"main_{cohort.lower()}_rotating_single_nature"
    save_fig(fig, outdirs, name)
    plt.close(fig)
    for outdir in outdirs:
        hvg.to_csv(outdir / f"{name}_hvg_denseK.csv", index=False)


def build_prostate_supplement(df: pd.DataFrame, outdirs: list[Path]) -> None:
    cohort = "Prostate"
    splits = (
        df[(df["cohort"].astype(str).eq(cohort)) & (df["method"].astype(str).eq("Morpho-FM"))][["train_slide", "test_slide"]]
        .drop_duplicates()
        .sort_values(["train_slide", "test_slide"])
        .itertuples(index=False, name=None)
    )
    splits = list(splits)

    fig_v, axes_v = plt.subplots(3, 4, figsize=(190 / 25.4, 120 / 25.4), sharey=True)
    fig_e, axes_e = plt.subplots(3, 4, figsize=(190 / 25.4, 120 / 25.4), sharey=True)
    hvg_rows = []
    for ax_v, ax_e, (train, test) in zip(axes_v.ravel(), axes_e.ravel(), splits):
        sub = split_rows(df, cohort, train, test)
        arrays = {str(row["method"]): load_corr_array(as_result_path(row)) for _, row in sub.iterrows()}
        data = []
        positions = []
        colors = []
        labels = []
        for i, method in enumerate(METHOD_ORDER):
            arr = arrays.get(method)
            if arr is None:
                continue
            arr = arr[np.isfinite(arr)]
            data.append(arr)
            positions.append(i)
            colors.append(METHOD_COLORS[method])
            labels.append(method)
        parts = ax_v.violinplot(data, positions=positions, widths=0.72, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.20)
        for i, arr in zip(positions, data):
            ax_v.plot([i - 0.18, i + 0.18], [np.nanmedian(arr), np.nanmedian(arr)], color=TEXT, lw=0.6)
        ax_v.set_title(f"{train}->{test}", fontsize=6.8)
        ax_v.set_xticks(range(len(METHOD_ORDER)))
        ax_v.set_xticklabels([])
        ax_v.set_ylim(-0.25, 1.0)
        ax_v.grid(axis="y", color="#E7E7E7", lw=0.4)
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)

        thresholds = np.linspace(-0.2, 1.0, 251)
        for method in METHOD_ORDER:
            arr = arrays.get(method)
            if arr is None:
                continue
            arr = arr[np.isfinite(arr)]
            ax_e.plot(thresholds, [(arr >= t).mean() for t in thresholds], color=METHOD_COLORS[method], lw=0.95)
        ax_e.set_title(f"{train}->{test}", fontsize=6.8)
        ax_e.set_xlim(-0.2, 1.0)
        ax_e.set_ylim(0, 1.0)
        ax_e.grid(color="#E7E7E7", lw=0.4)
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)

        for _, row in sub.iterrows():
            arr = arrays[str(row["method"])]
            hvg = dense_k_from_saved_indices(row, arr)
            if hvg is not None:
                hvg["train_slide"] = train
                hvg["test_slide"] = test
                hvg_rows.append(hvg)

    handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=2, label=m) for m in METHOD_ORDER]
    fig_v.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7)
    fig_v.suptitle("Prostate per-gene Pearson distributions across rotating directions", fontsize=9, y=0.995)
    fig_v.subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.09, wspace=0.2, hspace=0.35)
    save_fig(fig_v, outdirs, "supp_prostate_all_directions_violin")
    plt.close(fig_v)

    fig_e.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7)
    fig_e.suptitle("Prostate threshold curves across rotating directions", fontsize=9, y=0.995)
    fig_e.subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.09, wspace=0.2, hspace=0.35)
    save_fig(fig_e, outdirs, "supp_prostate_all_directions_ecdf")
    plt.close(fig_e)

    hvg_all = pd.concat(hvg_rows, ignore_index=True) if hvg_rows else pd.DataFrame()
    for outdir in outdirs:
        hvg_all.to_csv(outdir / "supp_prostate_all_directions_hvg_denseK.csv", index=False)

    fig_h, axes_h = plt.subplots(3, 4, figsize=(190 / 25.4, 122 / 25.4), sharex=True, sharey=True)
    for ax, (train, test) in zip(axes_h.ravel(), splits):
        sub = hvg_all[hvg_all["train_slide"].astype(str).eq(str(train)) & hvg_all["test_slide"].astype(str).eq(str(test))]
        for method in METHOD_ORDER:
            d = sub[sub["method"].eq(method)].sort_values("K")
            if d.empty:
                continue
            ax.plot(d["Recall@K"], d["MeanPearson_on_HVGpred"], color=METHOD_COLORS[method], lw=0.95)
            ax.scatter(d["Recall@K"], d["MeanPearson_on_HVGpred"], color=METHOD_COLORS[method], s=8, alpha=0.65, linewidths=0)
        ax.set_title(f"{train}->{test}", fontsize=6.8)
        ax.grid(color="#E7E7E7", lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig_h.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7)
    fig_h.suptitle("Prostate HVG dense-K trajectories across rotating directions", fontsize=9, y=0.995)
    fig_h.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.09, wspace=0.2, hspace=0.35)
    save_fig(fig_h, outdirs, "supp_prostate_all_directions_hvg_denseK")
    plt.close(fig_h)

    gene = REPRESENTATIVE[cohort]["gene"]
    genes = common_genes()
    gi = genes.index(gene)
    mat = pd.DataFrame(np.nan, index=PROSTATE_SLIDES, columns=PROSTATE_SLIDES)
    for _, row in df[(df["cohort"].astype(str).eq(cohort)) & (df["method"].astype(str).eq("Morpho-FM"))].iterrows():
        corr = load_corr_array(as_result_path(row))
        if gi < len(corr):
            mat.loc[str(row["train_slide"]), str(row["test_slide"])] = corr[gi]
    fig, ax = plt.subplots(figsize=(74 / 25.4, 66 / 25.4))
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)), vmin=0.20, vmax=0.60, cmap="viridis")
    ax.set_xticks(range(len(PROSTATE_SLIDES)))
    ax.set_yticks(range(len(PROSTATE_SLIDES)))
    ax.set_xticklabels(PROSTATE_SLIDES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(PROSTATE_SLIDES, fontsize=7)
    ax.set_title(f"Prostate {gene} Morpho-FM spot-level r", fontsize=8.5, loc="left", fontweight="bold")
    ax.set_xlabel("Test slide", fontsize=7)
    ax.set_ylabel("Training slide", fontsize=7)
    for i in range(len(PROSTATE_SLIDES)):
        for j in range(len(PROSTATE_SLIDES)):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.4 else TEXT, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    save_fig(fig, outdirs, "supp_prostate_KLK3_morphofm_heatmap")
    plt.close(fig)

    fig = plt.figure(figsize=(176 / 25.4, 72 / 25.4))
    gs = fig.add_gridspec(1, 1)
    draw_full_spatial_2x4(fig, gs[0, 0], df, cohort, "Prostate representative KLK3 spatial reconstruction")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.04)
    save_fig(fig, outdirs, "supp_prostate_KLK3_true_plus_methods_representative")
    plt.close(fig)


def build_kidney_supplement(df: pd.DataFrame, outdirs: list[Path]) -> None:
    cohort = "Kidney"
    sub_all = df[df["cohort"].astype(str).eq(cohort)].copy()
    handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=2, label=m) for m in METHOD_ORDER]

    direction_grid: list[tuple[int, int, str, str]] = []
    for col, train in enumerate(KIDNEY_SLIDES):
        tests = [slide for slide in KIDNEY_SLIDES if slide != train]
        for row, test in enumerate(tests):
            direction_grid.append((row, col, train, test))

    fig_v, axes_v = plt.subplots(7, 8, figsize=(320 / 25.4, 230 / 25.4), sharey=True)
    fig_e, axes_e = plt.subplots(7, 8, figsize=(320 / 25.4, 230 / 25.4), sharey=True)
    hvg_rows = []
    thresholds = np.linspace(-0.2, 1.0, 251)
    for row_i, col_i, train, test in direction_grid:
        ax_v = axes_v[row_i, col_i]
        ax_e = axes_e[row_i, col_i]
        direction_rows = sub_all[
            sub_all["train_slide"].astype(str).eq(train)
            & sub_all["test_slide"].astype(str).eq(test)
        ]
        arrays_by_method: dict[str, list[np.ndarray]] = {method: [] for method in METHOD_ORDER}
        for _, row in direction_rows.iterrows():
            method = str(row["method"])
            corr = load_corr_array(as_result_path(row))
            arrays_by_method[method].append(corr[np.isfinite(corr)])
            hvg = dense_k_from_saved_indices(row, corr)
            if hvg is not None:
                hvg["train_slide"] = str(row["train_slide"])
                hvg["test_slide"] = test
                hvg_rows.append(hvg)

        data = []
        positions = []
        colors = []
        for i, method in enumerate(METHOD_ORDER):
            chunks = arrays_by_method.get(method, [])
            if not chunks:
                continue
            arr = np.concatenate(chunks)
            data.append(arr)
            positions.append(i)
            colors.append(METHOD_COLORS[method])
            ax_e.plot(thresholds, [(arr >= t).mean() for t in thresholds], color=METHOD_COLORS[method], lw=0.95)

        if data:
            parts = ax_v.violinplot(data, positions=positions, widths=0.72, showextrema=False)
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.20)
            for i, arr in zip(positions, data):
                ax_v.plot([i - 0.18, i + 0.18], [np.nanmedian(arr), np.nanmedian(arr)], color=TEXT, lw=0.55)
        ax_v.set_title(f"{train}->{test}", fontsize=4.9, pad=1.5)
        ax_v.set_xticks(range(len(METHOD_ORDER)))
        ax_v.set_xticklabels([])
        ax_v.set_ylim(-0.25, 1.0)
        ax_v.grid(axis="y", color="#E7E7E7", lw=0.4)
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)
        ax_v.tick_params(labelsize=4.8, length=1.5)
        if col_i != 0:
            ax_v.tick_params(labelleft=False)

        ax_e.set_title(f"{train}->{test}", fontsize=4.9, pad=1.5)
        ax_e.set_xlim(-0.2, 1.0)
        ax_e.set_ylim(0, 1.0)
        ax_e.grid(color="#E7E7E7", lw=0.4)
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)
        ax_e.tick_params(labelsize=4.8, length=1.5)
        if col_i != 0:
            ax_e.tick_params(labelleft=False)

    fig_v.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.006), fontsize=7)
    fig_v.suptitle("Kidney per-gene Pearson distributions across all 56 train-test directions", fontsize=9, y=0.995)
    fig_v.subplots_adjust(left=0.045, right=0.995, top=0.955, bottom=0.08, wspace=0.13, hspace=0.42)
    save_fig(fig_v, outdirs, "supp_kidney_by_test_slide_violin")
    plt.close(fig_v)

    fig_e.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.006), fontsize=7)
    fig_e.suptitle("Kidney threshold curves across all 56 train-test directions", fontsize=9, y=0.995)
    fig_e.subplots_adjust(left=0.045, right=0.995, top=0.955, bottom=0.08, wspace=0.13, hspace=0.42)
    save_fig(fig_e, outdirs, "supp_kidney_by_test_slide_ecdf")
    plt.close(fig_e)

    hvg_all = pd.concat(hvg_rows, ignore_index=True) if hvg_rows else pd.DataFrame()
    for outdir in outdirs:
        hvg_all.to_csv(outdir / "supp_kidney_all_directions_hvg_denseK.csv", index=False)

    if not hvg_all.empty:
        fig_h, axes_h = plt.subplots(7, 8, figsize=(320 / 25.4, 230 / 25.4), sharex=True, sharey=True)
        for row_i, col_i, train, test in direction_grid:
            ax = axes_h[row_i, col_i]
            direction_hvg = hvg_all[
                hvg_all["train_slide"].astype(str).eq(train)
                & hvg_all["test_slide"].astype(str).eq(test)
            ]
            for method in METHOD_ORDER:
                d = (
                    direction_hvg[direction_hvg["method"].eq(method)]
                    .groupby("K", as_index=False)
                    .agg({"Recall@K": "mean", "MeanPearson_on_HVGpred": "mean"})
                    .sort_values("K")
                )
                if d.empty:
                    continue
                ax.plot(d["Recall@K"], d["MeanPearson_on_HVGpred"], color=METHOD_COLORS[method], lw=0.95)
                ax.scatter(d["Recall@K"], d["MeanPearson_on_HVGpred"], color=METHOD_COLORS[method], s=7, alpha=0.7, linewidths=0)
            ax.set_title(f"{train}->{test}", fontsize=4.9, pad=1.5)
            ax.grid(color="#E7E7E7", lw=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=4.8, length=1.5)
            if col_i != 0:
                ax.tick_params(labelleft=False)
        fig_h.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.006), fontsize=7)
        fig_h.suptitle("Kidney HVG dense-K trajectories across all 56 train-test directions", fontsize=9, y=0.995)
        fig_h.subplots_adjust(left=0.045, right=0.995, top=0.955, bottom=0.08, wspace=0.13, hspace=0.42)
        save_fig(fig_h, outdirs, "supp_kidney_by_test_slide_hvg_denseK")
        plt.close(fig_h)

    gene = REPRESENTATIVE[cohort]["gene"]
    genes = common_genes()
    gi = genes.index(gene)
    mat = pd.DataFrame(np.nan, index=KIDNEY_SLIDES, columns=KIDNEY_SLIDES)
    for _, row in sub_all[sub_all["method"].astype(str).eq("Morpho-FM")].iterrows():
        corr = load_corr_array(as_result_path(row))
        if gi < len(corr):
            mat.loc[str(row["train_slide"]), str(row["test_slide"])] = corr[gi]
    fig, ax = plt.subplots(figsize=(92 / 25.4, 78 / 25.4))
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)), vmin=0.50, vmax=0.95, cmap="viridis")
    ax.set_xticks(range(len(KIDNEY_SLIDES)))
    ax.set_yticks(range(len(KIDNEY_SLIDES)))
    ax.set_xticklabels(KIDNEY_SLIDES, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(KIDNEY_SLIDES, fontsize=6.5)
    ax.set_title(f"Kidney {gene} Morpho-FM spot-level r", fontsize=8.5, loc="left", fontweight="bold")
    ax.set_xlabel("Test slide", fontsize=7)
    ax.set_ylabel("Training slide", fontsize=7)
    for i in range(len(KIDNEY_SLIDES)):
        for j in range(len(KIDNEY_SLIDES)):
            val = mat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.73 else TEXT, fontsize=5.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    save_fig(fig, outdirs, "supp_kidney_B2M_morphofm_heatmap")
    plt.close(fig)

    fig = plt.figure(figsize=(176 / 25.4, 72 / 25.4))
    gs = fig.add_gridspec(1, 1)
    draw_full_spatial_2x4(
        fig,
        gs[0, 0],
        df,
        cohort,
        "Kidney representative B2M spatial reconstruction",
        spot_size=1.9,
        spot_alpha=0.95,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.04)
    save_fig(fig, outdirs, "supp_kidney_B2M_true_plus_methods_representative")
    plt.close(fig)


def write_summary_tables(df: pd.DataFrame, outdirs: list[Path]) -> None:
    method_summary = (
        df.groupby(["cohort", "method"], observed=True)
        .agg(
            n=("Mean_Pearson", "size"),
            mean_pearson=("Mean_Pearson", "mean"),
            median_pearson=("Median_Pearson", "mean"),
            hvg1000_mean=("HVG1000_Mean_Pearson", "mean"),
            mae=("MAE", "mean"),
            rmse=("RMSE", "mean"),
        )
        .reset_index()
    )
    split_summary = df[["cohort", "method", "train_slide", "test_slide", "Mean_Pearson", "Median_Pearson", "HVG1000_Mean_Pearson", "MAE", "RMSE", "result_dir"]].copy()
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        method_summary.to_csv(outdir / "nature_rotating_single_method_summary.csv", index=False)
        split_summary.to_csv(outdir / "nature_rotating_single_split_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    setup_style()
    df = load_metrics(args.metrics)
    outdirs = [args.repo_out, args.paper_out]
    write_summary_tables(df, outdirs)
    build_main_figure(df, "Prostate", outdirs)
    build_main_figure(df, "Kidney", outdirs)
    if args.supplement_prostate:
        build_prostate_supplement(df, outdirs)
    build_kidney_supplement(df, outdirs)
    print(f"saved {args.repo_out}")
    print(f"saved {args.paper_out}")


if __name__ == "__main__":
    main()
