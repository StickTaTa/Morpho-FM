"""Build Nature-style prostate Protocol-A multi-slide benchmark figures.

The layout mirrors the rotating single-slide figure package while using the
four held-out multi-slide folds:
two training slides, one validation slide and one held-out test slide.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_nature_rotating_benchmark_figures as single_fig  # noqa: E402


REMOTE_ROOT = "/share/home/nxy1zll2hjj/hjj/Morpho-VC"
DEFAULT_MANIFEST = ROOT / "results" / "benchmark_audit" / "prostate_protocol_a_benchmark_manifest.csv"
DEFAULT_MORPHO = ROOT / "results" / "prostate_hest" / "morpho" / "protocol_a_rotating" / "prostate_protocol_a_morpho_complete_metrics.csv"
DEFAULT_REPO_OUT = ROOT / "results" / "paper_figures" / "rotating_multi_benchmark_nature"
DEFAULT_PAPER_OUT = (
    Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
    / "Figure"
    / "Figure_2"
    / "benchmark_rebuild"
    / "rotating_multi_benchmark_nature"
)

METHOD_ORDER = single_fig.METHOD_ORDER
METHOD_COLORS = single_fig.METHOD_COLORS
TEXT = single_fig.TEXT
PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
REPRESENTATIVE_TAG = "test_INT28_val_INT25"
REPRESENTATIVE_TRAIN = "INT26,INT27"
REPRESENTATIVE_VAL = "INT25"
REPRESENTATIVE_TEST = "INT28"
REPRESENTATIVE_GENE = "KLK3"


FOLDS = [
    {"tag": "test_INT25_val_INT26", "train": "INT27,INT28", "val": "INT26", "test": "INT25"},
    {"tag": "test_INT26_val_INT27", "train": "INT25,INT28", "val": "INT27", "test": "INT26"},
    {"tag": "test_INT27_val_INT28", "train": "INT25,INT26", "val": "INT28", "test": "INT27"},
    {"tag": "test_INT28_val_INT25", "train": "INT26,INT27", "val": "INT25", "test": "INT28"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--morpho-metrics", type=Path, default=DEFAULT_MORPHO)
    parser.add_argument("--repo-out", type=Path, default=DEFAULT_REPO_OUT)
    parser.add_argument("--paper-out", type=Path, default=DEFAULT_PAPER_OUT)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def localize_result_dir(value: object) -> str:
    text = str(value)
    if text.startswith(REMOTE_ROOT):
        rel = text[len(REMOTE_ROOT) :].lstrip("/")
        return str(ROOT / Path(*rel.split("/")))
    return text


def safe_pearson_vec(true: np.ndarray, pred: np.ndarray, chunk: int = 1024) -> np.ndarray:
    n_spots = min(true.shape[0], pred.shape[0])
    n_genes = min(true.shape[1], pred.shape[1])
    out = np.full(n_genes, np.nan, dtype=np.float32)
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        t = np.asarray(true[:n_spots, start:end], dtype=np.float64)
        p = np.asarray(pred[:n_spots, start:end], dtype=np.float64)
        t = t - t.mean(axis=0, keepdims=True)
        p = p - p.mean(axis=0, keepdims=True)
        num = (t * p).sum(axis=0)
        den = np.sqrt((t * t).sum(axis=0) * (p * p).sum(axis=0))
        valid = den > 0
        values = np.full(end - start, np.nan, dtype=np.float32)
        values[valid] = (num[valid] / den[valid]).astype(np.float32)
        out[start:end] = values
    return out


def column_variance(matrix: np.ndarray, chunk: int = 1024) -> np.ndarray:
    n_genes = matrix.shape[1]
    out = np.full(n_genes, np.nan, dtype=np.float32)
    for start in range(0, n_genes, chunk):
        end = min(start + chunk, n_genes)
        arr = np.asarray(matrix[:, start:end], dtype=np.float32)
        out[start:end] = np.var(arr, axis=0)
    return out


def ensure_unified_payload(result_dir: Path, force: bool = False) -> dict:
    metrics_path = result_dir / "unified_metrics.json"
    corr_path = result_dir / "unified_gene_corr_all.npy"
    hvg_true_path = result_dir / "unified_hvg_idx_true.npy"
    hvg_pred_path = result_dir / "unified_hvg_idx_pred.npy"
    if metrics_path.exists() and corr_path.exists() and hvg_true_path.exists() and hvg_pred_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    pred = np.load(result_dir / "pred_bag.npy", mmap_mode="r")
    true = np.load(result_dir / "true_bag.npy", mmap_mode="r")
    n_spots = min(pred.shape[0], true.shape[0])
    n_genes = min(pred.shape[1], true.shape[1])
    pred_view = pred[:n_spots, :n_genes]
    true_view = true[:n_spots, :n_genes]

    corr_all = safe_pearson_vec(true_view, pred_view)
    var_true = column_variance(true_view)
    var_pred = column_variance(pred_view)
    top_hvg = min(1000, n_genes)
    hvg_idx_true = np.argsort(var_true)[::-1][:top_hvg]
    hvg_idx_pred = np.argsort(var_pred)[::-1][:top_hvg]
    overlap_idx = np.array(sorted(set(map(int, hvg_idx_true)) & set(map(int, hvg_idx_pred))), dtype=np.int64)
    corr_hvg = corr_all[hvg_idx_true]
    corr_overlap = corr_all[overlap_idx] if overlap_idx.size else np.array([], dtype=np.float32)
    valid = np.where(np.isfinite(corr_all))[0]
    best_idx = int(valid[np.nanargmax(corr_all[valid])]) if valid.size else None

    pred_arr = np.asarray(pred_view, dtype=np.float32)
    true_arr = np.asarray(true_view, dtype=np.float32)
    payload = {
        "n_spots": int(n_spots),
        "n_genes": int(n_genes),
        "MAE": float(np.mean(np.abs(pred_arr - true_arr))),
        "RMSE": float(np.sqrt(np.mean((pred_arr - true_arr) ** 2))),
        "Mean_Pearson": float(np.nanmean(corr_all)) if valid.size else float("nan"),
        "Median_Pearson": float(np.nanmedian(corr_all)) if valid.size else float("nan"),
        "HVG1000_Mean_Pearson": float(np.nanmean(corr_hvg)),
        "HVG1000_Median_Pearson": float(np.nanmedian(corr_hvg)),
        "HVG1000_Overlap": int(overlap_idx.size),
        "Best_Gene_Index": best_idx,
        "Best_Pearson": float(corr_all[best_idx]) if best_idx is not None else None,
    }

    np.save(corr_path, corr_all)
    np.save(result_dir / "unified_gene_corr_hvg.npy", corr_hvg)
    np.save(hvg_true_path, hvg_idx_true)
    np.save(hvg_pred_path, hvg_idx_pred)
    np.save(result_dir / "unified_corr_overlap_genes.npy", corr_overlap)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_baseline_rows(manifest_path: Path, force: bool) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest[manifest["status"].astype(str).eq("complete")].iterrows():
        result_dir = Path(str(row["result_dir"]))
        payload = ensure_unified_payload(result_dir, force=force)
        rows.append(
            {
                "cohort": "Prostate",
                "method": str(row["method"]),
                "train_slide": str(row["train_slides"]),
                "val_slide": str(row["val_slide"]),
                "test_slide": str(row["test_slide"]),
                "fold_tag": str(row["fold_tag"]),
                "protocol": "rotating_heldout_multi_slide",
                "result_dir": str(result_dir),
                **payload,
            }
        )
    return pd.DataFrame(rows)


def load_morpho_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"].astype(str).eq("complete")].copy()
    rows = []
    for _, row in df.iterrows():
        result_dir = Path(localize_result_dir(row["result_dir"]))
        rows.append(
            {
                "cohort": "Prostate",
                "method": "Morpho-FM",
                "train_slide": str(row["train_slides"]),
                "val_slide": str(row["val_slide"]),
                "test_slide": str(row["test_slide_id"]),
                "fold_tag": str(row["tag"]),
                "protocol": "rotating_heldout_multi_slide",
                "result_dir": str(result_dir),
                "n_spots": row.get("n_spots"),
                "n_genes": row.get("n_genes"),
                "MAE": row.get("MAE"),
                "RMSE": row.get("RMSE"),
                "Mean_Pearson": row.get("Mean_Pearson"),
                "Median_Pearson": row.get("Median_Pearson"),
                "HVG1000_Mean_Pearson": row.get("HVG1000_Mean_Pearson"),
                "HVG1000_Median_Pearson": row.get("HVG1000_Median_Pearson"),
                "HVG1000_Overlap": row.get("HVG1000_Overlap"),
                "Best_Gene": row.get("Best_Gene"),
                "Best_Pearson": row.get("Best_Pearson"),
            }
        )
    return pd.DataFrame(rows)


def load_metrics(args: argparse.Namespace) -> pd.DataFrame:
    baseline = load_baseline_rows(args.manifest, args.force_recompute)
    morpho = load_morpho_rows(args.morpho_metrics)
    df = pd.concat([morpho, baseline], ignore_index=True, sort=False)
    for col in [
        "n_spots",
        "n_genes",
        "MAE",
        "RMSE",
        "Mean_Pearson",
        "Median_Pearson",
        "HVG1000_Mean_Pearson",
        "HVG1000_Median_Pearson",
        "HVG1000_Overlap",
        "Best_Pearson",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    fold_order = {fold["tag"]: i for i, fold in enumerate(FOLDS)}
    df["fold_order"] = df["fold_tag"].map(fold_order)
    return df.sort_values(["fold_order", "method"]).reset_index(drop=True)


def setup_representative() -> None:
    single_fig.REPRESENTATIVE["Prostate"] = {
        "train": REPRESENTATIVE_TRAIN,
        "test": REPRESENTATIVE_TEST,
        "gene": REPRESENTATIVE_GENE,
        "best_baseline": "iStar",
    }


def save_tables(df: pd.DataFrame, outdirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_summary = (
        df.groupby(["method"], observed=True)
        .agg(
            n=("Mean_Pearson", "size"),
            mean_pearson=("Mean_Pearson", "mean"),
            median_pearson=("Median_Pearson", "mean"),
            hvg1000_mean=("HVG1000_Mean_Pearson", "mean"),
            hvg1000_median=("HVG1000_Median_Pearson", "mean"),
            mae=("MAE", "mean"),
            rmse=("RMSE", "mean"),
        )
        .reset_index()
    )
    split_summary = df[
        [
            "method",
            "fold_tag",
            "train_slide",
            "val_slide",
            "test_slide",
            "Mean_Pearson",
            "Median_Pearson",
            "HVG1000_Mean_Pearson",
            "HVG1000_Median_Pearson",
            "MAE",
            "RMSE",
            "result_dir",
        ]
    ].copy()
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "protocol_a_multi_all_metrics.csv", index=False)
        method_summary.to_csv(outdir / "nature_rotating_multi_method_summary.csv", index=False)
        split_summary.to_csv(outdir / "nature_rotating_multi_split_summary.csv", index=False)
    return method_summary, split_summary


def split_rows(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    return df[df["fold_tag"].astype(str).eq(tag)].sort_values("method").copy()


def corr_arrays_for_rows(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for _, row in rows.iterrows():
        arrays[str(row["method"])] = single_fig.load_corr_array(Path(str(row["result_dir"])))
    return arrays


def draw_test_slide_heatmap(ax: plt.Axes, df: pd.DataFrame, letter: str) -> None:
    mat = pd.DataFrame(np.nan, index=METHOD_ORDER, columns=PROSTATE_SLIDES)
    for _, row in df.iterrows():
        method = str(row["method"])
        test = str(row["test_slide"])
        if method in mat.index and test in mat.columns:
            mat.loc[method, test] = row["Mean_Pearson"]
    arr = mat.to_numpy(dtype=float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("multi_perf", ["#F4F4F4", "#B9D7D5", "#2B7C8F", "#E64B35"])
    im = ax.imshow(arr, vmin=0.0, vmax=max(0.38, float(np.nanmax(arr)) + 0.01), cmap=cmap)
    ax.set_xticks(range(len(PROSTATE_SLIDES)))
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(PROSTATE_SLIDES, rotation=40, ha="right", fontsize=6.5)
    ax.set_yticklabels(METHOD_ORDER, fontsize=6.5)
    ax.set_title(f"{letter}  Multi-slide test matrix", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("Held-out test slide", fontsize=7)
    ax.set_ylabel("Method", fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(PROSTATE_SLIDES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.7, color=("white" if value > 0.20 else TEXT))
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5, width=0.4)
    cbar.set_label("r", fontsize=6.0)


def draw_fold_summary(ax: plt.Axes, df: pd.DataFrame, letter: str) -> None:
    rng = np.random.default_rng(18)
    for i, method in enumerate(METHOD_ORDER):
        values = df[df["method"].astype(str).eq(method)]["Mean_Pearson"].dropna().to_numpy()
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
        ax.scatter(x, values, s=18, color=color, alpha=0.72, linewidths=0, zorder=3)
        ax.scatter([i], [values.mean()], s=34, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_title(f"{letter}  Across four held-out folds", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=33, ha="right", fontsize=6.5)
    ax.set_ylabel("Mean per-gene Pearson r", fontsize=7)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)
    n = df.groupby("method", observed=True).size()
    ax.text(
        0.0,
        -0.38,
        "folds: " + ", ".join(f"{m}={int(n.get(m, 0))}" for m in METHOD_ORDER),
        transform=ax.transAxes,
        fontsize=5.4,
        color="#666666",
        ha="left",
    )


def draw_violin(ax: plt.Axes, corr_arrays: dict[str, np.ndarray], title: str, letter: str) -> None:
    data = []
    labels = []
    colors = []
    for method in METHOD_ORDER:
        arr = corr_arrays.get(method)
        if arr is None:
            continue
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
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=33, ha="right", fontsize=6.4)
    ax.set_ylabel("Pearson r", fontsize=7)
    ax.set_ylim(-0.25, 1.0)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5)


def draw_ecdf(ax: plt.Axes, corr_arrays: dict[str, np.ndarray], letter: str) -> None:
    thresholds = np.linspace(-0.2, 1.0, 251)
    for method in METHOD_ORDER:
        arr = corr_arrays.get(method)
        if arr is None:
            continue
        arr = arr[np.isfinite(arr)]
        ax.plot(thresholds, [(arr >= t).mean() for t in thresholds], color=METHOD_COLORS[method], lw=1.35, label=method)
    ax.set_title(f"{letter}  Fraction of genes above threshold", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("Pearson r threshold", fontsize=7)
    ax.set_ylabel("Fraction of genes", fontsize=7)
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(color="#E4E4E4", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.5)


def draw_main_spatial(fig: plt.Figure, sub_spec, df: pd.DataFrame, letter: str) -> None:
    sub = split_rows(df, REPRESENTATIVE_TAG)
    spatial = single_fig.load_h5ad_spatial(REPRESENTATIVE_TEST)
    grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=sub_spec, wspace=0.08)
    truth_row = sub[sub["method"].astype(str).eq("Morpho-FM")].iloc[0]
    truth = single_fig.load_gene_vector(truth_row, REPRESENTATIVE_GENE, spatial, "true")
    morpho_pred = single_fig.load_gene_vector(truth_row, REPRESENTATIVE_GENE, spatial, "pred")
    morpho_true = single_fig.load_gene_vector(truth_row, REPRESENTATIVE_GENE, spatial, "true")
    assert truth is not None and morpho_pred is not None and morpho_true is not None
    r = single_fig.safe_pearson(morpho_pred["values"], morpho_true["values"])
    error = np.abs(morpho_pred["values"] - morpho_true["values"])
    panels = [
        (f"{letter}  {REPRESENTATIVE_GENE} Measured", truth["values"], truth["coords"], float("nan"), "viridis"),
        ("Morpho-FM", morpho_pred["values"], morpho_pred["coords"], r, "viridis"),
        ("Absolute error", error, morpho_pred["coords"], float("nan"), "magma"),
    ]
    for i, (label, values, coords, corr, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        title = label if not np.isfinite(corr) else f"{label} (r={corr:.3f})"
        single_fig.draw_spatial_panel(ax, spatial["image"], coords, values, title, cmap=cmap)


def build_main_figure(df: pd.DataFrame, outdirs: list[Path]) -> pd.DataFrame:
    rows = split_rows(df, REPRESENTATIVE_TAG)
    corr_arrays = corr_arrays_for_rows(rows)
    fig = plt.figure(figsize=(185 / 25.4, 164 / 25.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.10], width_ratios=[1.02, 1.0, 1.05], wspace=0.62, hspace=0.64)
    draw_test_slide_heatmap(fig.add_subplot(gs[0, 0]), df, "a")
    draw_fold_summary(fig.add_subplot(gs[0, 1:]), df, "b")
    draw_violin(fig.add_subplot(gs[1, 0]), corr_arrays, f"Per-gene Pearson ({REPRESENTATIVE_TRAIN.replace(',', '+')}->{REPRESENTATIVE_TEST})", "c")
    draw_ecdf(fig.add_subplot(gs[1, 1]), corr_arrays, "d")
    ax_hvg = fig.add_subplot(gs[1, 2])
    hvg = single_fig.draw_hvg(ax_hvg, rows, corr_arrays, "Prostate", "e")
    legend = ax_hvg.get_legend()
    if legend is not None:
        legend.remove()
    method_handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.8, label=m) for m in METHOD_ORDER]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="none",
            markerfacecolor="#AAAAAA",
            markeredgecolor="black",
            markersize=math.sqrt(single_fig.k_to_size(k)),
            label=f"K={k}",
        )
        for k in [50, 500, 2000]
    ]
    ax_hvg.legend(handles=method_handles + size_handles, frameon=False, fontsize=5.4, loc="lower left")
    draw_main_spatial(fig, gs[2, :], df, "f")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.055)
    single_fig.save_fig(fig, outdirs, "main_prostate_rotating_multi_nature")
    plt.close(fig)
    for outdir in outdirs:
        hvg.to_csv(outdir / "main_prostate_rotating_multi_nature_hvg_denseK.csv", index=False)
    return hvg


def build_supplement(df: pd.DataFrame, outdirs: list[Path]) -> pd.DataFrame:
    handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=2, label=m) for m in METHOD_ORDER]
    fig_v, axes_v = plt.subplots(2, 2, figsize=(150 / 25.4, 96 / 25.4), sharey=True)
    fig_e, axes_e = plt.subplots(2, 2, figsize=(150 / 25.4, 96 / 25.4), sharey=True)
    fig_h, axes_h = plt.subplots(2, 2, figsize=(150 / 25.4, 96 / 25.4), sharex=True, sharey=True)
    hvg_rows = []
    thresholds = np.linspace(-0.2, 1.0, 251)
    for ax_v, ax_e, ax_h, fold in zip(axes_v.ravel(), axes_e.ravel(), axes_h.ravel(), FOLDS):
        rows = split_rows(df, fold["tag"])
        arrays = corr_arrays_for_rows(rows)
        title = f"train {fold['train'].replace(',', '+')}; val {fold['val']}; test {fold['test']}"
        data = []
        positions = []
        colors = []
        for i, method in enumerate(METHOD_ORDER):
            arr = arrays.get(method)
            if arr is None:
                continue
            arr = arr[np.isfinite(arr)]
            data.append(arr)
            positions.append(i)
            colors.append(METHOD_COLORS[method])
            ax_e.plot(thresholds, [(arr >= t).mean() for t in thresholds], color=METHOD_COLORS[method], lw=0.95)
        parts = ax_v.violinplot(data, positions=positions, widths=0.72, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.20)
        for i, arr in zip(positions, data):
            ax_v.plot([i - 0.18, i + 0.18], [np.nanmedian(arr), np.nanmedian(arr)], color=TEXT, lw=0.6)
        ax_v.set_title(title, fontsize=6.4)
        ax_v.set_xticks(range(len(METHOD_ORDER)))
        ax_v.set_xticklabels([])
        ax_v.set_ylim(-0.25, 1.0)
        ax_v.grid(axis="y", color="#E7E7E7", lw=0.4)
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)

        ax_e.set_title(title, fontsize=6.4)
        ax_e.set_xlim(-0.2, 1.0)
        ax_e.set_ylim(0, 1.0)
        ax_e.grid(color="#E7E7E7", lw=0.4)
        ax_e.spines["top"].set_visible(False)
        ax_e.spines["right"].set_visible(False)

        for _, row in rows.iterrows():
            method = str(row["method"])
            arr = arrays[method]
            hvg = single_fig.dense_k_from_saved_indices(row, arr)
            if hvg is None:
                hvg = single_fig.dense_k_from_matrices(row, arr)
            hvg["fold_tag"] = fold["tag"]
            hvg["train_slide"] = fold["train"]
            hvg["val_slide"] = fold["val"]
            hvg["test_slide"] = fold["test"]
            hvg_rows.append(hvg)
            d = hvg.sort_values("K")
            ax_h.plot(d["Recall@K"], d["MeanPearson_on_HVGpred"], color=METHOD_COLORS[method], lw=0.95)
            ax_h.scatter(
                d["Recall@K"],
                d["MeanPearson_on_HVGpred"],
                color=METHOD_COLORS[method],
                s=[single_fig.k_to_size(int(k), s_min=4, s_max=24) for k in d["K"]],
                alpha=0.62,
                linewidths=0,
            )
        ax_h.set_title(title, fontsize=6.4)
        ax_h.grid(color="#E7E7E7", lw=0.4)
        ax_h.spines["top"].set_visible(False)
        ax_h.spines["right"].set_visible(False)

    for fig, name, title in [
        (fig_v, "supp_prostate_multi_all_folds_violin", "Prostate Protocol-A per-gene Pearson distributions across all folds"),
        (fig_e, "supp_prostate_multi_all_folds_ecdf", "Prostate Protocol-A threshold curves across all folds"),
        (fig_h, "supp_prostate_multi_all_folds_hvg_denseK", "Prostate Protocol-A HVG dense-K trajectories across all folds"),
    ]:
        fig.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7)
        fig.suptitle(title, fontsize=9, y=0.995)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.115, wspace=0.2, hspace=0.35)
        single_fig.save_fig(fig, outdirs, name)
        plt.close(fig)

    hvg_all = pd.concat(hvg_rows, ignore_index=True)
    for outdir in outdirs:
        hvg_all.to_csv(outdir / "supp_prostate_multi_all_folds_hvg_denseK.csv", index=False)

    build_klk3_heatmap(df, outdirs)
    build_full_spatial(df, outdirs)
    return hvg_all


def build_klk3_heatmap(df: pd.DataFrame, outdirs: list[Path]) -> None:
    mat = pd.DataFrame(np.nan, index=["Morpho-FM"], columns=PROSTATE_SLIDES)
    for _, row in df[df["method"].astype(str).eq("Morpho-FM")].iterrows():
        corr = single_fig.load_corr_array(Path(str(row["result_dir"])))
        genes = single_fig.method_gene_list("Morpho-FM", "Prostate", str(row["train_slide"]), str(row["test_slide"]), len(corr))
        gi = genes.index(REPRESENTATIVE_GENE)
        if gi < len(corr):
            mat.loc["Morpho-FM", str(row["test_slide"])] = corr[gi]
    fig, ax = plt.subplots(figsize=(92 / 25.4, 34 / 25.4))
    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(arr, vmin=0.20, vmax=max(0.60, float(np.nanmax(arr)) + 0.01), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(PROSTATE_SLIDES)))
    ax.set_yticks([0])
    ax.set_xticklabels(PROSTATE_SLIDES, rotation=35, ha="right", fontsize=7)
    ax.set_yticklabels(["Morpho-FM"], fontsize=7)
    ax.set_title(f"Prostate Protocol-A {REPRESENTATIVE_GENE} Morpho-FM spot-level r", fontsize=8.5, loc="left", fontweight="bold")
    ax.set_xlabel("Held-out test slide", fontsize=7)
    for j, value in enumerate(arr[0]):
        ax.text(j, 0, f"{value:.2f}", ha="center", va="center", color=("white" if value > 0.4 else TEXT), fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    single_fig.save_fig(fig, outdirs, "supp_prostate_multi_KLK3_morphofm_heatmap")
    plt.close(fig)


def build_full_spatial(df: pd.DataFrame, outdirs: list[Path]) -> None:
    fig = plt.figure(figsize=(176 / 25.4, 72 / 25.4))
    gs = fig.add_gridspec(1, 1)
    single_fig.draw_full_spatial_2x4(
        fig,
        gs[0, 0],
        df,
        "Prostate",
        f"Prostate Protocol-A representative {REPRESENTATIVE_GENE} spatial reconstruction",
        spot_size=1.9,
        spot_alpha=0.95,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.04)
    single_fig.save_fig(fig, outdirs, "supp_prostate_multi_KLK3_true_plus_methods_representative")
    plt.close(fig)


def print_summary(method_summary: pd.DataFrame) -> None:
    show = method_summary.copy()
    for col in ["mean_pearson", "median_pearson", "hvg1000_mean", "hvg1000_median", "mae", "rmse"]:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(4)
    print(show.sort_values("mean_pearson", ascending=False).to_string(index=False))


def main() -> None:
    args = parse_args()
    setup_representative()
    single_fig.setup_style()
    outdirs = [args.repo_out, args.paper_out]
    df = load_metrics(args)
    method_summary, _ = save_tables(df, outdirs)
    build_main_figure(df, outdirs)
    build_supplement(df, outdirs)
    print_summary(method_summary)
    print(f"saved {args.repo_out}")
    print(f"saved {args.paper_out}")


if __name__ == "__main__":
    main()
