from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID_FACTOR = 16
DEFAULT_SLIDES = "INT25,INT26,INT27,INT28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate spot-level HIPT features with spot size factors. "
            "HIPT channels are aggregated inside the same disk-shaped spot mask used by Morpho-FM."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--slides", default=DEFAULT_SLIDES, help="Comma-separated slide IDs.")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def setup_root(root_arg: Path | None) -> Path:
    root = root_arg or Path(os.environ.get("MORPHO_VC_ROOT", "."))
    return root.expanduser().resolve()


def split_slides(value: str) -> list[str]:
    slides = [item.strip() for item in value.split(",") if item.strip()]
    if not slides:
        raise ValueError("--slides must contain at least one slide ID")
    return slides


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


def read_spot_totals(cnts_path: Path) -> pd.Series:
    parts: list[pd.Series] = []
    for chunk in pd.read_csv(cnts_path, sep="\t", index_col=0, chunksize=512):
        parts.append(chunk.sum(axis=1).astype(np.float32))
    totals = pd.concat(parts)
    totals.index = totals.index.astype(str)
    return totals


def iter_feature_channels(embs: dict):
    for group in ("cls", "sub"):
        for idx, arr in enumerate(embs[group]):
            yield group, idx, arr
    rgb = embs.get("rgb")
    if rgb is not None:
        for idx in range(rgb.shape[0]):
            yield "rgb", idx, rgb[idx]


def feature_names(embs: dict) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    for group, idx, _ in iter_feature_channels(embs):
        names.append((group, f"{group}_{idx:03d}"))
    return names


def pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean()
    num = x.T @ y
    den = np.sqrt((x * x).sum(axis=0) * np.sum(y * y))
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out.astype(np.float32)


def aggregate_slide_features(slide_dir: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], dict]:
    locs_df = pd.read_csv(slide_dir / "locs.tsv", sep="\t")
    spot_ids = locs_df["SpotID"].astype(str).to_numpy()
    locs_px = np.stack([locs_df["y"].to_numpy(), locs_df["x"].to_numpy()], axis=-1).astype(np.float32)
    totals = read_spot_totals(slide_dir / "cnts.tsv")
    spot_total = pd.Series(spot_ids).map(totals).to_numpy(dtype=np.float32)
    if np.isnan(spot_total).any():
        # Fall back to file order if SpotID labels are unavailable or formatted differently.
        ordered = totals.to_numpy(dtype=np.float32)
        if ordered.shape[0] != spot_ids.shape[0]:
            raise ValueError(f"Cannot align spot totals for {slide_dir}")
        spot_total = ordered

    radius_px = float((slide_dir / "radius.txt").read_text(encoding="utf-8").strip())
    radius_grid = radius_px / float(GRID_FACTOR)
    mask = get_disk_mask(radius_grid)
    offsets = np.argwhere(mask).astype(np.int64) - (np.array(mask.shape, dtype=np.int64) // 2)

    with (slide_dir / "embeddings-hist.pickle").open("rb") as f:
        embs = pickle.load(f)

    first_arr = embs["cls"][0]
    height, width = first_arr.shape
    locs_grid = np.round(locs_px / float(GRID_FACTOR)).astype(np.int64)
    min_offsets = offsets.min(axis=0)
    max_offsets = offsets.max(axis=0)
    valid = (
        (locs_grid[:, 0] + min_offsets[0] >= 0)
        & (locs_grid[:, 0] + max_offsets[0] < height)
        & (locs_grid[:, 1] + min_offsets[1] >= 0)
        & (locs_grid[:, 1] + max_offsets[1] < width)
        & np.isfinite(spot_total)
        & (spot_total > 0)
    )

    locs_valid = locs_grid[valid]
    total_valid = spot_total[valid]
    yy = locs_valid[:, [0]] + offsets[:, 0][None, :]
    xx = locs_valid[:, [1]] + offsets[:, 1][None, :]

    names = feature_names(embs)
    x = np.empty((locs_valid.shape[0], len(names)), dtype=np.float32)
    for channel_idx, (_, _, arr) in enumerate(iter_feature_channels(embs)):
        vals = arr[yy, xx]
        x[:, channel_idx] = np.nanmean(vals, axis=1).astype(np.float32)

    finite = np.isfinite(x).all(axis=1) & np.isfinite(total_valid)
    x = x[finite]
    total_valid = total_valid[finite]
    size_factor = total_valid / (np.mean(total_valid) + 1e-8)

    meta = {
        "n_spots_raw": int(len(spot_ids)),
        "n_spots_valid": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "radius_px": float(radius_px),
        "radius_grid": float(radius_grid),
        "mask_size": int(mask.sum()),
        "feature_shape": [int(height), int(width)],
    }
    return x, size_factor.astype(np.float32), names, meta


def compute_correlations(root: Path, slides: list[str], outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    best_rows = []
    metadata = {}
    cache_root = root / "data" / "superres_cache"
    for slide in slides:
        slide_dir = cache_root / slide
        if not slide_dir.exists():
            raise FileNotFoundError(slide_dir)
        print(f"[{slide}] aggregating HIPT features...", flush=True)
        x, size_factor, names, meta = aggregate_slide_features(slide_dir)
        y = np.log1p(size_factor)
        corr = pearson_columns(x, y)
        abs_corr = np.abs(corr)
        best_idx = int(np.nanargmax(abs_corr))
        metadata[slide] = {
            **meta,
            "best_feature": names[best_idx][1],
            "best_group": names[best_idx][0],
            "best_pearson_log1p_size_factor": float(corr[best_idx]),
            "median_abs_pearson": float(np.nanmedian(abs_corr)),
            "p95_abs_pearson": float(np.nanpercentile(abs_corr, 95)),
            "max_abs_pearson": float(np.nanmax(abs_corr)),
        }
        for idx, (group, name) in enumerate(names):
            rows.append(
                {
                    "slide": slide,
                    "feature": name,
                    "group": group,
                    "pearson_log1p_size_factor": float(corr[idx]),
                    "abs_pearson_log1p_size_factor": float(abs_corr[idx]),
                }
            )
        best_rows.append(
            pd.DataFrame(
                {
                    "slide": slide,
                    "feature_value": x[:, best_idx],
                    "size_factor": size_factor,
                    "log1p_size_factor": y,
                    "best_feature": names[best_idx][1],
                    "pearson": float(corr[best_idx]),
                }
            )
        )
        print(
            f"[{slide}] n={x.shape[0]} features={x.shape[1]} "
            f"max|r|={np.nanmax(abs_corr):.3f} median|r|={np.nanmedian(abs_corr):.3f}",
            flush=True,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    corr_df = pd.DataFrame(rows)
    best_df = pd.concat(best_rows, ignore_index=True)
    corr_df.to_csv(outdir / "hipt_size_factor_feature_correlations.csv", index=False)
    best_df.to_csv(outdir / "hipt_size_factor_best_feature_scatter_points.csv", index=False)
    (outdir / "hipt_size_factor_correlation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return corr_df, best_df


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def plot_summary(corr_df: pd.DataFrame, outdir: Path, top_k: int) -> Path:
    configure_style()
    slides = list(dict.fromkeys(corr_df["slide"]))
    pivot = corr_df.pivot(index="feature", columns="slide", values="pearson_log1p_size_factor")
    top_features = (
        pivot.abs()
        .mean(axis=1)
        .sort_values(ascending=False)
        .head(top_k)
        .index
    )
    heat = pivot.loc[top_features, slides]

    summary = (
        corr_df.groupby("slide")["abs_pearson_log1p_size_factor"]
        .agg(median_abs_r="median", p95_abs_r=lambda s: np.percentile(s, 95), max_abs_r="max")
        .reindex(slides)
        .reset_index()
    )
    group_summary = (
        corr_df.groupby(["slide", "group"])["abs_pearson_log1p_size_factor"]
        .mean()
        .reset_index()
    )

    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.05, 1.25])

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(slides))
    ax0.plot(x, summary["max_abs_r"], marker="o", lw=1.5, label="Max |r|")
    ax0.plot(x, summary["p95_abs_r"], marker="s", lw=1.5, label="95th percentile |r|")
    ax0.plot(x, summary["median_abs_r"], marker="^", lw=1.5, label="Median |r|")
    ax0.set_xticks(x)
    ax0.set_xticklabels(slides)
    ax0.set_ylim(0, min(1.0, max(0.05, summary["max_abs_r"].max() + 0.08)))
    ax0.set_ylabel("|Pearson r| with log1p(size factor)")
    ax0.set_title("HIPT channel-size factor correlation by slide")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(frameon=False)

    ax1 = fig.add_subplot(gs[0, 1])
    data = [corr_df.loc[corr_df["slide"].eq(slide), "abs_pearson_log1p_size_factor"].to_numpy() for slide in slides]
    parts = ax1.violinplot(data, positions=np.arange(len(slides)), widths=0.75, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#7AA6DC")
        body.set_edgecolor("#2F4B7C")
        body.set_alpha(0.75)
    parts["cmedians"].set_color("#1B1B1B")
    parts["cmedians"].set_linewidth(1.2)
    ax1.set_xticks(np.arange(len(slides)))
    ax1.set_xticklabels(slides)
    ax1.set_ylabel("|Pearson r|")
    ax1.set_title("Distribution across HIPT channels")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0])
    groups = ["cls", "sub", "rgb"]
    width = 0.23
    for gi, group in enumerate(groups):
        vals = []
        for slide in slides:
            sub = group_summary[group_summary["slide"].eq(slide) & group_summary["group"].eq(group)]
            vals.append(float(sub["abs_pearson_log1p_size_factor"].iloc[0]) if len(sub) else np.nan)
        ax2.bar(np.arange(len(slides)) + (gi - 1) * width, vals, width=width, label=group)
    ax2.set_xticks(np.arange(len(slides)))
    ax2.set_xticklabels(slides)
    ax2.set_ylabel("Mean |Pearson r|")
    ax2.set_title("Correlation strength by feature group")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, ncol=3)

    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(heat.to_numpy())), vmax=np.nanmax(np.abs(heat.to_numpy())))
    ax3.set_xticks(np.arange(len(slides)))
    ax3.set_xticklabels(slides)
    ax3.set_yticks(np.arange(len(top_features)))
    ax3.set_yticklabels(top_features)
    ax3.set_title(f"Top {len(top_features)} HIPT channels by mean |r|")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)
    cbar.set_label("Pearson r")

    out = outdir / "hipt_size_factor_correlation_summary.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_best_feature_scatter(best_df: pd.DataFrame, outdir: Path) -> Path:
    configure_style()
    slides = list(dict.fromkeys(best_df["slide"]))
    ncols = 2
    nrows = int(np.ceil(len(slides) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.4 * nrows), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, slide in enumerate(slides):
        ax = axes.ravel()[idx]
        ax.axis("on")
        sub = best_df[best_df["slide"].eq(slide)].copy()
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=13)
        ax.scatter(
            sub["feature_value"],
            sub["log1p_size_factor"],
            s=8,
            alpha=0.28,
            color="#276FBF",
            edgecolors="none",
        )
        feature = str(sub["best_feature"].iloc[0])
        r = float(sub["pearson"].iloc[0])
        ax.set_title(f"{slide}: {feature}, r={r:.3f}")
        ax.set_xlabel("Aggregated HIPT feature value")
        ax.set_ylabel("log1p(size factor)")
        ax.grid(alpha=0.2)
    out = outdir / "hipt_size_factor_best_feature_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    root = setup_root(args.root)
    slides = split_slides(args.slides)
    outdir = args.outdir or root / "figures_size_factor_hipt_correlation"
    corr_df, best_df = compute_correlations(root, slides, outdir)
    summary_path = plot_summary(corr_df, outdir, top_k=args.top_k)
    scatter_path = plot_best_feature_scatter(best_df, outdir)
    print(f"Saved summary figure: {summary_path}")
    print(f"Saved scatter figure: {scatter_path}")
    print(f"Saved correlations: {outdir / 'hipt_size_factor_feature_correlations.csv'}")


if __name__ == "__main__":
    main()
