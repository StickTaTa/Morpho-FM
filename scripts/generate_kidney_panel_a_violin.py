from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path(__file__).resolve().parents[1])).resolve()
BENCH = ROOT / "benchmark" / "results" / "kidney_INT21"
SINGLE = ROOT / "results" / "kidney_hest"

TEST_ID = "INT21"
TRAIN_ID = "INT18"
# The single-slide half reads the rotating single-slide runs, not the older flat
# `single/INT18_to_INT21` directories. Both exist for several methods and they are not
# the same run: mclSTExp reads 0.0398 from the flat tree against 0.0353 from the
# rotating one, and Morpho-FM 0.2572 (single_superres) against 0.2463 (single_rotating).
# The rotating tree is the one the benchmark tables and Supplementary Table 2 use, so the
# figure follows it and every violin matches its table entry.
SINGLE_TAG = "INT18_to_INT13_INT14_INT15_INT17_INT19_INT21_INT24"
ROT_SINGLE_DIR = "single_rotating"
ROT_SINGLE_TAG = f"train_{SINGLE_TAG}"
OUTDIR_DEFAULT = ROOT / "figures_nature_kidney" / "panel_a_violin"
MM_TO_INCH = 1.0 / 25.4
# The composed Fig. 3a is sized to the figure height and gets 75.4 mm of the 170-mm row,
# so this aspect is what decides the on-page size -- widening it here would eat into
# panel b, not enlarge panel a. Nine violins fit that footprint at ~8.4 mm each as long
# as the labels run at 45 degrees; verified by rendering at the composite size.
FIG_WIDTH_MM = 85.0
FIG_HEIGHT_MM = 62.0

# Same palette as the prostate figures, so a method keeps one colour across the paper.
METHOD_COLORS = {
    "Morpho-FM": "#E64B35",
    "iStar": "#4DBBD5",
    "sCellST": "#00A087",
    "HiST": "#3C5488",
    "mclSTExp": "#F39B7F",
    "HisToGene": "#8491B4",
    "Path2Space": "#91D1C2",
    "ST-Net": "#8E7DBE",
    "STimage": "#7E6148",
}


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)
sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.6, "grid.alpha": 0.20, "axes.facecolor": "white"})


def req(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def pair(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = Path(folder)
    return np.load(req(folder / "pred_bag.npy")), np.load(req(folder / "true_bag.npy"))


def pearson_by_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-gene Pearson correlation across spots.

    Means are subtracted before the products are formed, which is what the benchmark's
    own per-fold `metrics.json` does and therefore what every table in the paper reports.

    The previous version used the algebraic identity num = Sxy - SxSy/n with the arrays
    cast to float32. That is mathematically the same quantity, but it loses all precision
    on genes whose values barely move: Sxx - Sx^2/n comes out slightly negative, sqrt
    returns NaN, and the gene is dropped from the mean without a word. ST-Net's
    predictions are near-constant across spots and it lost 7,031 of 17,217 genes on the
    single-slide side, reading 0.0132 where its own metrics.json says 0.0455. The other
    seven baselines, whose predictions do vary, were unaffected either way.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pm = pred - pred.mean(0, keepdims=True)
    tm = true - true.mean(0, keepdims=True)
    den = np.sqrt((pm * pm).sum(0)) * np.sqrt((tm * tm).sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (pm * tm).sum(0) / den
    out[~np.isfinite(out)] = np.nan
    return out


def chunk_start(path: Path) -> int:
    return int(path.stem.split("_")[2])


def load_istar_multi(scale_factor: int = 16) -> tuple[np.ndarray, np.ndarray]:
    folder = BENCH / "istar" / "data" / TEST_ID
    cache = folder / f"pred_spot_sf{scale_factor}.npz"
    if cache.exists():
        cached = np.load(cache, allow_pickle=True)
        pred = cached["pred"].astype(np.float32, copy=False)
        true = cached["true"].astype(np.float32, copy=False)
        if pred.shape == true.shape:
            return pred, true

    true_df = pd.read_csv(req(folder / "cnts.tsv"), sep="\t").set_index("SpotID")
    locs = pd.read_csv(req(folder / "locs.tsv"), sep="\t").set_index("SpotID").loc[true_df.index]
    grid_row = (locs["y"].to_numpy() / scale_factor).astype(int)
    grid_col = (locs["x"].to_numpy() / scale_factor).astype(int)

    pred = np.zeros(true_df.shape, dtype=np.float32)
    height = width = 0
    for i, gene in enumerate(true_df.columns.tolist()):
        gene_path = folder / "cnts-super" / f"{gene}.pickle"
        if not gene_path.exists():
            continue
        with open(gene_path, "rb") as handle:
            values = np.asarray(pickle.load(handle), dtype=np.float32)
        if height == 0:
            height, width = values.shape
            grid_row = np.clip(grid_row, 0, height - 1)
            grid_col = np.clip(grid_col, 0, width - 1)
        pred[:, i] = values[grid_row, grid_col]

    true = true_df.to_numpy(dtype=np.float32, copy=False)
    np.savez_compressed(
        cache,
        pred=pred,
        true=true,
        genes=np.asarray(true_df.columns.tolist(), dtype=object),
        spot_ids=np.asarray(true_df.index.tolist(), dtype=object),
    )
    return pred, true


def load_mclstexp_multi() -> tuple[np.ndarray, np.ndarray]:
    folder = BENCH / "mclSTExp"
    pred = np.concatenate(
        [np.load(path) for path in sorted(folder.glob("preds_chunk_*.npy"), key=chunk_start)],
        axis=1,
    ).astype(np.float32, copy=False)
    true = np.load(req(folder / "data" / TEST_ID / "preprocessed_matrix.npy")).T.astype(np.float32, copy=False)
    return pred, true


def methods_by_setting(setting: str) -> dict[str, callable]:
    # Nine methods on both sides. The five 2026-04 baselines read from the kidney_INT21
    # tree; Path2Space, ST-Net and STimage were added to the benchmark on 2026-09-23 and
    # their kidney runs were produced on 2026-10-08, into the same directories.
    if setting == "multi":
        return {
            "HisToGene": lambda: pair(BENCH / "histogene"),
            "iStar": load_istar_multi,
            "mclSTExp": load_mclstexp_multi,
            "sCellST": lambda: pair(BENCH / "sCellST"),
            "HiST": lambda: pair(BENCH / "hist"),
            "Path2Space": lambda: pair(BENCH / "path2space"),
            "ST-Net": lambda: pair(BENCH / "stnet"),
            "STimage": lambda: pair(BENCH / "stimage"),
            "Morpho-FM": lambda: pair(SINGLE / "morpho" / "multi_superres" / "INT21_holdout"),
        }
    return {
        "HisToGene": lambda: pair(SINGLE / "histogene" / "single" / SINGLE_TAG / TEST_ID),
        "iStar": lambda: pair(SINGLE / "istar" / "single" / SINGLE_TAG / TEST_ID),
        "mclSTExp": lambda: pair(SINGLE / "mclstexp" / "single" / SINGLE_TAG / TEST_ID),
        "sCellST": lambda: pair(SINGLE / "scellst" / "single" / SINGLE_TAG / TEST_ID),
        "HiST": lambda: pair(SINGLE / "hist" / "single" / SINGLE_TAG / TEST_ID),
        "Path2Space": lambda: pair(SINGLE / "path2space" / "single" / SINGLE_TAG / TEST_ID),
        "ST-Net": lambda: pair(SINGLE / "stnet" / "single" / SINGLE_TAG / TEST_ID),
        "STimage": lambda: pair(SINGLE / "stimage" / "single" / SINGLE_TAG / TEST_ID),
        "Morpho-FM": lambda: pair(SINGLE / "morpho" / ROT_SINGLE_DIR / ROT_SINGLE_TAG / TEST_ID),
    }


def collect_violin_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for setting in ("multi", "single"):
        setting_label = "multi-slide" if setting == "multi" else "single-slide"
        for method, loader in methods_by_setting(setting).items():
            print(f"[{setting}] loading {method}")
            pred, true = loader()
            corr = pearson_by_gene(pred, true)
            corr = corr[np.isfinite(corr)]
            rows.append(pd.DataFrame({"setting": setting_label, "method": method, "pearson": corr}))
            summary_rows.append(
                {
                    "setting": setting_label,
                    "method": method,
                    "mean_pearson": float(np.nanmean(corr)),
                    "median_pearson": float(np.nanmedian(corr)),
                    "n_genes": int(corr.size),
                }
            )
            del pred, true, corr
            gc.collect()
    violin_df = pd.concat(rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return violin_df, summary_df


def unified_order(summary_df: pd.DataFrame) -> list[str]:
    return (
        summary_df.groupby("method")["mean_pearson"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def plot_panel_a(violin_df: pd.DataFrame, summary_df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    order = unified_order(summary_df)
    hue_order = ["multi-slide", "single-slide"]
    hue_palette = {
        "multi-slide": "#E64B35",
        "single-slide": "#4DBBD5",
    }

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM * MM_TO_INCH))
    violin_kwargs = dict(
        data=violin_df,
        x="method",
        y="pearson",
        hue="setting",
        order=order,
        hue_order=hue_order,
        palette=hue_palette,
        split=True,
        inner="quart",
        cut=0,
        linewidth=0.8,
        width=0.88,
        saturation=1.0,
        ax=ax,
    )
    try:
        sns.violinplot(
            **violin_kwargs,
            density_norm="width",
            common_norm=False,
            bw_adjust=0.9,
        )
    except TypeError:
        sns.violinplot(
            **violin_kwargs,
            scale="width",
            scale_hue=False,
            bw_adjust=0.9,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.2, 1.0)
    # 45 degrees rather than the 24 used with six methods: at nine, the longest names
    # (Path2Space, HisToGene) collide with their neighbours at the shallower angle
    ax.tick_params(axis="x", rotation=45, labelsize=8, pad=1)
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.grid(axis="y", linewidth=0.5, alpha=0.18)
    sns.despine(ax=ax, top=True, right=True)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        if tick.get_text() == "Morpho-FM":
            tick.set_fontweight("bold")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles[:2],
            labels[:2],
            frameon=False,
            loc="upper right",
            title=None,
            borderaxespad=0.2,
            handletextpad=0.4,
            labelspacing=0.3,
        )

    fig.tight_layout(pad=0.4)

    base = outdir / "panel_a_kidney_multi_single_violin"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)

    summary_df.to_csv(outdir / "panel_a_kidney_multi_single_violin_summary.csv", index=False)
    return base.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate panel a only as a combined multi/single violin figure.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR_DEFAULT,
        help="Output directory for panel a.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    violin_df, summary_df = collect_violin_data()
    out_pdf = plot_panel_a(violin_df, summary_df, args.outdir)
    print("[done] panel a generated:")
    for path in sorted(args.outdir.iterdir()):
        print(f"  - {path}")
    print(f"[main] {out_pdf}")


if __name__ == "__main__":
    main()
