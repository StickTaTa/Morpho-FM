from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab
from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL
from st_pipeline.superres.feature_cache import load_cache_slide, normalize_slide


GENE_VOCAB_PATH = ROOT / "assets" / "cellfm" / "expand_gene_info.csv"
CELLFM_CHECKPOINT = ROOT / "checkpoints" / "CellFM" / "CellFM_80M_weight.pt"
OUT_DIR = ROOT / "results" / "st_superres" / "xenium_extended"
PANEL_DIR = OUT_DIR / "panel"
FIG_DIR = PANEL_DIR / "figures"
PANEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRID_FACTOR = 16
PANEL_BY_CATEGORY: dict[str, list[str]] = {
    "tumour_luminal": ["ERBB2", "ESR1", "PGR", "FOXA1", "GATA3"],
    "epithelial": ["EPCAM", "KRT7", "KRT8", "TACSTD2", "CEACAM6"],
    "proliferation": ["MKI67", "TOP2A", "CCND1", "CENPF"],
    "stromal": ["LUM", "POSTN", "MMP2", "LRRC15", "SFRP1"],
    "vascular": ["PECAM1", "VWF", "RAMP2", "AQP1", "CAV1"],
    "immune": ["PTPRC", "CD3D", "CD4", "CD8A", "C1QA"],
}


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    setting: str
    section: str
    train_section: str
    mode: str
    ckpt_dir: Path
    result_dir: Path


RUNS = [
    RunConfig(
        run_name="XEN1_within",
        setting="XEN1 train -> XEN1 predict",
        section="XEN1",
        train_section="XEN1",
        mode="within",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
        result_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN1",
    ),
    RunConfig(
        run_name="XEN2_to_XEN1",
        setting="XEN2 train -> XEN1 predict",
        section="XEN1",
        train_section="XEN2",
        mode="cross",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
        result_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN1",
    ),
    RunConfig(
        run_name="XEN2_within",
        setting="XEN2 train -> XEN2 predict",
        section="XEN2",
        train_section="XEN2",
        mode="within",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
        result_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN2",
    ),
    RunConfig(
        run_name="XEN1_to_XEN2",
        setting="XEN1 train -> XEN2 predict",
        section="XEN2",
        train_section="XEN1",
        mode="cross",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
        result_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN2",
    ),
]


def panel_records() -> pd.DataFrame:
    rows = []
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
    a = a[mask] - np.nanmean(a[mask])
    b = b[mask] - np.nanmean(b[mask])
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(values.max()),
    }


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs**2).sum(-1) <= radius**2


def valid_center_locations(embs: np.ndarray, locs_px: np.ndarray, radius_px: float) -> np.ndarray:
    locs_grid = np.round(np.asarray(locs_px, dtype=np.float32) / float(GRID_FACTOR)).astype(np.int64)
    radius_grid = float(radius_px) / float(GRID_FACTOR)
    mask = get_disk_mask(radius_grid)
    shape = np.array(mask.shape)
    center = shape // 2
    offsets = np.stack([-center, shape - center], -1)
    r0, r1 = offsets[0], offsets[1]
    h, w = embs.shape[:2]
    valid = (
        (locs_grid[:, 0] + r0[0] >= 0)
        & (locs_grid[:, 0] + r0[1] <= h)
        & (locs_grid[:, 1] + r1[0] >= 0)
        & (locs_grid[:, 1] + r1[1] <= w)
    )
    valid &= np.isfinite(embs[locs_grid[:, 0], locs_grid[:, 1]]).all(-1)
    return locs_grid[valid]


def build_model(input_dim: int, n_genes: int, vocab_size: int, device: torch.device) -> MorphoCellfmMIL:
    return MorphoCellfmMIL(
        input_dim=input_dim,
        n_genes=n_genes,
        cellfm_dim=1536,
        cellfm_layers=2,
        cellfm_heads=48,
        cellfm_checkpoint=str(CELLFM_CHECKPOINT),
        freeze_cellfm=True,
        use_mock=False,
        use_retention=True,
        vocab_size=vocab_size,
        dropout=0.1,
        aggregation="mean",
        dispersion="gene",
    ).to(device)


def dense_backproject_panel(run: RunConfig, genes: list[str], device: torch.device) -> np.ndarray:
    run_out = PANEL_DIR / run.run_name
    run_out.mkdir(parents=True, exist_ok=True)
    pred_path = run_out / "pred_dense_backproj_panel.npy"
    gene_path = run_out / "pred_dense_backproj_panel_genes.json"
    if pred_path.exists() and gene_path.exists():
        saved_genes = json.loads(gene_path.read_text(encoding="utf-8"))
        if saved_genes == genes:
            return np.load(pred_path)

    vocab = load_gene_vocab(GENE_VOCAB_PATH)
    kept_genes, gene_ids = map_genes_to_vocab(genes, vocab)
    if kept_genes != genes:
        missing = [gene for gene in genes if gene not in kept_genes]
        raise ValueError(f"Selected genes missing from CellFM vocabulary: {missing}")

    common_genes = json.loads((run.ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    embs, cnts, locs_px, radius_px, _, _ = load_cache_slide(run.section, root=ROOT, genes=kept_genes)
    embs_norm, _, _ = normalize_slide(embs, cnts)
    locs_grid = valid_center_locations(embs_norm, locs_px, radius_px)
    center_embs = embs_norm[locs_grid[:, 0], locs_grid[:, 1]].astype(np.float32)

    model = build_model(center_embs.shape[1], len(common_genes), len(vocab), device)
    state = torch.load(run.ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    batch_size = 16384
    pred = np.empty((center_embs.shape[0], len(genes)), dtype=np.float32)
    gene_ids_t = torch.as_tensor(gene_ids, device=device, dtype=torch.long)
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, center_embs.shape[0], batch_size):
            end = min(start + batch_size, center_embs.shape[0])
            x = torch.from_numpy(center_embs[start:end]).to(device)
            gene_ids_batch = gene_ids_t.unsqueeze(0).expand(x.shape[0], -1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model.adapter(x)
                cell_emb, gene_emb = model.cellfm(z, gene_ids_batch)
                y_hat = model.cellfm.cellwise_decode(cell_emb, gene_emb)
                y_hat = torch.nn.functional.softplus(y_hat)
            pred[start:end] = y_hat.float().cpu().numpy()
            print(f"[{run.run_name}] dense backprojection {end:,}/{center_embs.shape[0]:,}")

    np.save(pred_path, pred)
    gene_path.write_text(json.dumps(genes, indent=2), encoding="utf-8")
    return pred


def compute_full_panel_spot_distributions(common_genes: list[str]) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        corrs = np.load(run.result_dir / "gene_corrs.npy")
        for gene, corr in zip(common_genes, corrs):
            rows.append(
                {
                    "run_name": run.run_name,
                    "setting": run.setting,
                    "section": run.section,
                    "train_section": run.train_section,
                    "mode": run.mode,
                    "gene": gene,
                    "spot_pearson": float(corr),
                }
            )
    return pd.DataFrame(rows)


def compute_panel_metrics(panel_df: pd.DataFrame, common_genes: list[str], device: torch.device) -> pd.DataFrame:
    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    genes = panel_df["gene"].tolist()
    rows = []
    for run in RUNS:
        true_spot = np.load(run.result_dir / "true_spot.npy", mmap_mode="r")
        spot_corrs = np.load(run.result_dir / "gene_corrs.npy")
        dense_pred = dense_backproject_panel(run, genes, device=device)
        for panel_idx, record in panel_df.reset_index(drop=True).iterrows():
            gene = record["gene"]
            common_idx = gene_to_idx[gene]
            dense_r = pearson_corr(dense_pred[:, panel_idx], true_spot[:, common_idx])
            rows.append(
                {
                    "run_name": run.run_name,
                    "setting": run.setting,
                    "section": run.section,
                    "train_section": run.train_section,
                    "mode": run.mode,
                    "gene": gene,
                    "category": record["category"],
                    "panel_rank": int(record["panel_rank"]),
                    "spot_pearson": float(spot_corrs[common_idx]),
                    "dense_backproj_pearson": dense_r,
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(full_df: pd.DataFrame, panel_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for run in RUNS:
        run_full = full_df[full_df["run_name"] == run.run_name]
        run_panel = panel_metrics[panel_metrics["run_name"] == run.run_name]
        for metric_name, values in [
            ("full_306_spot_pearson", run_full["spot_pearson"].to_numpy()),
            ("panel_spot_pearson", run_panel["spot_pearson"].to_numpy()),
            ("panel_dense_backproj_pearson", run_panel["dense_backproj_pearson"].to_numpy()),
        ]:
            row = {
                "run_name": run.run_name,
                "setting": run.setting,
                "section": run.section,
                "train_section": run.train_section,
                "mode": run.mode,
                "metric": metric_name,
            }
            row.update(summarize_values(values))
            summary_rows.append(row)

    category_rows = []
    for (run_name, category), df in panel_metrics.groupby(["run_name", "category"], sort=False):
        run = next(item for item in RUNS if item.run_name == run_name)
        for metric in ["spot_pearson", "dense_backproj_pearson"]:
            row = {
                "run_name": run_name,
                "setting": run.setting,
                "section": run.section,
                "train_section": run.train_section,
                "mode": run.mode,
                "category": category,
                "metric": metric,
            }
            row.update(summarize_values(df[metric].to_numpy()))
            category_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(category_rows)


def plot_setting_distributions(full_df: pd.DataFrame, panel_metrics: pd.DataFrame) -> None:
    plot_rows = []
    for run in RUNS:
        full_vals = full_df.loc[full_df["run_name"] == run.run_name, "spot_pearson"].to_numpy()
        panel_run = panel_metrics[panel_metrics["run_name"] == run.run_name]
        for value in full_vals:
            plot_rows.append({"setting": run.setting, "metric": "306-gene spot", "pearson": value})
        for _, row in panel_run.iterrows():
            plot_rows.append({"setting": run.setting, "metric": "29-gene spot", "pearson": row["spot_pearson"]})
            plot_rows.append(
                {
                    "setting": run.setting,
                    "metric": "29-gene dense backproj",
                    "pearson": row["dense_backproj_pearson"],
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    metric_order = ["306-gene spot", "29-gene spot", "29-gene dense backproj"]
    colors = {"within": "#2878b5", "cross": "#c75d2c"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, metric in zip(axes, metric_order):
        data = [plot_df[(plot_df["metric"] == metric) & (plot_df["setting"] == run.setting)]["pearson"].dropna() for run in RUNS]
        vp = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.78)
        for body, run in zip(vp["bodies"], RUNS):
            body.set_facecolor(colors[run.mode])
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            vp[key].set_color("black")
            vp[key].set_linewidth(1.0)
        ax.boxplot(data, widths=0.18, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "alpha": 0.7})
        ax.set_title(metric)
        ax.set_xticks(range(1, len(RUNS) + 1))
        ax.set_xticklabels([run.setting.replace(" -> ", "\n-> ") for run in RUNS], rotation=0, fontsize=8)
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xenium_four_setting_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "xenium_four_setting_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmaps(category_summary: pd.DataFrame) -> None:
    categories = list(PANEL_BY_CATEGORY.keys())
    for metric in ["spot_pearson", "dense_backproj_pearson"]:
        pivot = category_summary[category_summary["metric"] == metric].pivot(index="category", columns="setting", values="mean")
        pivot = pivot.loc[categories, [run.setting for run in RUNS]]
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([col.replace(" -> ", "\n-> ") for col in pivot.columns], fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
        ax.set_title(metric.replace("_", " "))
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mean Pearson")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"xenium_category_{metric}_heatmap.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"xenium_category_{metric}_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ROOT:", ROOT)
    print("Device:", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0))

    panel_df = panel_records()
    panel_df.to_csv(PANEL_DIR / "xenium_balanced_panel_genes.csv", index=False)
    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8"))
    missing = sorted(set(panel_df["gene"]) - set(common_genes))
    if missing:
        raise ValueError(f"Panel genes missing from common genes: {missing}")

    full_df = compute_full_panel_spot_distributions(common_genes)
    full_df.to_csv(PANEL_DIR / "xenium_306gene_spot_metrics_detail.csv", index=False)

    panel_metrics = compute_panel_metrics(panel_df, common_genes, device=device)
    panel_metrics.to_csv(PANEL_DIR / "xenium_balanced_panel_metrics_detail.csv", index=False)

    setting_summary, category_summary = summarize_metrics(full_df, panel_metrics)
    setting_summary.to_csv(PANEL_DIR / "xenium_setting_distribution_summary.csv", index=False)
    category_summary.to_csv(PANEL_DIR / "xenium_category_distribution_summary.csv", index=False)

    avg = (
        panel_metrics.groupby(["gene", "category", "mode"], as_index=False)[["spot_pearson", "dense_backproj_pearson"]]
        .mean()
        .pivot(index=["gene", "category"], columns="mode")
    )
    avg.columns = [f"{metric}_{mode}" for metric, mode in avg.columns.to_flat_index()]
    avg = avg.reset_index()
    avg.to_csv(PANEL_DIR / "xenium_balanced_panel_metrics_average_by_mode.csv", index=False)

    plot_setting_distributions(full_df, panel_metrics)
    plot_category_heatmaps(category_summary)
    print("[done] Extended Xenium panel results written to", PANEL_DIR)


if __name__ == "__main__":
    main()
