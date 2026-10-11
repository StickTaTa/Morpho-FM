from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image
from skimage.transform import resize


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", Path(__file__).resolve().parents[1])).resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from st_pipeline.data.gene_vocab import load_gene_vocab, map_genes_to_vocab
from st_pipeline.model.morpho_cellfm_mil import MorphoCellfmMIL
from st_pipeline.superres.feature_cache import load_cache_slide, normalize_slide


GENE_VOCAB_PATH = ROOT / "assets" / "cellfm" / "expand_gene_info.csv"
CELLFM_CHECKPOINT = ROOT / "checkpoints" / "CellFM" / "CellFM_80M_weight.pt"
OUT_ROOT = ROOT / "results" / "st_superres" / "xenium_extended" / "whole_slide"
FIG_ROOT = OUT_ROOT / "figures"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

GRID_FACTOR = 16
PLOT_WORKER_STATE: dict[str, object] = {}
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
    truth_dir: Path


RUNS = [
    RunConfig(
        run_name="XEN1_within",
        setting="XEN1 train -> XEN1 whole-slide",
        section="XEN1",
        train_section="XEN1",
        mode="within",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
        truth_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN1",
    ),
    RunConfig(
        run_name="XEN2_to_XEN1",
        setting="XEN2 train -> XEN1 whole-slide",
        section="XEN1",
        train_section="XEN2",
        mode="cross",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
        truth_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN1",
    ),
    RunConfig(
        run_name="XEN2_within",
        setting="XEN2 train -> XEN2 whole-slide",
        section="XEN2",
        train_section="XEN2",
        mode="within",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm_2",
        truth_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN2",
    ),
    RunConfig(
        run_name="XEN1_to_XEN2",
        setting="XEN1 train -> XEN2 whole-slide",
        section="XEN2",
        train_section="XEN1",
        mode="cross",
        ckpt_dir=ROOT / "checkpoints" / "st_superres" / "xenium_cellfm",
        truth_dir=ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN2",
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


def single_vrange(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(
        values,
        target_hw,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)
    up_valid = resize(
        valid,
        target_hw,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def get_disk_mask(radius: float) -> np.ndarray:
    radius_ceil = int(np.ceil(radius))
    locs = np.meshgrid(
        np.arange(-radius_ceil, radius_ceil + 1),
        np.arange(-radius_ceil, radius_ceil + 1),
        indexing="ij",
    )
    locs = np.stack(locs, axis=-1)
    return (locs ** 2).sum(-1) <= radius ** 2


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


def dense_predict_run(
    run: RunConfig,
    genes: list[str],
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    run_dir = OUT_ROOT / run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "pred_superres_balanced_panel.npy"
    gene_path = run_dir / "pred_superres_balanced_panel_genes.json"
    meta_path = run_dir / "dense_decode_meta.json"

    if not force and pred_path.exists() and gene_path.exists():
        saved_genes = json.loads(gene_path.read_text(encoding="utf-8"))
        if saved_genes == genes:
            pred_maps = np.load(pred_path, mmap_mode="r")
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return pred_maps, saved_genes, meta

    vocab = load_gene_vocab(GENE_VOCAB_PATH)
    kept_genes, gene_ids = map_genes_to_vocab(genes, vocab)
    if kept_genes != genes:
        missing = [g for g in genes if g not in kept_genes]
        raise ValueError(f"Genes missing from CellFM vocab: {missing}")

    common_genes = json.loads((run.ckpt_dir / "common_genes.json").read_text(encoding="utf-8"))
    embs, cnts, _, _, _, _ = load_cache_slide(run.section, root=ROOT, genes=kept_genes)
    embs_norm, _, _ = normalize_slide(embs, cnts)

    cache_dir = ROOT / "data" / "superres_cache" / run.section
    mask_small = np.array(Image.open(cache_dir / "mask-small.png")) > 0
    h, w, channels = embs_norm.shape
    if mask_small.shape != (h, w):
        mask_small = resize(
            mask_small.astype(np.float32),
            (h, w),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5

    finite_mask = np.isfinite(embs_norm).all(-1)
    valid_mask = finite_mask & mask_small
    valid_idx = np.flatnonzero(valid_mask.ravel())
    valid_embs = embs_norm.reshape(-1, channels)[valid_idx].astype(np.float32)

    model = build_model(
        input_dim=channels,
        n_genes=len(common_genes),
        vocab_size=len(vocab),
        device=device,
    )
    state = torch.load(run.ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    pred_valid = np.empty((len(valid_embs), len(genes)), dtype=np.float32)
    gene_ids_t = torch.as_tensor(gene_ids, device=device, dtype=torch.long)
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for start in range(0, len(valid_embs), batch_size):
            end = min(start + batch_size, len(valid_embs))
            x = torch.from_numpy(valid_embs[start:end]).to(device)
            gene_ids_batch = gene_ids_t.unsqueeze(0).expand(x.shape[0], -1)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model.adapter(x)
                cell_emb, gene_emb = model.cellfm(z, gene_ids_batch)
                pred = model.cellfm.cellwise_decode(cell_emb, gene_emb)
                pred = torch.nn.functional.softplus(pred)
            pred_valid[start:end] = pred.float().cpu().numpy()
            if start == 0 or (start // batch_size) % 10 == 0:
                print(f"[{run.run_name}] dense whole-slide decode {end}/{len(valid_embs)}")

    pred_maps = np.full((h, w, len(genes)), np.nan, dtype=np.float32)
    pred_maps.reshape(-1, len(genes))[valid_idx] = pred_valid

    np.save(pred_path, pred_maps)
    gene_path.write_text(json.dumps(genes, indent=2), encoding="utf-8")
    meta = {
        "height": int(h),
        "width": int(w),
        "channels": int(channels),
        "valid_grid_points": int(len(valid_idx)),
        "n_genes": int(len(genes)),
        "batch_size": int(batch_size),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[saved] {pred_path}")
    return pred_maps, genes, meta


def render_gene_panel(
    section: str,
    he_img: np.ndarray,
    gene: str,
    coords: np.ndarray,
    measured_values: np.ndarray,
    within_map: np.ndarray,
    cross_map: np.ndarray,
    out_pdf: Path,
    out_png: Path,
) -> dict[str, float]:
    within_up = upsample_map(within_map, he_img.shape[:2])
    cross_up = upsample_map(cross_map, he_img.shape[:2])

    measured_vmin, measured_vmax = single_vrange(measured_values)
    within_vmin, within_vmax = single_vrange(within_up)
    cross_vmin, cross_vmax = single_vrange(cross_up)

    fig = Figure(figsize=(20.0, 5.2))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)
    titles = [
        "H&E",
        f"Measured sparse obs\n{gene}",
        f"Within whole-slide dense\n{gene}",
        f"Cross whole-slide dense\n{gene}",
    ]

    for ax, title in zip(axes, titles):
        ax.imshow(he_img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    x = coords[:, 1]
    y = coords[:, 0]
    sc = axes[1].scatter(
        x,
        y,
        c=measured_values,
        s=2,
        alpha=0.85,
        cmap="viridis",
        vmin=measured_vmin,
        vmax=measured_vmax,
        edgecolors="none",
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)

    overlays = [
        (axes[2], within_up, within_vmin, within_vmax),
        (axes[3], cross_up, cross_vmin, cross_vmax),
    ]
    for ax, overlay, vmin, vmax in overlays:
        alpha = np.where(np.isfinite(overlay), 0.92, 0.0)
        im = ax.imshow(overlay, cmap="viridis", vmin=vmin, vmax=vmax, alpha=alpha)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"{section} | {gene}", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.08)
    fig.savefig(out_pdf, dpi=220, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.clear()
    return {
        "measured_vmin": measured_vmin,
        "measured_vmax": measured_vmax,
        "within_vmin": within_vmin,
        "within_vmax": within_vmax,
        "cross_vmin": cross_vmin,
        "cross_vmax": cross_vmax,
    }


def build_contact_sheet(image_paths: list[Path], out_path: Path, per_row: int = 3) -> None:
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    widths, heights = zip(*(image.size for image in images))
    cell_w = max(widths)
    cell_h = max(heights)
    rows = (len(images) + per_row - 1) // per_row
    canvas = Image.new("RGB", (cell_w * per_row, cell_h * rows), color=(255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // per_row
        col = idx % per_row
        canvas.paste(image, (col * cell_w, row * cell_h))
        image.close()
    canvas.save(out_path)


def build_bundle_from_pngs(image_paths: list[Path], out_pdf: Path) -> None:
    with PdfPages(out_pdf) as pdf:
        for image_path in image_paths:
            img = np.array(Image.open(image_path).convert("RGB"))
            fig = Figure(figsize=(20.0, 5.2))
            FigureCanvasAgg(fig)
            ax = fig.subplots(1, 1)
            ax.imshow(img)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            fig.clear()


def init_plot_worker(
    section: str,
    cfg: dict[str, Path],
    genes: list[str],
    category_by_gene: dict[str, str],
) -> None:
    Image.MAX_IMAGE_PIXELS = None
    section_dir = FIG_ROOT / section
    section_dir.mkdir(parents=True, exist_ok=True)

    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8"))
    PLOT_WORKER_STATE.clear()
    PLOT_WORKER_STATE["section"] = section
    PLOT_WORKER_STATE["section_dir"] = section_dir
    PLOT_WORKER_STATE["he_img"] = np.array(Image.open(ROOT / "data" / "superres_cache" / section / "he.jpg"))
    PLOT_WORKER_STATE["coords"] = get_valid_coords(section)
    PLOT_WORKER_STATE["true_arr"] = np.load(cfg["truth"], mmap_mode="r")
    PLOT_WORKER_STATE["within_maps"] = np.load(cfg["within"], mmap_mode="r")
    PLOT_WORKER_STATE["cross_maps"] = np.load(cfg["cross"], mmap_mode="r")
    PLOT_WORKER_STATE["gene_to_idx"] = {gene: idx for idx, gene in enumerate(common_genes)}
    PLOT_WORKER_STATE["panel_gene_to_idx"] = {gene: idx for idx, gene in enumerate(genes)}
    PLOT_WORKER_STATE["category_by_gene"] = category_by_gene


def empty_range_dict() -> dict[str, float]:
    return {
        "measured_vmin": float("nan"),
        "measured_vmax": float("nan"),
        "within_vmin": float("nan"),
        "within_vmax": float("nan"),
        "cross_vmin": float("nan"),
        "cross_vmax": float("nan"),
    }


def render_gene_worker(gene: str) -> dict[str, object]:
    section = str(PLOT_WORKER_STATE["section"])
    section_dir = Path(PLOT_WORKER_STATE["section_dir"])
    he_img = np.asarray(PLOT_WORKER_STATE["he_img"])
    coords = np.asarray(PLOT_WORKER_STATE["coords"])
    true_arr = PLOT_WORKER_STATE["true_arr"]
    within_maps = PLOT_WORKER_STATE["within_maps"]
    cross_maps = PLOT_WORKER_STATE["cross_maps"]
    gene_to_idx = dict(PLOT_WORKER_STATE["gene_to_idx"])
    panel_gene_to_idx = dict(PLOT_WORKER_STATE["panel_gene_to_idx"])
    category_by_gene = dict(PLOT_WORKER_STATE["category_by_gene"])

    out_pdf = section_dir / f"{gene}_whole_slide_balanced_panel.pdf"
    out_png = section_dir / f"{gene}_whole_slide_balanced_panel.png"
    row = {
        "section": section,
        "gene": gene,
        "category": category_by_gene[gene],
        "panel_pdf": str(out_pdf),
        "panel_png": str(out_png),
        "view_type": "whole_slide_dense_superres",
    }

    if out_pdf.exists() and out_png.exists():
        row.update(empty_range_dict())
        row["render_status"] = "existing"
        return row

    common_idx = gene_to_idx[gene]
    panel_idx = panel_gene_to_idx[gene]
    ranges = render_gene_panel(
        section=section,
        he_img=he_img,
        gene=gene,
        coords=coords,
        measured_values=np.asarray(true_arr[:, common_idx], dtype=np.float32),
        within_map=np.asarray(within_maps[..., panel_idx], dtype=np.float32),
        cross_map=np.asarray(cross_maps[..., panel_idx], dtype=np.float32),
        out_pdf=out_pdf,
        out_png=out_png,
    )
    row.update(ranges)
    row["render_status"] = "rendered"
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")
    return row


def export_panels(panel_df: pd.DataFrame, plot_workers: int) -> None:
    genes = panel_df["gene"].tolist()
    common_genes = json.loads((ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8"))
    category_by_gene = dict(zip(panel_df["gene"], panel_df["category"]))

    section_configs = {
        "XEN1": {
            "within": OUT_ROOT / "XEN1_within" / "pred_superres_balanced_panel.npy",
            "cross": OUT_ROOT / "XEN2_to_XEN1" / "pred_superres_balanced_panel.npy",
            "truth": ROOT / "results" / "st_superres" / "xenium_cellfm_1" / "xenium_XEN1" / "true_spot.npy",
        },
        "XEN2": {
            "within": OUT_ROOT / "XEN2_within" / "pred_superres_balanced_panel.npy",
            "cross": OUT_ROOT / "XEN1_to_XEN2" / "pred_superres_balanced_panel.npy",
            "truth": ROOT / "results" / "st_superres" / "xenium_cellfm_2" / "xenium_XEN2" / "true_spot.npy",
        },
    }

    manifest_rows: list[dict[str, object]] = []
    for section, cfg in section_configs.items():
        section_dir = FIG_ROOT / section
        section_dir.mkdir(parents=True, exist_ok=True)
        print(f"[plot] {section}: workers={plot_workers}")
        if plot_workers <= 1:
            init_plot_worker(section, cfg, genes, category_by_gene)
            section_rows = [render_gene_worker(gene) for gene in genes]
        else:
            with ProcessPoolExecutor(
                max_workers=plot_workers,
                initializer=init_plot_worker,
                initargs=(section, cfg, genes, category_by_gene),
            ) as ex:
                section_rows = list(ex.map(render_gene_worker, genes, chunksize=1))
        section_rows.sort(key=lambda row: genes.index(str(row["gene"])))

        png_paths = [Path(str(row["panel_png"])) for row in section_rows]
        missing_pngs = [str(path) for path in png_paths if not path.exists()]
        if missing_pngs:
            raise FileNotFoundError(f"Missing rendered pngs for {section}: {missing_pngs[:3]}")

        bundle_pdf = section_dir / f"{section}_whole_slide_balanced_panel_bundle.pdf"
        build_bundle_from_pngs(png_paths, bundle_pdf)
        contact_sheet = section_dir / f"{section}_whole_slide_balanced_panel_contact_sheet.png"
        build_contact_sheet(png_paths, contact_sheet, per_row=3)
        print(f"[saved] {bundle_pdf}")
        print(f"[saved] {contact_sheet}")
        for row in section_rows:
            row["bundle_pdf"] = str(bundle_pdf)
        manifest_rows.extend(section_rows)

    pd.DataFrame(manifest_rows).to_csv(FIG_ROOT / "whole_slide_balanced_panel_manifest.csv", index=False)
    print(f"[saved] {FIG_ROOT / 'whole_slide_balanced_panel_manifest.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 29-gene Xenium whole-slide dense prediction maps and figures.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Inference batch size for dense grid decoding.")
    parser.add_argument("--plot-workers", type=int, default=1, help="Number of parallel worker processes for full-resolution figure rendering.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if output arrays already exist.")
    parser.add_argument("--predict-only", action="store_true", help="Only generate whole-slide arrays, skip plotting.")
    parser.add_argument("--plot-only", action="store_true", help="Only render figures from existing arrays.")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit CPU execution. Without this flag, CUDA is required.")
    return parser.parse_args()


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    args = parse_args()
    if args.predict_only and args.plot_only:
        raise ValueError("--predict-only and --plot-only cannot be used together.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ROOT:", ROOT)
    print("Device:", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0))
    elif not args.allow_cpu:
        raise RuntimeError("CUDA device not available. Re-run on the server GPU or pass --allow-cpu intentionally.")

    panel_df = panel_records()
    panel_df.to_csv(OUT_ROOT / "xenium_balanced_panel_genes.csv", index=False)
    genes = panel_df["gene"].tolist()

    if not args.plot_only:
        run_rows = []
        for run in RUNS:
            _, _, meta = dense_predict_run(
                run=run,
                genes=genes,
                device=device,
                batch_size=args.batch_size,
                force=args.force,
            )
            run_rows.append(
                {
                    "run_name": run.run_name,
                    "setting": run.setting,
                    "section": run.section,
                    "train_section": run.train_section,
                    "mode": run.mode,
                    **meta,
                }
            )
        pd.DataFrame(run_rows).to_csv(OUT_ROOT / "whole_slide_run_summary.csv", index=False)
        print(f"[saved] {OUT_ROOT / 'whole_slide_run_summary.csv'}")

    if not args.predict_only:
        export_panels(panel_df, plot_workers=args.plot_workers)


if __name__ == "__main__":
    main()
