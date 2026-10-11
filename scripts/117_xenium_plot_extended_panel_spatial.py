from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "st_superres"
PANEL_ROOT = RESULT_ROOT / "xenium_extended" / "panel"
FIG_ROOT = PANEL_ROOT / "figures_spatial"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

RUN_DIRS = {
    "XEN1": {
        "within_dense": PANEL_ROOT / "XEN1_within",
        "cross_dense": PANEL_ROOT / "XEN2_to_XEN1",
        "truth": RESULT_ROOT / "xenium_cellfm_1" / "xenium_XEN1",
    },
    "XEN2": {
        "within_dense": PANEL_ROOT / "XEN2_within",
        "cross_dense": PANEL_ROOT / "XEN1_to_XEN2",
        "truth": RESULT_ROOT / "xenium_cellfm_2" / "xenium_XEN2",
    },
}

GRID_FACTOR = 16
DEFAULT_MAX_WORKERS = 4
WORKER_STATE: dict[str, object] = {}


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    a = a[mask] - np.nanmean(a[mask])
    b = b[mask] - np.nanmean(b[mask])
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


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


def available_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def auto_workers() -> int:
    cpu_count = os.cpu_count() or 1
    mem_bytes = available_memory_bytes()
    largest_he_bytes = 15616 * 12288 * 3
    bytes_per_worker = int(largest_he_bytes * 2.5)

    by_cpu = max(1, min(DEFAULT_MAX_WORKERS, cpu_count // 4 or 1))
    if mem_bytes is None:
        return by_cpu
    by_mem = max(1, min(DEFAULT_MAX_WORKERS, mem_bytes // bytes_per_worker))
    return max(1, min(by_cpu, by_mem))


def init_section_worker(section: str) -> None:
    Image.MAX_IMAGE_PIXELS = None
    section_runs = RUN_DIRS[section]
    panel_df = pd.read_csv(PANEL_ROOT / "xenium_balanced_panel_genes.csv")
    genes = panel_df["gene"].tolist()
    categories = dict(zip(panel_df["gene"], panel_df["category"]))
    common_genes = json.loads(
        (ROOT / "checkpoints" / "st_superres" / "xenium_cellfm" / "common_genes.json").read_text(encoding="utf-8")
    )

    WORKER_STATE.clear()
    WORKER_STATE["section"] = section
    WORKER_STATE["section_dir"] = FIG_ROOT / section
    WORKER_STATE["he_img"] = np.array(Image.open(ROOT / "data" / "superres_cache" / section / "he.jpg"))
    WORKER_STATE["coords"] = get_valid_coords(section)
    WORKER_STATE["true_arr"] = np.load(section_runs["truth"] / "true_spot.npy", mmap_mode="r")
    WORKER_STATE["within_arr"] = np.load(section_runs["within_dense"] / "pred_dense_backproj_panel.npy", mmap_mode="r")
    WORKER_STATE["cross_arr"] = np.load(section_runs["cross_dense"] / "pred_dense_backproj_panel.npy", mmap_mode="r")
    WORKER_STATE["gene_to_idx"] = {gene: idx for idx, gene in enumerate(common_genes)}
    WORKER_STATE["panel_gene_to_idx"] = {gene: idx for idx, gene in enumerate(genes)}
    WORKER_STATE["gene_category"] = categories


def render_gene(gene: str) -> dict[str, object]:
    section = WORKER_STATE["section"]
    section_dir = WORKER_STATE["section_dir"]
    he_img = WORKER_STATE["he_img"]
    coords = WORKER_STATE["coords"]
    true_arr = WORKER_STATE["true_arr"]
    within_arr = WORKER_STATE["within_arr"]
    cross_arr = WORKER_STATE["cross_arr"]
    gene_to_idx = WORKER_STATE["gene_to_idx"]
    panel_gene_to_idx = WORKER_STATE["panel_gene_to_idx"]
    gene_category = WORKER_STATE["gene_category"]

    common_idx = gene_to_idx[gene]
    panel_idx = panel_gene_to_idx[gene]

    measured_values = np.asarray(true_arr[:, common_idx], dtype=np.float32)
    within_values = np.asarray(within_arr[:, panel_idx], dtype=np.float32)
    cross_values = np.asarray(cross_arr[:, panel_idx], dtype=np.float32)

    x = coords[:, 1]
    y = coords[:, 0]
    within_r = pearson_corr(measured_values, within_values)
    cross_r = pearson_corr(measured_values, cross_values)
    measured_vmin, measured_vmax = single_vrange(measured_values)
    within_vmin, within_vmax = single_vrange(within_values)
    cross_vmin, cross_vmax = single_vrange(cross_values)

    fig = Figure(figsize=(18.6, 5.0))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)
    titles = [
        "H&E",
        f"Measured {gene}",
        f"Within observation-level recon\nr={within_r:.3f}",
        f"Cross observation-level recon\nr={cross_r:.3f}",
    ]

    for ax, title in zip(axes, titles):
        ax.imshow(he_img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    scatter_specs = [
        (axes[1], measured_values, measured_vmin, measured_vmax),
        (axes[2], within_values, within_vmin, within_vmax),
        (axes[3], cross_values, cross_vmin, cross_vmax),
    ]
    for ax, values, vmin, vmax in scatter_specs:
        sc = ax.scatter(
            x,
            y,
            c=values,
            s=2,
            alpha=0.85,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"{section} | {gene}", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.08)

    out_pdf = section_dir / f"{gene}_balanced_panel_spatial.pdf"
    out_png = section_dir / f"{gene}_balanced_panel_spatial.png"
    fig.savefig(out_pdf, dpi=220, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.clear()

    return {
        "section": section,
        "gene": gene,
        "category": gene_category[gene],
        "within_dense_backproj_pearson": within_r,
        "cross_dense_backproj_pearson": cross_r,
        "panel_pdf": str(out_pdf),
        "panel_png": str(out_png),
        "measured_vmin": measured_vmin,
        "measured_vmax": measured_vmax,
        "within_vmin": within_vmin,
        "within_vmax": within_vmax,
        "cross_vmin": cross_vmin,
        "cross_vmax": cross_vmax,
        "view_type": "observation_level_reconstruction",
    }


def build_bundle_from_pngs(image_paths: list[Path], out_pdf: Path) -> None:
    with PdfPages(out_pdf) as pdf:
        for image_path in image_paths:
            img = np.array(Image.open(image_path).convert("RGB"))
            fig = Figure(figsize=(18.6, 5.0))
            FigureCanvasAgg(fig)
            ax = fig.subplots(1, 1)
            ax.imshow(img)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            fig.clear()


def render_section(section: str, genes: list[str], workers: int) -> list[dict[str, object]]:
    section_dir = FIG_ROOT / section
    section_dir.mkdir(parents=True, exist_ok=True)
    effective_workers = max(1, workers)
    print(f"[{section}] rendering {len(genes)} genes with {effective_workers} worker(s)")

    init_section_worker(section)
    if effective_workers == 1:
        rows = [render_gene(gene) for gene in genes]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix=f"{section.lower()}-plot") as ex:
            rows = list(ex.map(render_gene, genes))

    rows.sort(key=lambda row: genes.index(str(row["gene"])))
    png_paths = [Path(str(row["panel_png"])) for row in rows]
    bundle_pdf = section_dir / f"{section}_balanced_panel_spatial_bundle.pdf"
    build_bundle_from_pngs(png_paths, bundle_pdf)
    contact_sheet = section_dir / f"{section}_balanced_panel_contact_sheet.png"
    build_contact_sheet(png_paths, contact_sheet, per_row=3)

    for row in rows:
        row["bundle_pdf"] = str(bundle_pdf)

    print(f"[saved] {bundle_pdf}")
    print(f"[saved] {contact_sheet}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Xenium 29-gene observation-level reconstruction panels.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes per section. Defaults to an auto-selected conservative value.",
    )
    return parser.parse_args()


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    args = parse_args()
    panel_df = pd.read_csv(PANEL_ROOT / "xenium_balanced_panel_genes.csv")
    genes = panel_df["gene"].tolist()
    workers = args.workers if args.workers is not None else auto_workers()
    print(f"[config] workers={workers}")

    manifest_rows: list[dict[str, object]] = []
    for section in RUN_DIRS:
        manifest_rows.extend(render_section(section, genes, workers=workers))

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(FIG_ROOT / "balanced_panel_spatial_manifest.csv", index=False)
    print(f"[saved] {FIG_ROOT / 'balanced_panel_spatial_manifest.csv'}")


if __name__ == "__main__":
    main()
