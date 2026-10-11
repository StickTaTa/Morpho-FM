from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


ROOT = Path(r"E:\Morpho-FM")
RESULT_ROOT = ROOT / "results" / "her2st_patient_h" / "whole_slide"
FILTERED_RESULT_ROOT = ROOT / "results" / "her2st_patient_h" / "whole_slide_fourier_notch"
SLIDES = ("H1", "H2", "H3")


def find_paper_dir() -> Path:
    desktop = Path.home() / "Desktop"
    for path in desktop.iterdir():
        if path.is_dir() and (path / "main.tex").exists() and (path / "Figure").exists():
            return path
    raise FileNotFoundError("Could not find manuscript directory under Desktop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Fourier-notch periodic artifact suppression to HER2ST dense gene maps."
    )
    parser.add_argument("--slides", nargs="+", default=list(SLIDES), choices=SLIDES)
    parser.add_argument("--genes", default=None, help="Comma-separated genes. Defaults to all selected genes per slide.")
    parser.add_argument("--notch-radius", type=float, default=3.0)
    parser.add_argument("--notch-strength", type=float, default=0.98)
    parser.add_argument("--background-sigma", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-panels", action="store_true", help="Only save filtered arrays.")
    return parser.parse_args()


def load_slide_dense(slide: str) -> tuple[np.ndarray, list[str], Path]:
    pred_dir = RESULT_ROOT / f"train_{slide}_to_{slide}" / slide
    genes = json.loads((pred_dir / "pred_superres_selected_genes.json").read_text(encoding="utf-8"))
    maps = np.load(pred_dir / "pred_superres_selected.npy", mmap_mode="r")
    return maps, genes, pred_dir


def estimate_period(slide: str) -> float:
    locs = pd.read_csv(ROOT / "data" / "superres_cache" / slide / "locs.tsv", sep=r"\s+", engine="python")
    xs = np.sort(np.unique(np.round(locs["x"].to_numpy(dtype=float) / 16.0).astype(int)))
    ys = np.sort(np.unique(np.round(locs["y"].to_numpy(dtype=float) / 16.0).astype(int)))
    diffs: list[int] = []
    for arr in (xs, ys):
        d = np.diff(arr)
        d = d[d > 4]
        diffs.extend(d.tolist())
    return float(np.median(diffs)) if diffs else 16.0


def normalized_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    v = np.nan_to_num(values, nan=0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = gaussian_filter(v * w, sigma=sigma, mode="nearest")
    den = gaussian_filter(w, sigma=sigma, mode="nearest")
    out = np.full_like(v, np.nan, dtype=np.float32)
    keep = den > 1e-4
    out[keep] = num[keep] / den[keep]
    return out


def make_notch_mask(shape: tuple[int, int], period: float, radius: float, strength: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    fy = h / period
    fx = w / period
    mask = np.ones((h, w), dtype=np.float32)

    def apply_notch(y0: float, x0: float) -> None:
        nonlocal mask
        dist2 = (yy - y0) ** 2 + (xx - x0) ** 2
        notch = 1.0 - strength * np.exp(-dist2 / (2 * radius * radius))
        mask *= notch.astype(np.float32)

    for m in range(1, 8):
        for sign in (-1, 1):
            if m * fx < w / 2 - 2:
                apply_notch(cy, cx + sign * m * fx)
            if m * fy < h / 2 - 2:
                apply_notch(cy + sign * m * fy, cx)
        if m * fx >= w / 2 - 2 and m * fy >= h / 2 - 2:
            break
    return np.clip(mask, 0.0, 1.0)


def fourier_notch_filter(
    raw: np.ndarray,
    notch_mask: np.ndarray,
    background_sigma: float,
) -> np.ndarray:
    valid = np.isfinite(raw)
    if not valid.any():
        return raw.astype(np.float32)
    background = normalized_gaussian(raw, valid, sigma=background_sigma)
    fill_value = float(np.nanmedian(raw[valid]))
    filled_raw = np.where(valid, raw, fill_value).astype(np.float32)
    filled_bg = np.where(np.isfinite(background), background, fill_value).astype(np.float32)
    high = filled_raw - filled_bg
    high -= float(np.mean(high[valid]))
    fft = np.fft.fftshift(np.fft.fft2(high))
    filtered_high = np.real(np.fft.ifft2(np.fft.ifftshift(fft * notch_mask))).astype(np.float32)
    out = filled_bg + filtered_high
    out[~valid] = np.nan
    return out.astype(np.float32)


def upsample_map(map_small: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    valid = np.isfinite(map_small).astype(np.float32)
    values = np.nan_to_num(map_small, nan=0.0).astype(np.float32)
    up_values = resize(values, target_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    up_valid = resize(valid, target_hw, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    up_values[~up_valid] = np.nan
    return up_values


def single_vrange(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(arr, [1, 99])
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_measurements(slide: str, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    locs = pd.read_csv(ROOT / "data" / "superres_cache" / slide / "locs.tsv", sep=r"\s+", engine="python")
    cnts = pd.read_csv(
        ROOT / "data" / "superres_cache" / slide / "cnts.tsv",
        sep="\t",
        usecols=["spot_id", *genes],
    )
    return locs, cnts


def render_gene_panel(
    slide: str,
    gene: str,
    raw: np.ndarray,
    filtered: np.ndarray,
    measured: np.ndarray,
    locs: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> dict[str, object]:
    he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
    raw_up = upsample_map(raw, he.shape[:2])
    filtered_up = upsample_map(filtered, he.shape[:2])
    meas_vmin, meas_vmax = single_vrange(measured)
    dense_vmin, dense_vmax = single_vrange(np.concatenate([raw_up[np.isfinite(raw_up)], filtered_up[np.isfinite(filtered_up)]]))

    fig, axes = plt.subplots(1, 4, figsize=(6.7, 1.95))
    titles = ["H&E", "Measured", "Raw dense", "Fourier-notch dense"]
    for ax, title in zip(axes, titles):
        ax.imshow(he)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    x = locs["x"].to_numpy(dtype=float)
    y = locs["y"].to_numpy(dtype=float)
    axes[1].scatter(
        x,
        y,
        c=measured,
        s=5,
        alpha=0.85,
        cmap="viridis",
        vmin=meas_vmin,
        vmax=meas_vmax,
        edgecolors="none",
        rasterized=True,
    )
    alpha_raw = np.where(np.isfinite(raw_up), 0.92, 0.0)
    axes[2].imshow(raw_up, cmap="viridis", vmin=dense_vmin, vmax=dense_vmax, alpha=alpha_raw)
    alpha_filtered = np.where(np.isfinite(filtered_up), 0.92, 0.0)
    im = axes[3].imshow(filtered_up, cmap="viridis", vmin=dense_vmin, vmax=dense_vmax, alpha=alpha_filtered)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.008)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_ticks([dense_vmin, dense_vmax])
    cbar.set_ticklabels(["low", "high"])
    fig.suptitle(f"{slide} {gene}", fontsize=9, y=0.99)
    fig.subplots_adjust(left=0.005, right=0.965, top=0.82, bottom=0.02, wspace=0.015)
    out_pdf = out_dir / f"{gene}_fourier_notch_panel.pdf"
    out_png = out_dir / f"{gene}_fourier_notch_panel.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "slide": slide,
        "gene": gene,
        "panel_pdf": str(out_pdf),
        "panel_png": str(out_png),
        "measured_vmin": meas_vmin,
        "measured_vmax": meas_vmax,
        "dense_vmin": dense_vmin,
        "dense_vmax": dense_vmax,
    }


def render_contact_sheet(slide: str, genes: list[str], filtered_maps: np.ndarray, out_dir: Path, dpi: int) -> None:
    show_genes = [g for g in ["KRT19", "ERBB2", "MGP", "IGKC", "IGHA1", "CD74", "ACTB"] if g in genes]
    if not show_genes:
        show_genes = genes[: min(6, len(genes))]
    he = np.array(Image.open(ROOT / "data" / "superres_cache" / slide / "he.jpg"))
    fig, axes = plt.subplots(1, len(show_genes), figsize=(6.7, 1.6))
    if len(show_genes) == 1:
        axes = [axes]
    ups = []
    for gene in show_genes:
        ups.append(upsample_map(filtered_maps[..., genes.index(gene)], he.shape[:2]))
    vals = np.concatenate([u[np.isfinite(u)] for u in ups])
    vmin, vmax = single_vrange(vals)
    im = None
    for ax, gene, arr in zip(axes, show_genes, ups):
        ax.imshow(he)
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, alpha=np.where(np.isfinite(arr), 0.92, 0.0))
        ax.set_title(gene, fontsize=8, fontstyle="italic")
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.008)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["low", "high"])
    fig.suptitle(f"{slide} Fourier-notch dense marker maps", fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.005, right=0.965, top=0.82, bottom=0.02, wspace=0.015)
    fig.savefig(out_dir / f"{slide}_fourier_notch_contact_sheet.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{slide}_fourier_notch_contact_sheet.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_slide(slide: str, selected_arg: str | None, args: argparse.Namespace, paper_out: Path) -> list[dict[str, object]]:
    raw_maps, genes, _ = load_slide_dense(slide)
    selected = [g.strip() for g in selected_arg.split(",") if g.strip()] if selected_arg else genes
    selected = [g for g in selected if g in genes]
    if not selected:
        raise ValueError(f"No selected genes found for {slide}")

    period = estimate_period(slide)
    notch_mask = make_notch_mask(raw_maps.shape[:2], period, args.notch_radius, args.notch_strength)
    out_result = FILTERED_RESULT_ROOT / f"train_{slide}_to_{slide}" / slide
    out_result.mkdir(parents=True, exist_ok=True)
    out_fig = paper_out / slide
    out_fig.mkdir(parents=True, exist_ok=True)

    filtered_path = out_result / "pred_superres_selected_fourier_notch.npy"
    gene_path = out_result / "pred_superres_selected_genes.json"
    if (
        filtered_path.exists()
        and gene_path.exists()
        and json.loads(gene_path.read_text(encoding="utf-8")) == genes
        and not args.force
    ):
        print(f"[{slide}] reusing filtered array: {filtered_path}")
        filtered_all = np.load(filtered_path, mmap_mode="r")
    else:
        filtered_all = np.full(raw_maps.shape, np.nan, dtype=np.float32)
        for idx, gene in enumerate(genes):
            filtered_all[..., idx] = fourier_notch_filter(
                np.asarray(raw_maps[..., idx], dtype=np.float32),
                notch_mask,
                background_sigma=args.background_sigma,
            )
            print(f"[{slide}] filtered {idx + 1}/{len(genes)} {gene}")

        np.save(filtered_path, filtered_all)
        np.save(out_result / "fourier_notch_mask.npy", notch_mask.astype(np.float32))
        gene_path.write_text(json.dumps(genes, indent=2), encoding="utf-8")
    meta = {
        "slide": slide,
        "source": str(RESULT_ROOT / f"train_{slide}_to_{slide}" / slide / "pred_superres_selected.npy"),
        "period_dense_grid": period,
        "notch_radius": args.notch_radius,
        "notch_strength": args.notch_strength,
        "background_sigma": args.background_sigma,
        "note": "Fourier-notch filtered maps are for visualization of periodic sampling artifacts; quantitative analyses should use raw predictions.",
    }
    (out_result / "fourier_notch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_fig / "fourier_notch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest: list[dict[str, object]] = []
    if not args.no_panels:
        locs, cnts = load_measurements(slide, selected)
        cnts_by_spot = cnts.set_index("spot_id")
        locs = locs.loc[locs["SpotID"].isin(cnts_by_spot.index)].copy()
        cnts_by_spot = cnts_by_spot.loc[locs["SpotID"].to_numpy()]
        for gene in selected:
            idx = genes.index(gene)
            manifest.append(
                render_gene_panel(
                    slide=slide,
                    gene=gene,
                    raw=np.asarray(raw_maps[..., idx], dtype=np.float32),
                    filtered=filtered_all[..., idx],
                    measured=cnts_by_spot[gene].to_numpy(dtype=float),
                    locs=locs,
                    out_dir=out_fig,
                    dpi=args.dpi,
                )
            )
        render_contact_sheet(slide, genes, filtered_all, out_fig, args.dpi)

    pd.DataFrame(manifest).to_csv(out_fig / "fourier_notch_panel_manifest.csv", index=False)
    return manifest


def main() -> None:
    args = parse_args()
    paper = find_paper_dir()
    paper_out = paper / "Figure" / "Figure_6" / "HER2ST_patient_H" / "12_dense_fourier_notch_marker_maps"
    paper_out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for slide in args.slides:
        all_rows.extend(run_slide(slide, args.genes, args, paper_out))
    pd.DataFrame(all_rows).to_csv(paper_out / "fourier_notch_panel_manifest.csv", index=False)
    print(f"[done] {paper_out}")


if __name__ == "__main__":
    main()
