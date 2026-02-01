from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_wsi_path(hest_dir: Path, slide_id: str) -> Path:
    wsis_dir = hest_dir / "wsis"
    for ext in (".tif", ".tiff", ".svs"):
        p = wsis_dir / f"{slide_id}{ext}"
        if p.exists():
            return p
    matches = list(wsis_dir.glob(f"{slide_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"WSI not found for {slide_id} in {wsis_dir}")


def _pad_to_multiple(img: np.ndarray, multiple: int, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img
    if img.ndim == 2:
        pad_spec = ((0, pad_h), (0, pad_w))
    else:
        pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(img, pad_spec, mode="constant", constant_values=pad_value)


def _save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def _thumbnail_wsi(wsi_path: Path, max_dim: int = 16000):
    try:
        import openslide
        slide = openslide.OpenSlide(str(wsi_path))
        w0, h0 = slide.dimensions
        scale = min(max_dim / max(w0, h0), 1.0)
        new_w = max(1, int(w0 * scale))
        new_h = max(1, int(h0 * scale))
        thumb = slide.get_thumbnail((new_w, new_h))
        return np.array(thumb), (w0, h0), (new_w, new_h)
    except Exception as e:
        warnings.warn(f"openslide failed for {wsi_path}: {e}. Falling back to tifffile.")

    try:
        import tifffile as tiff
    except Exception as e:
        raise RuntimeError("openslide failed and tifffile is not available") from e

    with tiff.TiffFile(wsi_path) as tf:
        series = tf.series[0]
        axes = series.axes.upper()
        arr = series.asarray(out="memmap")

    # Normalize to YXS (H, W, C) where possible
    if axes == "YXS":
        img = arr
    elif axes == "SYX":
        img = np.moveaxis(arr, 0, -1)
    elif axes == "YX":
        img = arr[..., None]
    else:
        # Minimal fallback: try to locate Y/X and move to front
        if "Y" in axes and "X" in axes:
            y_idx = axes.index("Y")
            x_idx = axes.index("X")
            img = np.moveaxis(arr, [y_idx, x_idx], [0, 1])
        else:
            raise RuntimeError(f"Unsupported TIFF axes '{axes}' in {wsi_path}")

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    h0, w0 = img.shape[:2]
    scale = min(max_dim / max(w0, h0), 1.0)
    if scale < 1.0:
        step = int(np.ceil(1.0 / scale))
        img = img[::step, ::step]
    img = np.asarray(img)

    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        vmin = float(np.nanmin(img_f))
        vmax = float(np.nanmax(img_f))
        denom = vmax - vmin if vmax > vmin else 1.0
        img = ((img_f - vmin) / denom * 255.0).astype(np.uint8)

    new_h, new_w = img.shape[:2]
    return img, (w0, h0), (new_w, new_h)


def _compute_tissue_mask(img: np.ndarray):
    # Tissue mask from color deconvolution (H) + grayscale fallback
    try:
        from skimage.color import rgb2gray, rgb2hed
        from skimage.filters import threshold_otsu
        from skimage.morphology import binary_opening, binary_closing, disk

        # Hematoxylin channel tends to highlight nuclei/tissue
        hed = rgb2hed(img)
        h = hed[..., 0]
        h_min = float(h.min())
        h_max = float(h.max())
        h = (h - h_min) / (h_max - h_min + 1e-8)

        # thr_h越小越宽松
        thr_h = threshold_otsu(h) * 0.8
        mask_h = h > thr_h

        gray = rgb2gray(img)
        thr_g = min(1.0, threshold_otsu(gray) * 1.2)
        mask_g = gray < thr_g

        mask = mask_h | mask_g
        mask = binary_opening(mask, disk(1))
        mask = binary_closing(mask, disk(2))
    except Exception:
        # Fallback: simple brightness threshold
        gray = img.mean(axis=2) / 255.0
        thr = np.percentile(gray, 80)
        mask = gray < thr
    return mask

# def _compute_tissue_mask(img: np.ndarray):
    try:
        from skimage.color import rgb2gray
        from skimage.filters import threshold_otsu
        gray = rgb2gray(img)
        thr = threshold_otsu(gray)
        mask = gray < thr
    except Exception:
        gray = img.mean(axis=2) / 255.0
        thr = np.percentile(gray, 80)
        mask = gray < thr

    # Light cleanup if scipy exists
    try:
        from scipy.ndimage import binary_opening, binary_closing
        mask = binary_opening(mask, iterations=1)
        mask = binary_closing(mask, iterations=2)
    except Exception:
        pass
    return mask

def _clean_tissue_mask(mask: np.ndarray, min_size: int, hole_size: int, keep_largest: bool = False):
    # Remove small islands/holes to reduce speckle noise.
    try:
        from skimage.morphology import remove_small_objects, remove_small_holes
        mask = remove_small_objects(mask, min_size=max(1, int(min_size)))
        mask = remove_small_holes(mask, area_threshold=max(1, int(hole_size)))
    except Exception:
        # If skimage is unavailable, skip cleaning.
        return mask

    if keep_largest:
        try:
            from scipy.ndimage import label
            lbl, n = label(mask)
            if n > 1:
                counts = np.bincount(lbl.ravel())
                counts[0] = 0
                keep = counts.argmax()
                mask = lbl == keep
        except Exception:
            pass
    return mask


def _downsample_mask(mask: np.ndarray, factor: int = 16) -> np.ndarray:
    h, w = mask.shape[:2]
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    mask = mask[:h2, :w2]
    mask = mask.reshape(h2 // factor, factor, w2 // factor, factor)
    mask = mask.mean(axis=(1, 3)) > 0.5
    return mask


def _load_tissue_contours(tissue_seg_dir: Path, slide_id: str):
    path = tissue_seg_dir / f"{slide_id}_contours.geojson"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    if not feats:
        return None
    return feats


def _rasterize_contours(
    feats,
    out_shape,
    scale_x: float,
    scale_y: float,
):
    # Rasterize GeoJSON polygons onto a binary mask.
    # Coordinates are assumed to be in full-res pixel space.
    h, w = out_shape
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

    def _draw_polygon(poly_coords):
        # poly_coords: list of rings (outer + holes)
        if not poly_coords:
            return
        outer = poly_coords[0]
        outer_xy = [(float(x) * scale_x, float(y) * scale_y) for x, y in outer]
        draw.polygon(outer_xy, fill=1)
        # holes
        for hole in poly_coords[1:]:
            hole_xy = [(float(x) * scale_x, float(y) * scale_y) for x, y in hole]
            draw.polygon(hole_xy, fill=0)

    for feat in feats:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            _draw_polygon(coords)
        elif gtype == "MultiPolygon":
            for poly in coords:
                _draw_polygon(poly)

    return np.array(mask_img).astype(bool)


def prepare_slide_cache(
    slide_id: str,
    cache_dir: Path,
    hest_dir: Path,
    spatial_dir: Path,
    max_dim: int = 16000,
    use_tissue_seg: bool = True,
    force: bool = False,
) -> Path:
    cache_dir = Path(cache_dir)
    slide_dir = cache_dir / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)

    meta_path = hest_dir / "metadata" / f"{slide_id}.json"
    meta = _read_json(meta_path)

    he_raw_path = slide_dir / "he-raw.jpg"
    he_scaled_path = slide_dir / "he-scaled.jpg"
    he_path = slide_dir / "he.jpg"
    mask_small_path = slide_dir / "mask-small.png"

    if force or (not he_path.exists()):
        wsi_path = _find_wsi_path(hest_dir, slide_id)
        img, (w0, h0), (w1, h1) = _thumbnail_wsi(wsi_path, max_dim=max_dim)
        _save_image(img, he_raw_path)
        _save_image(img, he_scaled_path)

        he_img = _pad_to_multiple(img, 256, pad_value=255)
        _save_image(he_img, he_path)

        mask = None
        if use_tissue_seg:
            tissue_seg_dir = hest_dir / "tissue_seg"
            feats = _load_tissue_contours(tissue_seg_dir, slide_id)
            if feats is not None:
                # Scale contours (full-res -> he)
                meta_full_w = int(meta.get("fullres_px_width", meta.get("fullres_width", w0)))
                meta_full_h = int(meta.get("fullres_px_height", meta.get("fullres_height", h0)))
                scale_x = he_img.shape[1] / float(meta_full_w)
                scale_y = he_img.shape[0] / float(meta_full_h)
                mask = _rasterize_contours(
                    feats=feats,
                    out_shape=he_img.shape[:2],
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                print(f"Used tissue contours from {tissue_seg_dir}")

        if mask is None:
            # Fallback: simple threshold-based mask
            mask = _compute_tissue_mask(he_img)

        # Clean mask before downsampling to reduce speckle noise
        mask = _clean_tissue_mask(mask, min_size=2000, hole_size=2000, keep_largest=True)

        mask_small = _downsample_mask(mask, factor=16)
        _save_image((mask_small * 255).astype(np.uint8), mask_small_path)

        meta_out = {
            "fullres_width": int(meta.get("fullres_px_width", meta.get("fullres_width", w0))),
            "fullres_height": int(meta.get("fullres_px_height", meta.get("fullres_height", h0))),
            "he_width": int(he_img.shape[1]),
            "he_height": int(he_img.shape[0]),
        }
        (slide_dir / "cache_meta.json").write_text(
            json.dumps(meta_out, indent=2), encoding="utf-8"
        )
        print(f"Saved images and mask for {slide_id}")

    # Build locs / counts / genes / radius
    locs_path = slide_dir / "locs.tsv"
    cnts_path = slide_dir / "cnts.tsv"
    genes_path = slide_dir / "gene-names.txt"
    radius_path = slide_dir / "radius.txt"

    if force or (not locs_path.exists()) or (not cnts_path.exists()):
        try:
            import scanpy as sc
            adata = sc.read_h5ad(spatial_dir / f"{slide_id}.h5ad")
        except Exception:
            import anndata as ad
            adata = ad.read_h5ad(spatial_dir / f"{slide_id}.h5ad")

        # counts
        import pandas as pd
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        cnts_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
        cnts_df.to_csv(cnts_path, sep="	")
        genes_path.write_text("\n".join(map(str, adata.var_names)), encoding="utf-8")

        # locs
        coords = adata.obsm["spatial"]
        meta_out = json.loads((slide_dir / "cache_meta.json").read_text(encoding="utf-8"))
        full_w = float(meta_out["fullres_width"])
        full_h = float(meta_out["fullres_height"])
        he_w = float(meta_out["he_width"])
        he_h = float(meta_out["he_height"])
        scale_x = he_w / full_w
        scale_y = he_h / full_h

        x = coords[:, 0] * scale_x
        y = coords[:, 1] * scale_y

        locs_df = pd.DataFrame({"SpotID": adata.obs_names, "x": x, "y": y})
        locs_df.to_csv(locs_path, sep="	", index=False)

        # radius
        spot_diameter = meta.get("spot_diameter", None)
        pixel_size = meta.get("pixel_size_um_embedded", None)
        if spot_diameter is not None and pixel_size is not None:
            spot_diameter_px = float(spot_diameter) / float(pixel_size)
        else:
            spot_diameter_px = float(spot_diameter) if spot_diameter is not None else 100.0
        radius_px_full = spot_diameter_px / 2.0
        radius_scaled = radius_px_full * ((scale_x + scale_y) / 2.0)
        radius_path.write_text(str(int(round(radius_scaled))), encoding="utf-8")

        print(f"Saved locs/cnts/genes/radius for {slide_id}")

    return slide_dir
