from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


SLIDES = ("H1", "H2", "H3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HER2ST patient H slices as Morpho-FM spatial_data + hest_data inputs."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def make_spot_id(x, y) -> str:
    return f"{int(round(float(x)))}x{int(round(float(y)))}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_hest_row(root: Path, slide_id: str) -> dict:
    csv_path = root / "benchmark" / "sCellST" / "data" / "HEST_v1_1_0.csv"
    hest = pd.read_csv(csv_path)
    row = hest.loc[hest["subseries"] == slide_id]
    if row.empty:
        raise FileNotFoundError(f"Could not find HEST metadata row for {slide_id} in {csv_path}")
    return row.iloc[0].to_dict()


def sanitize_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_obs(root: Path, slide_id: str) -> pd.DataFrame:
    raw_dir = root / "data" / "HER2ST_breast_cancer" / "raw"
    sel_path = raw_dir / "spot-selections" / f"{slide_id}_selection.tsv.gz"
    sel = pd.read_csv(sel_path, sep="\t")
    sel["spot_id"] = [make_spot_id(x, y) for x, y in zip(sel["x"], sel["y"])]
    sel = sel.set_index("spot_id", drop=True)

    if slide_id == "H1":
        label_path = raw_dir / "meta" / "H1_labeled_coordinates.tsv"
        labels = pd.read_csv(label_path, sep="\t")
        labels["spot_id"] = [make_spot_id(x, y) for x, y in zip(labels["x"], labels["y"])]
        labels = labels[["spot_id", "label"]].drop_duplicates("spot_id").set_index("spot_id")
        sel = sel.join(labels, how="left")
    else:
        sel["label"] = pd.Series(index=sel.index, dtype="object")

    sel["label"] = sel["label"].fillna("unlabeled")
    sel["selected"] = sel["selected"].astype(int)
    sel["array_x"] = sel["x"].astype(int)
    sel["array_y"] = sel["y"].astype(int)
    sel["pixel_x"] = sel["pixel_x"].astype(float)
    sel["pixel_y"] = sel["pixel_y"].astype(float)
    sel["new_x"] = sel["new_x"].astype(float)
    sel["new_y"] = sel["new_y"].astype(float)
    return sel


def build_count_matrix(root: Path, slide_id: str) -> pd.DataFrame:
    raw_dir = root / "data" / "HER2ST_breast_cancer" / "raw"
    cnt_path = raw_dir / "count-matrices" / "count-matrices" / f"{slide_id}.tsv.gz"
    cnt = pd.read_csv(cnt_path, sep="\t", index_col=0)
    cnt.index = cnt.index.astype(str)
    cnt = cnt.loc[~cnt.index.duplicated(keep="first")]
    return cnt


def build_adata(root: Path, slide_id: str) -> ad.AnnData:
    obs = build_obs(root, slide_id)
    cnt = build_count_matrix(root, slide_id)

    common_ids = [sid for sid in obs.index if sid in cnt.index]
    if not common_ids:
        raise ValueError(f"No overlapping spot IDs between selection and count matrix for {slide_id}")

    obs = obs.loc[common_ids].copy()
    cnt = cnt.loc[common_ids].copy()

    X = sparse.csr_matrix(cnt.to_numpy(dtype=np.float32))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=cnt.columns.astype(str)))
    adata.obs_names = obs.index.astype(str)
    adata.var_names = cnt.columns.astype(str)
    adata.obsm["spatial"] = obs[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32)
    adata.obsm["array"] = obs[["array_x", "array_y"]].to_numpy(dtype=np.int32)
    adata.uns["dataset"] = "HER2ST"
    adata.uns["patient"] = "H"
    adata.uns["subseries"] = slide_id
    adata.uns["label_available"] = bool((adata.obs["label"] != "unlabeled").any())
    return adata


def write_h5ad(root: Path, slide_id: str, force: bool) -> Path:
    out_path = root / "data" / "spatial_data" / f"{slide_id}.h5ad"
    if out_path.exists() and not force:
        return out_path
    adata = build_adata(root, slide_id)
    adata.write_h5ad(out_path)
    return out_path


def write_metadata_json(root: Path, slide_id: str, force: bool) -> Path:
    out_path = root / "data" / "hest_data" / "metadata" / f"{slide_id}.json"
    if out_path.exists() and not force:
        return out_path

    row = load_hest_row(root, slide_id)
    pixel_size = row.get("pixel_size_um_embedded")
    if pd.isna(pixel_size):
        pixel_size = row.get("pixel_size_um_estimated")

    payload = {
        "dataset_title": f"HER2ST Patient H {slide_id}",
        "id": row.get("id", slide_id),
        "image_filename": f"{slide_id}.tif",
        "organ": row.get("organ", "Breast"),
        "disease_state": row.get("disease_state", "Cancer"),
        "oncotree_code": row.get("oncotree_code"),
        "species": row.get("species", "Homo sapiens"),
        "patient": row.get("patient", "Patient H"),
        "st_technology": row.get("st_technology", "Spatial Transcriptomics"),
        "data_publication_date": row.get("data_publication_date"),
        "license": row.get("license"),
        "study_link": row.get("study_link"),
        "download_page_link1": row.get("download_page_link1"),
        "inter_spot_dist": float(row.get("inter_spot_dist", 200.0)),
        "spot_diameter": float(row.get("spot_diameter", 100.0)),
        "spots_under_tissue": int(row.get("spots_under_tissue")),
        "preservation_method": row.get("preservation_method"),
        "nb_genes": int(build_count_matrix(root, slide_id).shape[1]),
        "treatment_comment": row.get("treatment_comment"),
        "pixel_size_um_embedded": float(pixel_size),
        "pixel_size_um_estimated": float(row.get("pixel_size_um_estimated")),
        "magnification": row.get("magnification", "20x"),
        "fullres_px_width": int(row.get("fullres_px_width")),
        "fullres_px_height": int(row.get("fullres_px_height")),
        "tissue": row.get("tissue", "Breast"),
        "disease_comment": row.get("disease_comment", "HER2-positive breast cancer"),
        "subseries": slide_id,
        "hest_version_added": row.get("hest_version_added"),
    }
    payload = {k: sanitize_json_value(v) for k, v in payload.items()}
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return out_path


def copy_wsi(root: Path, slide_id: str, force: bool) -> Path:
    raw_img = root / "data" / "HER2ST_breast_cancer" / "raw" / "images" / "HE" / f"{slide_id}.jpg"
    out_img = root / "data" / "hest_data" / "wsis" / f"{slide_id}.tif"
    if out_img.exists() and not force:
        return out_img
    with Image.open(raw_img) as img:
        img.save(out_img, format="TIFF")
    return out_img


def write_common_genes(root: Path, force: bool) -> Path:
    out_path = root / "data" / "spatial_data" / "genesets" / "her2st_patient_h_H1_H2_H3.txt"
    if out_path.exists() and not force:
        return out_path

    cnts = [build_count_matrix(root, sid) for sid in SLIDES]
    common = list(cnts[0].columns.astype(str))
    for cnt in cnts[1:]:
        gene_set = set(cnt.columns.astype(str))
        common = [g for g in common if g in gene_set]
    out_path.write_text("\n".join(common), encoding="utf-8")
    return out_path


def write_summary(root: Path, force: bool) -> Path:
    out_path = root / "data" / "HER2ST_breast_cancer" / "her2st_patient_h_summary.csv"
    if out_path.exists() and not force:
        return out_path

    rows = []
    for sid in SLIDES:
        adata = build_adata(root, sid)
        rows.append(
            {
                "slide_id": sid,
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "label_available": bool(adata.uns["label_available"]),
                "n_labeled_spots": int((adata.obs["label"] != "unlabeled").sum()),
                "fullres_px_width": int(load_hest_row(root, sid)["fullres_px_width"]),
                "fullres_px_height": int(load_hest_row(root, sid)["fullres_px_height"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    ensure_dir(root / "data" / "spatial_data")
    ensure_dir(root / "data" / "spatial_data" / "genesets")
    ensure_dir(root / "data" / "hest_data" / "metadata")
    ensure_dir(root / "data" / "hest_data" / "wsis")
    ensure_dir(root / "data" / "HER2ST_breast_cancer")

    outputs = []
    for sid in SLIDES:
        outputs.append(write_h5ad(root, sid, force=args.force))
        outputs.append(write_metadata_json(root, sid, force=args.force))
        outputs.append(copy_wsi(root, sid, force=args.force))
    outputs.append(write_common_genes(root, force=args.force))
    outputs.append(write_summary(root, force=args.force))

    print("Prepared HER2ST patient H inputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
