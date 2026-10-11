from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = (
    Path.home()
    / "Desktop"
    / "人生的第4篇论文"
    / "Figure"
    / "Figure_6"
    / "HER2ST_patient_H"
)
SECTION_FOLDERS = {
    "H1": "16_loso_H1_transfer_summary",
    "H2": "16_loso_H2_transfer_summary",
    "H3": "16_loso_H3_transfer_summary",
}
FILLED_DOMAINS = {
    "H1": "0_1_8",
    "H2": "3",
    "H3": "0_9",
}
FOLDS = {
    "H1": "loso_train_H2_H3_to_H1",
    "H2": "loso_train_H1_H3_to_H2",
    "H3": "loso_train_H1_H2_to_H3",
}

Image.MAX_IMAGE_PIXELS = None


def load_display_he(path: Path, max_dim: int = 4800) -> np.ndarray:
    img = Image.open(path)
    orig_w, orig_h = img.size
    scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))),
            resample=Image.Resampling.LANCZOS,
        )
    return np.asarray(img)


def label_rgba(label_map: np.ndarray, out_hw: tuple[int, int], n_clusters: int, alpha: float = 0.72) -> np.ndarray:
    encoded = np.full(label_map.shape, -1, dtype=np.int16)
    finite = np.isfinite(label_map)
    encoded[finite] = label_map[finite].astype(np.int16)
    img = Image.fromarray(encoded).resize((out_hw[1], out_hw[0]), resample=Image.Resampling.NEAREST)
    up = np.asarray(img)
    cmap = plt.get_cmap("tab10", n_clusters)
    rgba = np.zeros((*out_hw, 4), dtype=np.float32)
    valid = up >= 0
    rgba[valid] = cmap(up[valid].astype(int) % n_clusters)
    rgba[valid, 3] = alpha
    return rgba


def render(section: str, dpi: int = 900) -> None:
    folder = PAPER_ROOT / SECTION_FOLDERS[section]
    suffix = FILLED_DOMAINS[section]
    meta_path = folder / f"{section}_dense_latent_domain_map_display_filled_domains_{suffix}_meta.json"
    map_path = folder / f"{section}_dense_latent_domain_map_display_filled_domains_{suffix}.npy"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_map = np.load(map_path)
    n_clusters = int(meta.get("n_clusters", int(np.nanmax(label_map)) + 1))
    he = load_display_he(ROOT / "data" / "superres_cache" / section / "he.jpg")
    train_label = FOLDS[section].removeprefix("loso_train_").split("_to_")[0].replace("_", "+")

    fig, ax = plt.subplots(figsize=(6.69, 4.15))
    ax.imshow(he)
    ax.imshow(label_rgba(label_map, he.shape[:2], n_clusters=n_clusters))
    ax.set_title(f"{section} dense latent candidate domains (train {train_label})", fontsize=10, pad=3)
    ax.axis("off")

    domains = sorted(int(x) for x in np.unique(label_map[np.isfinite(label_map)]))
    cmap = plt.get_cmap("tab10", n_clusters)
    handles = [mpatches.Patch(color=cmap(i), label=f"Domain {i}") for i in domains]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.005, right=0.86, top=0.94, bottom=0.005)

    stem = f"{section}_loso_dense_latent_cluster_summary_filled_domains_{suffix}"
    fig.savefig(folder / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.tif", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(folder / f"{stem}.pdf")


def main() -> None:
    for section in ["H1", "H2", "H3"]:
        render(section)


if __name__ == "__main__":
    main()
