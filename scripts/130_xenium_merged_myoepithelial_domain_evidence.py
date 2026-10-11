from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path(os.environ.get("MORPHO_VC_ROOT", r"E:\Morpho-FM")).resolve()
BASE_SCRIPT = ROOT / "scripts" / "129_xenium_krt14_krt5_domain_evidence_figure.py"
RESULT_DIR = ROOT / "results" / "st_superres" / "xenium_latent_clustering" / "domain_biology_enrichment"

MYO_LABELS = ("Myoepi_ACTA2+", "Myoepi_KRT15+")
MYO_DISPLAY = "Myoepithelial cells\n(ACTA2+ or KRT15+)"
MYO_COLOR = "#b65aa5"
OTHER_COLOR = "#d9d9d9"
N_CLUSTERS = 15


def load_base_module():
    spec = importlib.util.spec_from_file_location("xenium_krt14_krt5_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def domain_color(domain: int):
    return plt.get_cmap("tab20", N_CLUSTERS)(int(domain))


def myo_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    annotation = np.asarray(data["annotation"])
    return np.isin(annotation, MYO_LABELS)


def rank_domains(section: str, data: dict[str, np.ndarray]) -> pd.DataFrame:
    labels = np.asarray(data["labels"], dtype=int)
    myo = myo_mask(data)
    rows = []
    for domain in sorted(np.unique(labels)):
        target = labels == domain
        overlap = int(np.count_nonzero(target & myo))
        n_domain = int(np.count_nonzero(target))
        n_myo = int(np.count_nonzero(myo))
        precision = overlap / n_domain if n_domain else 0.0
        recall = overlap / n_myo if n_myo else 0.0
        union = int(np.count_nonzero(target | myo))
        iou = overlap / union if union else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "source_section": section,
                "section": BASE.SECTION_CONFIG[section]["label"],
                "domain": int(domain),
                "n_domain_cells": n_domain,
                "n_myoepithelial_cells": n_myo,
                "overlap_cells": overlap,
                "myoepithelial_fraction_in_domain": precision,
                "myoepithelial_recall": recall,
                "iou": iou,
                "f1": f1,
                "domain_color": matplotlib.colors.to_hex(domain_color(domain)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["f1", "iou", "myoepithelial_fraction_in_domain"], ascending=False
    )


def select_candidate_domains(ranked: pd.DataFrame) -> pd.DataFrame:
    candidates = ranked[
        (ranked["f1"] >= 0.18)
        & (ranked["myoepithelial_fraction_in_domain"] >= 0.15)
        & (ranked["myoepithelial_recall"] >= 0.05)
    ].copy()
    if candidates.empty:
        return ranked.head(1).copy()
    return candidates


def plot_merged_annotation(ax: plt.Axes, data: dict[str, np.ndarray], extent) -> None:
    coords = np.asarray(data["coords"])
    mask = myo_mask(data)
    ax.scatter(coords[:, 1], coords[:, 0], s=0.12, color=OTHER_COLOR, alpha=0.14, edgecolors="none", rasterized=True)
    ax.scatter(
        coords[mask, 1],
        coords[mask, 0],
        s=0.24,
        color=MYO_COLOR,
        alpha=0.94,
        edgecolors="none",
        rasterized=True,
    )
    BASE.format_axes(ax, extent)


def plot_matched_domain(ax: plt.Axes, data: dict[str, np.ndarray], domain: int, extent) -> None:
    coords = np.asarray(data["coords"])
    labels = np.asarray(data["labels"], dtype=int)
    target = labels == int(domain)
    ax.scatter(coords[:, 1], coords[:, 0], s=0.12, color=OTHER_COLOR, alpha=0.14, edgecolors="none", rasterized=True)
    ax.scatter(
        coords[target, 1],
        coords[target, 0],
        s=0.26,
        color=domain_color(domain),
        alpha=0.96,
        edgecolors="none",
        rasterized=True,
    )
    BASE.format_axes(ax, extent)


def save_figure(fig: plt.Figure, out_dirs: list[Path], prefix: str, manifest: pd.DataFrame) -> None:
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "tif"):
            out_path = out_dir / f"{prefix}.{ext}"
            kwargs = {"facecolor": "white"}
            if ext in {"png", "tif"}:
                kwargs["dpi"] = 600
            if ext == "tif":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(out_path, **kwargs)
        manifest.to_csv(out_dir / f"{prefix}_manifest.csv", index=False)


def figure_output_dirs() -> list[Path]:
    result_dir = RESULT_DIR / "KRT14_KRT5_merged_myoepithelial_domain_evidence"
    fig6_dir = (
        BASE.find_paper_figure6_dir()
        / "KRT14_KRT5_myoepithelial_DCiS"
        / "00_composite_evidence"
    )
    return [result_dir, fig6_dir]


def draw_rows(rows: pd.DataFrame, section_data: dict[str, dict[str, np.ndarray]], prefix: str) -> None:
    n_rows = len(rows)
    width_in = 170 / 25.4
    height_in = max(3.1, 1.95 * n_rows + 0.55)
    fig, axes = plt.subplots(n_rows, 4, figsize=(width_in, height_in), dpi=600, squeeze=False)
    fig.patch.set_facecolor("white")

    col_titles = [
        "Myoepithelial annotation",
        r"$\it{KRT14}$",
        r"$\it{KRT5}$",
        "Matched latent domain",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9.5, pad=4)

    for row_idx, row in rows.reset_index(drop=True).iterrows():
        section = str(row["source_section"])
        data = section_data[section]
        extent = data["extent"]
        domain = int(row["domain"])
        plot_merged_annotation(axes[row_idx, 0], data, extent)
        BASE.plot_expression(axes[row_idx, 1], data, "KRT14", extent)
        BASE.plot_expression(axes[row_idx, 2], data, "KRT5", extent)
        plot_matched_domain(axes[row_idx, 3], data, domain, extent)

        axes[row_idx, 0].text(
            -0.08,
            0.5,
            f"{row['section']} D{domain}",
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9.5,
        )
        axes[row_idx, 3].text(
            0.02,
            0.04,
            f"Domain {domain}\nF1={row['f1']:.2f}",
            transform=axes[row_idx, 3].transAxes,
            fontsize=7.2,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    bottom = 0.25 if n_rows <= 2 else 0.14
    plt.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=bottom, wspace=0.035, hspace=0.08)

    ann_handles = [
        mpatches.Patch(color=MYO_COLOR, label=MYO_DISPLAY),
        mpatches.Patch(color=OTHER_COLOR, label="Other cells/domains"),
    ]
    fig.legend(
        handles=ann_handles,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.045 if n_rows <= 2 else 0.035),
        ncol=1,
        frameon=False,
        fontsize=6.6,
        handlelength=0.9,
        columnspacing=0.9,
        labelspacing=0.35,
    )

    domain_handles = []
    for _, row in rows.reset_index(drop=True).iterrows():
        domain = int(row["domain"])
        domain_handles.append(
            mpatches.Patch(color=domain_color(domain), label=f"{row['section']} Domain {domain}")
        )
    fig.legend(
        handles=domain_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.06 if n_rows <= 2 else 0.04),
        ncol=min(3, max(1, len(domain_handles))),
        frameon=False,
        fontsize=6.6,
        handlelength=0.9,
        columnspacing=0.9,
        labelspacing=0.35,
    )

    cax = fig.add_axes([0.85, 0.09 if n_rows <= 2 else 0.055, 0.11, 0.014])
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="viridis")
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7, length=0, pad=1)
    cbar.outline.set_linewidth(0.4)
    cax.set_title("Expression", fontsize=8, pad=2)

    save_figure(fig, figure_output_dirs(), prefix, rows)
    plt.close(fig)


def main() -> None:
    section_data = {section: BASE.load_section(section) for section in BASE.SECTION_CONFIG}
    ranked_all = pd.concat([rank_domains(section, data) for section, data in section_data.items()], ignore_index=True)

    primary_rows = ranked_all.sort_values(["source_section", "f1"], ascending=[True, False]).groupby(
        "source_section", as_index=False
    ).head(1)
    primary_rows = primary_rows.sort_values("source_section").reset_index(drop=True)

    candidate_rows = pd.concat(
        [select_candidate_domains(rank_domains(section, section_data[section])) for section in BASE.SECTION_CONFIG],
        ignore_index=True,
    )
    candidate_rows = candidate_rows.sort_values(["source_section", "f1"], ascending=[True, False]).reset_index(drop=True)

    draw_rows(
        primary_rows,
        section_data,
        "KRT14_KRT5_merged_myoepithelial_primary_domain_evidence",
    )
    draw_rows(
        candidate_rows,
        section_data,
        "KRT14_KRT5_merged_myoepithelial_matched_candidate_domains",
    )

    out_dirs = figure_output_dirs()
    for out_dir in out_dirs:
        ranked_all.to_csv(out_dir / "KRT14_KRT5_merged_myoepithelial_domain_ranking.csv", index=False)

    print("[primary domains]")
    print(primary_rows.to_string(index=False))
    print("\n[matched candidate domains]")
    print(candidate_rows.to_string(index=False))
    print(f"\n[saved] {out_dirs[-1]}")


if __name__ == "__main__":
    main()
