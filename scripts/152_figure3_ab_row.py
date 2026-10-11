from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE6_OUTPUT = ROOT / "results" / "phase6_structure_preservation"
PROJECT_ROOT = Path.home() / "Desktop" / "\u4eba\u751f\u7684\u7b2c4\u7bc7\u8bba\u6587"
# Point at the generator's own output rather than at a copy under Figure/Figure_3: that
# copy is from 2026-04-21 and still has the six-method panel a, so defaulting to it
# silently recomposes the row with the stale panel.
DEFAULT_PANEL_A = (
    ROOT / "figures_nature_kidney" / "panel_a_violin" / "panel_a_kidney_multi_single_violin.pdf"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figure" / "Figure_3"
ARIAL_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

FIGURE_WIDTH_MM = 170.0
FIGURE_HEIGHT_MM = 55.0
PANEL_GAP_MM = 0.5
MM_TO_INCH = 1.0 / 25.4
PT_PER_MM = 72.0 / 25.4

SECTION_ORDER = [
    ("Prostate", "INT25"),
    ("Prostate", "INT26"),
    ("Prostate", "INT27"),
    ("Prostate", "INT28"),
    ("Kidney", "INT13"),
    ("Kidney", "INT14"),
    ("Kidney", "INT15"),
    ("Kidney", "INT17"),
    ("Kidney", "INT18"),
    ("Kidney", "INT19"),
    ("Kidney", "INT21"),
    ("Kidney", "INT24"),
]

METHOD_COLORS = {"Morpho-FM": "#E64B35", "iStar": "#4DBBD5"}
COHORT_COLORS = {"Prostate": "#2C6EAE", "Kidney": "#D97706"}
ANCHOR_PROGRAMMES = {
    "HALLMARK_ANDROGEN_RESPONSE": ("Androgen response", COHORT_COLORS["Prostate"]),
    "HALLMARK_HYPOXIA": ("Hypoxia", COHORT_COLORS["Kidney"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 170-mm Figure 3 a-b row from the existing Fig. 3a and Phase 6 programme results."
    )
    parser.add_argument("--phase6-output", type=Path, default=DEFAULT_PHASE6_OUTPUT)
    parser.add_argument("--panel-a", type=Path, default=DEFAULT_PANEL_A)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_programme_data(phase6_output: Path) -> dict[str, object]:
    programme_spec = json.loads(
        (phase6_output / "gate6_1" / "programmes.json").read_text(encoding="utf-8")
    )
    programme_names = [item["name"] for item in programme_spec["programmes"]]
    metrics = pd.read_csv(phase6_output / "full" / "programme_metrics.csv")
    keys = ["cohort", "train", "test", "task_id", "programme"]
    wide = metrics.pivot(index=keys, columns="method", values="concordance").reset_index()
    wide["score_advantage"] = wide["Morpho-FM"] - wide["iStar"]
    section = (
        wide.groupby(["cohort", "test", "programme"], as_index=False)[
            ["Morpho-FM", "iStar", "score_advantage"]
        ]
        .median()
        .rename(columns={"Morpho-FM": "score_morpho", "iStar": "score_istar"})
    )
    tissue = (
        section.groupby(["cohort", "programme"], as_index=False)[
            ["score_morpho", "score_istar", "score_advantage"]
        ]
        .median()
    )

    if len(programme_names) != 50 or len(set(programme_names)) != 50:
        raise RuntimeError("Expected 50 unique Hallmark programmes")
    if len(section) != 600:
        raise RuntimeError(f"Expected 600 section-programme rows, observed {len(section)}")
    if not np.isfinite(
        section[["score_morpho", "score_istar", "score_advantage"]]
    ).all().all():
        raise RuntimeError("Non-finite programme-score result")
    if not (section["score_advantage"] > 0.0).all():
        raise RuntimeError("Expected all 600 paired programme-score advantages to be positive")

    pooled_order = (
        tissue.groupby("programme")["score_advantage"]
        .median()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    section_index = pd.MultiIndex.from_tuples(
        SECTION_ORDER, names=["cohort", "test"]
    )

    def absolute_matrix(column: str) -> np.ndarray:
        matrix = (
            section.pivot(
                index=["cohort", "test"], columns="programme", values=column
            )
            .reindex(index=section_index, columns=pooled_order)
            .to_numpy(dtype=float)
        )
        if matrix.shape != (12, 50) or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid {column} matrix: {matrix.shape}")
        return matrix

    section_frame = (
        pd.read_csv(phase6_output / "full" / "section_summary.csv")
        .set_index(["cohort", "test"])
        .reindex(section_index)
    )
    if section_frame[["programme_morpho", "programme_istar"]].isna().any().any():
        raise RuntimeError("Missing section-level programme summary")

    return {
        "pooled_order": pooled_order,
        "morpho_matrix": absolute_matrix("score_morpho"),
        "istar_matrix": absolute_matrix("score_istar"),
        "section_frame": section_frame,
    }


def render_programme_panel(data: dict[str, object], width_mm: float) -> bytes:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    pooled_order = data["pooled_order"]
    morpho_matrix = data["morpho_matrix"]
    istar_matrix = data["istar_matrix"]
    section_frame = data["section_frame"]
    anchor_indices = [pooled_order.index(name) for name in ANCHOR_PROGRAMMES]

    fig = plt.figure(
        figsize=(width_mm * MM_TO_INCH, FIGURE_HEIGHT_MM * MM_TO_INCH),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=[24.0, 2.5, 24.0, 4.0, 1.0, 8.0, 18.0],
        left=0.11,
        right=0.972,
        top=0.86,
        bottom=0.255,
        wspace=0.0,
    )

    heat_norm = Normalize(vmin=-0.55, vmax=1.0)
    heat_image = None
    for column, (method, matrix) in enumerate(
        (("Morpho-FM", morpho_matrix), ("iStar", istar_matrix))
    ):
        ax = fig.add_subplot(grid[0, column * 2])
        heat_image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=plt.get_cmap("magma_r"),
            norm=heat_norm,
        )
        ax.axhline(3.5, color="white", lw=1.5)
        ax.axhline(3.5, color="#404040", lw=0.4)
        ax.set_title(method, fontsize=7.4, fontweight="normal", pad=3)
        ax.set_xticks(
            anchor_indices,
            [ANCHOR_PROGRAMMES[name][0] for name in ANCHOR_PROGRAMMES],
            rotation=30,
            ha="right",
            fontsize=6.2,
        )
        for label, name in zip(ax.get_xticklabels(), ANCHOR_PROGRAMMES, strict=True):
            label.set_color(ANCHOR_PROGRAMMES[name][1])
            label.set_fontweight("normal")
        for name, index in zip(ANCHOR_PROGRAMMES, anchor_indices, strict=True):
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -0.5),
                    1.0,
                    12.0,
                    fill=False,
                    edgecolor=ANCHOR_PROGRAMMES[name][1],
                    linewidth=0.7,
                    clip_on=False,
                )
            )
        if column == 0:
            ax.set_yticks(
                np.arange(12),
                [slide for _, slide in SECTION_ORDER],
                fontsize=6.8,
            )
            for label, (cohort, _) in zip(
                ax.get_yticklabels(), SECTION_ORDER, strict=True
            ):
                label.set_color(COHORT_COLORS[cohort])
                label.set_fontweight("normal")
            ax.text(
                -0.36,
                0.83,
                "Prostate",
                transform=ax.transAxes,
                rotation=90,
                color=COHORT_COLORS["Prostate"],
                fontsize=6.6,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                -0.36,
                0.31,
                "Kidney",
                transform=ax.transAxes,
                rotation=90,
                color=COHORT_COLORS["Kidney"],
                fontsize=6.6,
                fontweight="bold",
                ha="center",
                va="center",
            )
        else:
            ax.set_yticks([])
        ax.tick_params(axis="both", length=0, pad=1.4)
        ax.spines[:].set_visible(False)

    cax = fig.add_subplot(grid[0, 4])
    colorbar = fig.colorbar(heat_image, cax=cax)
    cbar_position = cax.get_position()
    cbar_shrink = 0.88
    cax.set_position(
        [
            cbar_position.x0,
            cbar_position.y0 + cbar_position.height * (1.0 - cbar_shrink) / 2.0,
            cbar_position.width,
            cbar_position.height * cbar_shrink,
        ]
    )
    colorbar.set_ticks(np.arange(-0.4, 1.01, 0.2))
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(
        labelsize=6.3,
        length=2.0,
        pad=1.5,
        direction="out",
        labelleft=False,
        labelright=True,
    )
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.set_label("PCC", fontsize=6.8, rotation=90, labelpad=2)

    section_ax = fig.add_subplot(grid[0, 6])
    y = np.arange(12)
    section_morpho = section_frame["programme_morpho"].to_numpy(dtype=float)
    section_istar = section_frame["programme_istar"].to_numpy(dtype=float)
    section_ax.hlines(y, section_istar, section_morpho, color="#A6A6A6", lw=1.0)
    section_ax.scatter(
        section_istar,
        y,
        s=12,
        marker="s",
        color=METHOD_COLORS["iStar"],
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    section_ax.scatter(
        section_morpho,
        y,
        s=16,
        marker="o",
        color=METHOD_COLORS["Morpho-FM"],
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )
    section_ax.set_xlim(0.0, 1.02)
    section_ax.set_ylim(11.6, -0.6)
    section_ax.set_yticks([])
    section_ax.set_xticks([0.0, 0.5, 1.0])
    section_ax.set_xlabel("Median PCC", fontsize=7.0, labelpad=1.5)
    section_ax.set_title(
        "Held-out sections", fontsize=7.2, fontweight="normal", pad=3, x=0.46
    )
    section_ax.grid(axis="x", color="#E3E3E3", lw=0.4)
    section_ax.spines[["top", "right", "left"]].set_visible(False)
    section_ax.tick_params(axis="x", labelsize=6.8, length=1.5)
    section_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=METHOD_COLORS["Morpho-FM"],
                label="Morpho-FM",
                markersize=3.3,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                color=METHOD_COLORS["iStar"],
                label="iStar",
                markersize=3.0,
            ),
        ],
        frameon=False,
        fontsize=6.4,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.31, -0.25),
        handletextpad=0.25,
        columnspacing=0.45,
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def compose_row(panel_a_path: Path, panel_b_pdf: bytes, output_pdf: Path) -> None:
    if not ARIAL_BOLD.exists():
        raise FileNotFoundError(ARIAL_BOLD)
    panel_a_doc = fitz.open(panel_a_path)
    panel_b_doc = fitz.open(stream=panel_b_pdf, filetype="pdf")
    panel_a_page = panel_a_doc[0]

    panel_a_width_mm = FIGURE_HEIGHT_MM * (
        panel_a_page.rect.width / panel_a_page.rect.height
    )
    panel_b_width_mm = FIGURE_WIDTH_MM - PANEL_GAP_MM - panel_a_width_mm
    if panel_b_width_mm <= 0:
        raise RuntimeError("Panel widths exceed the 170-mm figure width")

    document = fitz.open()
    page = document.new_page(
        width=FIGURE_WIDTH_MM * PT_PER_MM,
        height=FIGURE_HEIGHT_MM * PT_PER_MM,
    )
    panel_a_rect = fitz.Rect(
        0.0,
        0.0,
        panel_a_width_mm * PT_PER_MM,
        FIGURE_HEIGHT_MM * PT_PER_MM,
    )
    panel_b_left_mm = panel_a_width_mm + PANEL_GAP_MM
    panel_b_rect = fitz.Rect(
        panel_b_left_mm * PT_PER_MM,
        0.0,
        FIGURE_WIDTH_MM * PT_PER_MM,
        FIGURE_HEIGHT_MM * PT_PER_MM,
    )
    page.show_pdf_page(panel_a_rect, panel_a_doc, 0, keep_proportion=True)
    page.show_pdf_page(panel_b_rect, panel_b_doc, 0, keep_proportion=True)
    page.insert_font(fontname="ArialBold", fontfile=str(ARIAL_BOLD))
    page.insert_text(
        (0.8 * PT_PER_MM, 5.5 * PT_PER_MM),
        "a",
        fontsize=12.0,
        fontname="ArialBold",
        color=(0, 0, 0),
    )
    page.insert_text(
        ((panel_b_left_mm + 0.8) * PT_PER_MM, 5.5 * PT_PER_MM),
        "b",
        fontsize=12.0,
        fontname="ArialBold",
        color=(0, 0, 0),
    )
    document.set_metadata(
        {
            "title": "Figure 3 panels a-b",
            "creator": "152_figure3_ab_row.py",
            "producer": "PyMuPDF and Matplotlib",
        }
    )
    if output_pdf.exists():
        output_pdf.unlink()
    document.save(output_pdf, garbage=4, deflate=True)
    document.close()
    panel_a_doc.close()
    panel_b_doc.close()


def render_png(pdf_path: Path, png_path: Path) -> tuple[int, int]:
    document = fitz.open(pdf_path)
    pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600.0 / 72.0, 600.0 / 72.0), alpha=False)
    pixmap.save(png_path)
    size = (pixmap.width, pixmap.height)
    document.close()
    return size


def validate_output(pdf_path: Path, png_size: tuple[int, int]) -> None:
    document = fitz.open(pdf_path)
    page = document[0]
    width_mm = page.rect.width / PT_PER_MM
    height_mm = page.rect.height / PT_PER_MM
    text = page.get_text()
    spans = [
        (
            span["text"].strip(),
            float(span["size"]),
            fitz.Rect(span["bbox"]),
            span["font"],
        )
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"].strip()
    ]
    outside_spans = [
        (label, bounds)
        for label, _, bounds, _ in spans
        if bounds.x0 < -0.1
        or bounds.y0 < -0.1
        or bounds.x1 > page.rect.width + 0.1
        or bounds.y1 > page.rect.height + 0.1
    ]
    font_names = {font[3].split("+")[-1] for font in page.get_fonts(full=True)}
    panel_letters = [span for span in spans if span[0] in {"a", "b"}]
    document.close()

    if abs(width_mm - FIGURE_WIDTH_MM) > 0.01:
        raise RuntimeError(f"Unexpected PDF width: {width_mm:.3f} mm")
    if abs(height_mm - FIGURE_HEIGHT_MM) > 0.01:
        raise RuntimeError(f"Unexpected PDF height: {height_mm:.3f} mm")
    if outside_spans:
        raise RuntimeError(f"Text extends outside the PDF page: {outside_spans}")
    if any("Arial" not in name for name in font_names):
        raise RuntimeError(f"Non-Arial font embedded in PDF: {sorted(font_names)}")
    if len(panel_letters) != 2 or any(
        abs(size - 12.0) > 0.05 or "Arial" not in font or "Bold" not in font
        for _, size, _, font in panel_letters
    ):
        raise RuntimeError(f"Panel letters are not 12-pt Arial Bold: {panel_letters}")
    if min(size for _, size, _, _ in spans) < 6.15:
        raise RuntimeError("Figure contains text smaller than 6.2 pt")
    for forbidden in (
        "Initialization",
        "Direct programme-level agreement across held-out sections",
        "Morpho-FM achieved higher programme-score concordance",
        "Median gain",
    ):
        if forbidden in text:
            raise RuntimeError(f"Removed annotation remains in PDF: {forbidden}")
    for required in (
        "Morpho-FM",
        "iStar",
        "Held-out sections",
        "Hypoxia",
        "Androgen response",
        "PCC",
    ):
        if required not in text:
            raise RuntimeError(f"Required annotation missing from PDF: {required}")
    expected_width = round(FIGURE_WIDTH_MM / 25.4 * 600)
    expected_height = round(FIGURE_HEIGHT_MM / 25.4 * 600)
    if abs(png_size[0] - expected_width) > 1 or abs(png_size[1] - expected_height) > 1:
        raise RuntimeError(f"Unexpected PNG dimensions: {png_size}")


def main() -> None:
    args = parse_args()
    phase6_output = args.phase6_output.expanduser().resolve()
    panel_a = args.panel_a.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not panel_a.exists():
        raise FileNotFoundError(panel_a)
    output_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(panel_a) as panel_a_doc:
        panel_a_width_mm = FIGURE_HEIGHT_MM * (
            panel_a_doc[0].rect.width / panel_a_doc[0].rect.height
        )
    panel_b_width_mm = FIGURE_WIDTH_MM - PANEL_GAP_MM - panel_a_width_mm
    programme_data = build_programme_data(phase6_output)
    panel_b_pdf = render_programme_panel(programme_data, panel_b_width_mm)

    output_pdf = output_dir / "fig3_ab_row.pdf"
    output_png = output_dir / "fig3_ab_row.png"
    compose_row(panel_a, panel_b_pdf, output_pdf)
    png_size = render_png(output_pdf, output_png)
    validate_output(output_pdf, png_size)
    print(f"[saved] {output_pdf}")
    print(f"[saved] {output_png}")
    print(
        f"[verified] {FIGURE_WIDTH_MM:.1f} x {FIGURE_HEIGHT_MM:.1f} mm; "
        f"panel a {panel_a_width_mm:.1f} mm; panel b {panel_b_width_mm:.1f} mm; PNG {png_size[0]} x {png_size[1]} px"
    )


if __name__ == "__main__":
    main()
