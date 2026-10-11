from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE_151 = ROOT / "scripts" / "151_phase6_gene_identified_figure.py"
DEFAULT_PHASE6_OUTPUT = ROOT / "results" / "phase6_structure_preservation"
DEFAULT_GENERALIZATION = ROOT / "figures_nature_kidney" / "panel_b_generalization"
PROJECT_ROOT = Path.home() / "Desktop" / "\u4eba\u751f\u7684\u7b2c4\u7bc7\u8bba\u6587"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Figure" / "Figure_3"

FIGURE_WIDTH_MM = 170.0
FIGURE_HEIGHT_MM = 90.0
MM_TO_INCH = 1.0 / 25.4
PT_PER_MM = 72.0 / 25.4

SERIES_ORDER = ["INT21_holdout", "GSM6415706_GLMF1", "GSM6415706_GLMF2"]
SERIES_LABELS = {
    "INT21_holdout": "Internal holdout",
    "GSM6415706_GLMF1": "External slide 1",
    "GSM6415706_GLMF2": "External slide 2",
}
SERIES_COLORS = {
    "INT21_holdout": "#777777",
    "GSM6415706_GLMF1": "#3C5488",
    "GSM6415706_GLMF2": "#E64B35",
}
COHORT_COLORS = {"Prostate": "#2C6EAE", "Kidney": "#D97706"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 170-mm Figure 3 c-e block with editable Arial text."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase6-output", type=Path, default=DEFAULT_PHASE6_OUTPUT)
    parser.add_argument("--generalization", type=Path, default=DEFAULT_GENERALIZATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_source_module():
    spec = importlib.util.spec_from_file_location("phase6_gene_figure_151", SOURCE_151)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_151)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_programme_examples(root: Path, phase6_output: Path) -> tuple[object, list[dict]]:
    source = load_source_module()
    base = source.load_phase6_module()
    programmes = base.load_programmes(phase6_output)
    tasks = base.build_tasks(root)
    examples = [
        source.load_example(base, root, phase6_output, programmes, tasks, item)
        for item in source.choose_examples(phase6_output)
    ]
    if [(item["cohort"], item["train"], item["test"]) for item in examples] != [
        ("Prostate", "INT25", "INT28"),
        ("Kidney", "INT19", "INT24"),
    ]:
        raise RuntimeError("Unexpected programme examples")
    return source, examples


def load_generalization_data(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = pd.read_csv(directory / "panel_b_generalization_fraction_threshold.csv")
    hvg = pd.read_csv(directory / "panel_b_generalization_hvg_denseK.csv")
    if set(threshold["slide_id"]) != set(SERIES_ORDER):
        raise RuntimeError("Unexpected threshold series")
    if set(hvg["slide_id"]) != set(SERIES_ORDER):
        raise RuntimeError("Unexpected HVG series")
    return threshold, hvg


def marker_area(k: int, maximum: int) -> float:
    fraction = min(max(k / max(maximum, 1), 0.0), 1.0)
    return 8.0 + 52.0 * fraction**0.7


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#E6E6E6", linewidth=0.45, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#BDBDBD")
    axis.spines[["left", "bottom"]].set_linewidth(0.7)
    axis.tick_params(axis="both", labelsize=6.4, width=0.6, length=2.2, pad=1.5)


def plot_threshold(axis: plt.Axes, threshold: pd.DataFrame) -> None:
    for series in SERIES_ORDER:
        selected = threshold[threshold["slide_id"].eq(series)].sort_values("Threshold")
        axis.plot(
            selected["Threshold"],
            selected["Fraction"],
            color=SERIES_COLORS[series],
            linewidth=1.35,
        )
    axis.set_xlim(-0.2, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_xlabel("Threshold (Pearson r)", fontsize=7.0, labelpad=1.5)
    axis.set_ylabel("Fraction of genes", fontsize=6.6, labelpad=2.0)
    axis.set_title(
        "Fraction of genes with Pearson r >= threshold",
        fontsize=7.4,
        fontweight="normal",
        pad=3.0,
    )
    style_axis(axis)


def plot_hvg(axis: plt.Axes, hvg: pd.DataFrame) -> None:
    for series in SERIES_ORDER:
        selected = hvg[hvg["slide_id"].eq(series)].sort_values("K")
        maximum = int(selected["Kmax"].iloc[0])
        color = SERIES_COLORS[series]
        sizes = [marker_area(int(value), maximum) for value in selected["K"]]
        axis.plot(
            selected["Recall@K"],
            selected["MeanPearson_on_HVGpred"],
            color=color,
            linewidth=1.35,
            alpha=0.88,
        )
        axis.scatter(
            selected["Recall@K"],
            selected["MeanPearson_on_HVGpred"],
            s=sizes,
            color=color,
            alpha=0.28,
            edgecolor="none",
        )
        highlights = selected[selected["K"].isin([50, 200, 500, 1000, 2000])]
        axis.scatter(
            highlights["Recall@K"],
            highlights["MeanPearson_on_HVGpred"],
            s=[marker_area(int(value), maximum) for value in highlights["K"]],
            color=color,
            alpha=0.95,
            edgecolor="#222222",
            linewidth=0.3,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.arange(0.0, 1.01, 0.2))
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_xlabel("HVG overlap (Recall@K)", fontsize=7.0, labelpad=1.5)
    axis.set_ylabel(
        "Mean Pearson r on predicted\ntop-K HVGs",
        fontsize=6.6,
        labelpad=2.0,
        linespacing=0.95,
    )
    axis.set_title(
        "HVG recovery vs prediction quality (dense K)",
        fontsize=7.4,
        fontweight="normal",
        pad=3.0,
    )
    style_axis(axis)

    line_handles = [
        Line2D([0], [0], color=SERIES_COLORS[item], lw=1.6, label=SERIES_LABELS[item])
        for item in SERIES_ORDER
    ]
    series_legend = axis.legend(
        handles=line_handles,
        frameon=False,
        fontsize=5.9,
        loc="upper right",
        borderaxespad=0.15,
        handlelength=2.0,
        labelspacing=0.25,
    )
    maximum = int(hvg["Kmax"].max())
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#B3B3B3",
            markeredgecolor="#333333",
            markeredgewidth=0.3,
            markersize=np.sqrt(marker_area(value, maximum)),
            label=f"K={value}",
        )
        for value in [50, 200, 500, 1000, 2000]
    ]
    axis.legend(
        handles=size_handles,
        frameon=False,
        fontsize=5.8,
        loc="lower right",
        borderaxespad=0.15,
        handletextpad=0.35,
        labelspacing=0.18,
    )
    axis.add_artist(series_legend)


def plot_programme_row(
    figure: plt.Figure,
    parent,
    source,
    root: Path,
    example: dict,
) -> tuple[float, float, float]:
    grid = parent.subgridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.048],
        wspace=0.075,
    )
    row_position = parent.get_position(figure)
    programme = source.short_programme_name(example["programme"]).capitalize()
    title_text = f"{example['cohort']} | {programme}"
    title_kwargs = dict(
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=COHORT_COLORS[example["cohort"]],
        fontweight="normal",
    )

    image, coordinates = source.load_h5ad_background(root, example["test"])
    if len(coordinates) != len(example["scores"]["Truth"]):
        raise RuntimeError(f"Coordinate mismatch for {example['test']}")
    combined = np.concatenate(
        [example["scores"][name] for name in ("Truth", "Morpho-FM", "iStar")]
    )
    low, high = np.quantile(combined[np.isfinite(combined)], [0.02, 0.98])
    if high <= low:
        high = low + 1.0
    normalizer = Normalize(float(low), float(high))
    map_axes = []
    mapped = None
    for column, method in enumerate(("Truth", "Morpho-FM", "iStar")):
        axis = figure.add_subplot(grid[0, column])
        map_axes.append(axis)
        axis.imshow(image, interpolation="bilinear")
        mapped = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=example["scores"][method],
            cmap="viridis",
            norm=normalizer,
            s=2.4,
            alpha=0.80,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_anchor("C")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
        axis.set_title(
            "Measured" if method == "Truth" else method,
            fontsize=6.8,
            fontweight="normal",
            pad=1.5,
        )
        if method != "Truth":
            axis.text(
                0.03,
                0.04,
                f"PCC = {example['score_concordance'][method]:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=6,
                color="black",
                zorder=5,
                bbox=dict(
                    facecolor="white",
                    alpha=0.78,
                    edgecolor="none",
                    boxstyle="square,pad=0.12",
                ),
            )

    color_axis = figure.add_subplot(grid[0, 3])
    colorbar = figure.colorbar(mapped, cax=color_axis)
    colorbar.set_ticks(np.linspace(float(low), float(high), 3))
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(labelsize=5.5, length=1.6, width=0.5, pad=1.0)
    colorbar.outline.set_linewidth(0.55)
    figure.canvas.draw()
    map_position = map_axes[0].get_position()
    color_position = color_axis.get_position()
    color_axis.set_position(
        [color_position.x0, map_position.y0, color_position.width, map_position.height]
    )
    title_x = (
        map_axes[0].get_position().x0
        + map_axes[-1].get_position().x1
    ) / 2
    figure.text(
        title_x,
        row_position.y1 + 0.003,
        title_text,
        **title_kwargs,
    )
    return row_position.x0, color_axis.get_position().x1, map_position.height


def build_figure(
    root: Path,
    phase6_output: Path,
    generalization: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    arial = Path(r"C:\Windows\Fonts\arial.ttf")
    arial_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if not arial.exists() or not arial_bold.exists():
        raise FileNotFoundError("Arial font files are required")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    source, examples = load_programme_examples(root, phase6_output)
    threshold, hvg = load_generalization_data(generalization)
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, FIGURE_HEIGHT_MM * MM_TO_INCH),
        facecolor="white",
    )
    outer = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.65, 0.28, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.025,
        right=0.985,
        top=0.955,
        bottom=0.075,
        wspace=0.0,
        hspace=0.25,
    )
    for row in (0, 1):
        c_row = outer[row, 0].get_position(figure)
        comparison_row = outer[row, 2].get_position(figure)
        if not np.allclose(
            [c_row.y0, c_row.y1],
            [comparison_row.y0, comparison_row.y1],
            atol=1e-12,
        ):
            raise RuntimeError(f"Panel row {row + 1} is not height-aligned")
    top_left, top_right, top_map_height = plot_programme_row(
        figure, outer[0, 0], source, root, examples[0]
    )
    bottom_left, bottom_right, bottom_map_height = plot_programme_row(
        figure, outer[1, 0], source, root, examples[1]
    )
    c_left = min(top_left, bottom_left)
    c_right = max(top_right, bottom_right)
    axis_d = figure.add_subplot(outer[0, 2])
    axis_e = figure.add_subplot(outer[1, 2])
    plot_threshold(axis_d, threshold)
    plot_hvg(axis_e, hvg)
    figure.canvas.draw()
    for axis, target_height in ((axis_d, top_map_height), (axis_e, bottom_map_height)):
        position = axis.get_position()
        target_height_inches = target_height * figure.get_figheight()
        axis_width_inches = position.width * figure.get_figwidth()
        axis.set_box_aspect(target_height_inches / axis_width_inches)
        axis.set_anchor("C")
    figure.canvas.draw()
    for axis, target_height in ((axis_d, top_map_height), (axis_e, bottom_map_height)):
        if abs(axis.get_position().height - target_height) > 5e-4:
            raise RuntimeError("Programme map and comparison panel heights are not aligned")

    figure.text(c_left - 0.018, 0.982, "c", fontsize=12.0, fontweight="bold", va="top")
    for letter, axis in (("d", axis_d), ("e", axis_e)):
        position = axis.get_position()
        figure.text(
            c_right - 0.016,
            position.y1 + (0.055 if letter == "e" else 0.050),
            letter,
            fontsize=12.0,
            fontweight="bold",
            va="top",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig3_cde_block.pdf"
    png_path = output_dir / "fig3_cde_block.png"
    figure.savefig(pdf_path, format="pdf", facecolor="white")
    figure.savefig(png_path, format="png", dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def validate_output(pdf_path: Path, png_path: Path) -> None:
    document = fitz.open(pdf_path)
    page = document[0]
    width_mm = page.rect.width / PT_PER_MM
    height_mm = page.rect.height / PT_PER_MM
    text_dict = page.get_text("dict")
    spans = [
        (
            span["text"].strip(),
            float(span["size"]),
            span["font"],
            fitz.Rect(span["bbox"]),
        )
        for block in text_dict["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"].strip()
    ]
    fonts = {item[3].split("+")[-1] for item in page.get_fonts(full=True)}
    outside = [
        (label, bounds)
        for label, _, _, bounds in spans
        if bounds.x0 < -0.1
        or bounds.y0 < -0.1
        or bounds.x1 > page.rect.width + 0.1
        or bounds.y1 > page.rect.height + 0.1
    ]
    image_dpi = []
    for image in page.get_images(full=True):
        xref = image[0]
        width, height = image[2], image[3]
        for rectangle in page.get_image_rects(xref):
            image_dpi.append(
                min(width / (rectangle.width / 72.0), height / (rectangle.height / 72.0))
            )
    panel_letters = [item for item in spans if item[0] in {"c", "d", "e"}]
    extracted = page.get_text()
    document.close()

    if abs(width_mm - FIGURE_WIDTH_MM) > 0.01 or abs(height_mm - FIGURE_HEIGHT_MM) > 0.01:
        raise RuntimeError(f"Unexpected page size: {width_mm:.3f} x {height_mm:.3f} mm")
    if outside:
        raise RuntimeError(f"Text extends outside the page: {outside}")
    if any("Arial" not in name for name in fonts):
        raise RuntimeError(f"Non-Arial font embedded: {sorted(fonts)}")
    if len(panel_letters) != 3 or any(
        abs(size - 12.0) > 0.05 or "Arial" not in font or "Bold" not in font
        for _, size, font, _ in panel_letters
    ):
        raise RuntimeError(f"Panel letters are not 12-pt Arial Bold: {panel_letters}")
    if image_dpi and min(image_dpi) < 300.0:
        raise RuntimeError(f"Raster layer below 300 dpi: {min(image_dpi):.1f}")
    required = [
        "Measured",
        "Morpho-FM",
        "iStar",
        "Prostate | Androgen response",
        "Kidney | Hypoxia",
        "Fraction of genes with Pearson r >= threshold",
        "HVG recovery vs prediction quality (dense K)",
        "Internal holdout",
        "External slide 1",
        "External slide 2",
        "PCC = 0.993",
        "PCC = 0.527",
        "PCC = 0.974",
        "PCC = 0.283",
    ]
    for label in required:
        if label not in extracted:
            raise RuntimeError(f"Required label missing: {label}")
    for forbidden in (
        "variable genes",
        "Standardized programme score",
        "INT25 to INT28",
        "INT19 to INT24",
    ):
        if forbidden in extracted:
            raise RuntimeError(f"Removed label remains in PDF: {forbidden}")
    span_text = {label for label, _, _, _ in spans}
    whole_span_required = [
        "Prostate | Androgen response",
        "Kidney | Hypoxia",
        "Fraction of genes with Pearson r >= threshold",
        "HVG recovery vs prediction quality (dense K)",
        "Internal holdout",
        "External slide 1",
        "External slide 2",
        "PCC = 0.993",
        "PCC = 0.527",
        "PCC = 0.974",
        "PCC = 0.283",
    ]
    for label in whole_span_required:
        if label not in span_text:
            raise RuntimeError(f"Label is not stored as one editable text span: {label}")
    expected_width = round(FIGURE_WIDTH_MM * MM_TO_INCH * 600)
    expected_height = round(FIGURE_HEIGHT_MM * MM_TO_INCH * 600)
    pixmap = fitz.Pixmap(str(png_path))
    png_size = (pixmap.width, pixmap.height)
    pixmap = None
    if abs(png_size[0] - expected_width) > 1 or abs(png_size[1] - expected_height) > 1:
        raise RuntimeError(f"Unexpected PNG dimensions: {png_size}")
    print(
        f"[verified] {width_mm:.1f} x {height_mm:.1f} mm; "
        f"fonts {sorted(fonts)}; minimum raster resolution {min(image_dpi):.0f} dpi; "
        f"PNG {png_size[0]} x {png_size[1]} px"
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    phase6_output = args.phase6_output.expanduser().resolve()
    generalization = args.generalization.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pdf_path, png_path = build_figure(root, phase6_output, generalization, output_dir)
    validate_output(pdf_path, png_path)
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
