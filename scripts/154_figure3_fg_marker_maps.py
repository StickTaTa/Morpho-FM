from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


PROJECT_ROOT = Path.home() / "Desktop" / "\u4eba\u751f\u7684\u7b2c4\u7bc7\u8bba\u6587"
FIGURE_DIR = PROJECT_ROOT / "Figure" / "Figure_3"
DEFAULT_SECTION_1 = FIGURE_DIR / "marker_maps_top6_separate1.pdf"
DEFAULT_SECTION_2 = FIGURE_DIR / "marker_maps_top6_separate2.pdf"
DEFAULT_OUTPUT_DIR = FIGURE_DIR

FIGURE_WIDTH_MM = 170.0
FIGURE_HEIGHT_MM = 76.0
MM_TO_INCH = 1.0 / 25.4
PT_PER_MM = 72.0 / 25.4
RASTER_DPI = 600.0

ARIAL = Path(r"C:\Windows\Fonts\arial.ttf")
ARIAL_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
ARIAL_ITALIC = Path(r"C:\Windows\Fonts\ariali.ttf")

MAP_WIDTH_MM = 34.8
MAP_HEIGHT_MM = MAP_WIDTH_MM / (272.64 / 227.76)
TOP_MAP_TOP_MM = 8.8
BOTTOM_MAP_TOP_MM = 40.2
ROW_LABEL_X_MM = 1.5
ROW_LABEL_WIDTH_MM = 4.8
SECTION_BAR_TOP_MM = 70.35
SECTION_BAR_HEIGHT_MM = 4.8


PANELS = [
    {
        "letter": "f",
        "section": "Section 1",
        "source_key": "section_1",
        "letter_x_mm": 0.8,
        "map_x_mm": [7.5, 43.8],
        "genes": [
            {
                "name": "SMARCB1",
                "pcc": 0.717,
                "measured": (7.20, 73.93, 279.84, 301.69),
                "prediction": (335.70, 73.93, 608.34, 301.69),
            },
            {
                "name": "TFE3",
                "pcc": 0.674,
                "measured": (7.20, 353.36, 279.84, 581.12),
                "prediction": (335.70, 353.36, 608.34, 581.12),
            },
        ],
    },
    {
        "letter": "g",
        "section": "Section 2",
        "source_key": "section_2",
        "letter_x_mm": 82.0,
        "map_x_mm": [88.0, 124.3],
        "genes": [
            {
                "name": "AQP1",
                "pcc": 0.530,
                "measured": (7.20, 630.90, 284.40, 862.50),
                "prediction": (334.11, 630.90, 611.31, 862.50),
            },
            {
                "name": "VCAM1",
                "pcc": 0.410,
                "measured": (661.03, 630.90, 938.23, 862.50),
                "prediction": (987.94, 630.90, 1265.14, 862.50),
            },
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Figure 3 panels f-g from the existing external marker-map PDFs."
    )
    parser.add_argument("--section-1", type=Path, default=DEFAULT_SECTION_1)
    parser.add_argument("--section-2", type=Path, default=DEFAULT_SECTION_2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def add_axes_mm(
    figure: plt.Figure,
    left_mm: float,
    top_mm: float,
    width_mm: float,
    height_mm: float,
) -> plt.Axes:
    return figure.add_axes(
        [
            left_mm / FIGURE_WIDTH_MM,
            1.0 - (top_mm + height_mm) / FIGURE_HEIGHT_MM,
            width_mm / FIGURE_WIDTH_MM,
            height_mm / FIGURE_HEIGHT_MM,
        ]
    )


def render_clip(page: fitz.Page, clip_values: tuple[float, ...]) -> np.ndarray:
    clip = fitz.Rect(clip_values)
    target_pixels = round(MAP_WIDTH_MM / 25.4 * RASTER_DPI)
    scale = target_pixels / clip.width
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return array[:, :, :3]


def add_map(figure: plt.Figure, image: np.ndarray, left_mm: float, top_mm: float) -> None:
    axis = add_axes_mm(figure, left_mm, top_mm, MAP_WIDTH_MM, MAP_HEIGHT_MM)
    axis.imshow(image, interpolation="bilinear")
    axis.set_axis_off()


def build_figure(section_1: Path, section_2: Path, output_dir: Path) -> tuple[Path, Path]:
    for font in (ARIAL, ARIAL_BOLD, ARIAL_ITALIC):
        if not font.exists():
            raise FileNotFoundError(font)
        font_manager.fontManager.addfont(str(font))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": RASTER_DPI,
        }
    )

    sources = {
        "section_1": fitz.open(section_1),
        "section_2": fitz.open(section_2),
    }
    figure = plt.figure(
        figsize=(
            FIGURE_WIDTH_MM * MM_TO_INCH,
            FIGURE_HEIGHT_MM * MM_TO_INCH,
        ),
        facecolor="white",
    )

    for top_mm, label in (
        (TOP_MAP_TOP_MM, "True"),
        (BOTTOM_MAP_TOP_MM, "Prediction"),
    ):
        axis = add_axes_mm(
            figure,
            ROW_LABEL_X_MM,
            top_mm,
            ROW_LABEL_WIDTH_MM,
            MAP_HEIGHT_MM,
        )
        axis.set_facecolor("#E6E6E6")
        axis.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.8,
            fontweight="normal",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)

    for panel in PANELS:
        page = sources[panel["source_key"]][0]
        figure.text(
            panel["letter_x_mm"] / FIGURE_WIDTH_MM,
            0.982,
            panel["letter"],
            ha="left",
            va="top",
            fontsize=12.0,
            fontweight="bold",
        )
        for left_mm, gene in zip(panel["map_x_mm"], panel["genes"], strict=True):
            center_x = (left_mm + MAP_WIDTH_MM / 2.0) / FIGURE_WIDTH_MM
            figure.text(
                center_x,
                0.991,
                gene["name"],
                ha="center",
                va="top",
                fontsize=7.4,
                fontstyle="italic",
                fontweight="normal",
            )
            figure.text(
                center_x,
                0.950,
                f"(r={gene['pcc']:.3f})",
                ha="center",
                va="top",
                fontsize=6.8,
                fontweight="normal",
            )
            add_map(
                figure,
                render_clip(page, gene["measured"]),
                left_mm,
                TOP_MAP_TOP_MM,
            )
            add_map(
                figure,
                render_clip(page, gene["prediction"]),
                left_mm,
                BOTTOM_MAP_TOP_MM,
            )

        section_left = panel["map_x_mm"][0]
        section_right = panel["map_x_mm"][-1] + MAP_WIDTH_MM
        section_axis = add_axes_mm(
            figure,
            section_left,
            SECTION_BAR_TOP_MM,
            section_right - section_left,
            SECTION_BAR_HEIGHT_MM,
        )
        section_axis.set_facecolor("#E6E6E6")
        section_axis.text(
            0.5,
            0.5,
            panel["section"],
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="normal",
        )
        section_axis.set_xticks([])
        section_axis.set_yticks([])
        section_axis.spines[:].set_visible(False)

    color_axis = add_axes_mm(
        figure,
        162.0,
        TOP_MAP_TOP_MM,
        1.8,
        BOTTOM_MAP_TOP_MM + MAP_HEIGHT_MM - TOP_MAP_TOP_MM,
    )
    colorbar = ColorbarBase(
        color_axis,
        cmap=plt.get_cmap("viridis"),
        norm=Normalize(0.0, 1.0),
        orientation="vertical",
    )
    colorbar.set_ticks([])
    colorbar.outline.set_linewidth(0.55)
    figure.text(
        164.4 / FIGURE_WIDTH_MM,
        1.0 - TOP_MAP_TOP_MM / FIGURE_HEIGHT_MM,
        "High",
        ha="left",
        va="top",
        fontsize=6.4,
    )
    figure.text(
        164.4 / FIGURE_WIDTH_MM,
        1.0 - (BOTTOM_MAP_TOP_MM + MAP_HEIGHT_MM) / FIGURE_HEIGHT_MM,
        "Low",
        ha="left",
        va="bottom",
        fontsize=6.4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig3_fg_block.pdf"
    png_path = output_dir / "fig3_fg_block.png"
    figure.savefig(pdf_path, format="pdf", facecolor="white")
    figure.savefig(png_path, format="png", dpi=RASTER_DPI, facecolor="white")
    plt.close(figure)
    for document in sources.values():
        document.close()
    return pdf_path, png_path


def validate_output(pdf_path: Path, png_path: Path) -> None:
    document = fitz.open(pdf_path)
    page = document[0]
    width_mm = page.rect.width / PT_PER_MM
    height_mm = page.rect.height / PT_PER_MM
    spans = [
        (
            span["text"].strip(),
            float(span["size"]),
            span["font"],
            fitz.Rect(span["bbox"]),
        )
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"].strip()
    ]
    fonts = {font[3].split("+")[-1] for font in page.get_fonts(full=True)}
    outside = [
        (text, bounds)
        for text, _, _, bounds in spans
        if bounds.x0 < -0.1
        or bounds.y0 < -0.1
        or bounds.x1 > page.rect.width + 0.1
        or bounds.y1 > page.rect.height + 0.1
    ]
    image_dpi = []
    for image in page.get_images(full=True):
        width, height = image[2], image[3]
        for rectangle in page.get_image_rects(image[0]):
            image_dpi.append(
                min(
                    width / (rectangle.width / 72.0),
                    height / (rectangle.height / 72.0),
                )
            )
    document.close()

    if abs(width_mm - FIGURE_WIDTH_MM) > 0.01 or abs(height_mm - FIGURE_HEIGHT_MM) > 0.01:
        raise RuntimeError(f"Unexpected page size: {width_mm:.3f} x {height_mm:.3f} mm")
    if outside:
        raise RuntimeError(f"Text extends outside the page: {outside}")
    if any("Arial" not in font for font in fonts):
        raise RuntimeError(f"Non-Arial font embedded: {sorted(fonts)}")
    panel_letters = [item for item in spans if item[0] in {"f", "g"}]
    if len(panel_letters) != 2 or any(
        abs(size - 12.0) > 0.05 or "Bold" not in font
        for _, size, font, _ in panel_letters
    ):
        raise RuntimeError(f"Panel letters are not 12-pt Arial Bold: {panel_letters}")
    required = [
        "SMARCB1",
        "(r=0.717)",
        "TFE3",
        "(r=0.674)",
        "AQP1",
        "(r=0.530)",
        "VCAM1",
        "(r=0.410)",
        "True",
        "Prediction",
        "Section 1",
        "Section 2",
        "High",
        "Low",
    ]
    span_text = [text for text, _, _, _ in spans]
    for label in required:
        if span_text.count(label) != 1:
            raise RuntimeError(f"Required editable label is missing or duplicated: {label}")
    if image_dpi and min(image_dpi) < 590.0:
        raise RuntimeError(f"Raster layer below 590 dpi: {min(image_dpi):.1f}")

    expected_width = round(FIGURE_WIDTH_MM * MM_TO_INCH * RASTER_DPI)
    expected_height = round(FIGURE_HEIGHT_MM * MM_TO_INCH * RASTER_DPI)
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
    pdf_path, png_path = build_figure(
        args.section_1.expanduser().resolve(),
        args.section_2.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    validate_output(pdf_path, png_path)
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
