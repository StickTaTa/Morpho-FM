from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


BLUE = "#2F6BB1"
BLUE_FILL = "#E5F0FA"
PURPLE = "#60439A"
PURPLE_FILL = "#EEE8F7"
ORANGE = "#D97732"
ORANGE_FILL = "#FBE9DB"
GREEN = "#2F7D5B"
GREEN_FILL = "#E8F3ED"
GRAY = "#667085"
GRAY_FILL = "#F3F4F6"
TEXT = "#202124"


def rounded_box(ax, x, y, w, h, text, face, edge, fontsize=11, weight="normal", radius=0.012):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=BLUE, dashed=False, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.1,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def crop_normalized(image: Image.Image, bounds):
    width, height = image.size
    left, top, right, bottom = bounds
    return image.crop(
        (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        )
    )


def add_image(ax, image, extent, edge=GRAY):
    x0, x1, y0, y1 = extent
    ax.imshow(image, extent=extent, aspect="auto", zorder=1)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=1.4,
            edgecolor=edge,
            zorder=3,
        )
    )


def feature_field(rows=42, cols=34):
    y, x = np.mgrid[0:1:complex(rows), 0:1:complex(cols)]
    field = (
        0.95 * np.exp(-((x - 0.28) ** 2 + (y - 0.72) ** 2) / 0.030)
        + 0.80 * np.exp(-((x - 0.70) ** 2 + (y - 0.35) ** 2) / 0.045)
        + 0.45 * np.exp(-((x - 0.55) ** 2 + (y - 0.80) ** 2) / 0.020)
        + 0.12 * np.sin(7 * x) * np.cos(8 * y)
    )
    field -= field.min()
    return field / field.max()


def draw_spot_array(ax, origin, rows=3, cols=4, spacing=0.018):
    x0, y0 = origin
    colors = [PURPLE, "#7283B6"]
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(
                Circle(
                    (x0 + col * spacing, y0 - row * spacing),
                    0.0068,
                    facecolor=colors[(row + col) % 2],
                    edgecolor="#35255F",
                    linewidth=0.8,
                    zorder=3,
                )
            )


def draw_overview(ax, source):
    he_crop = crop_normalized(source, (0.027, 0.115, 0.152, 0.305))
    observed_crop = crop_normalized(source, (0.806, 0.180, 0.918, 0.345))
    dense_crop = crop_normalized(source, (0.806, 0.430, 0.918, 0.565))

    ax.text(0.012, 0.962, "a", fontsize=25, fontweight="bold", va="top")
    ax.text(
        0.035,
        0.962,
        "Morpho-FM couples pathology and transcriptomic pretraining",
        fontsize=23,
        fontweight="bold",
        va="top",
        color=TEXT,
    )

    ax.text(0.067, 0.878, "H&E whole-slide image", fontsize=10.5, ha="center", fontweight="bold")
    add_image(ax, he_crop, (0.018, 0.115, 0.625, 0.855), edge=BLUE)
    arrow(ax, (0.120, 0.740), (0.168, 0.740), BLUE)

    rounded_box(
        ax,
        0.170,
        0.650,
        0.130,
        0.175,
        "Pretrained HIPT\nimage-side prior",
        BLUE_FILL,
        BLUE,
        fontsize=13,
        weight="bold",
    )
    arrow(ax, (0.305, 0.740), (0.347, 0.740), BLUE)

    grid_extent = (0.350, 0.455, 0.650, 0.825)
    ax.imshow(feature_field(), extent=grid_extent, cmap="viridis", aspect="auto", zorder=1)
    ax.add_patch(
        Rectangle(
            (grid_extent[0], grid_extent[2]),
            grid_extent[1] - grid_extent[0],
            grid_extent[3] - grid_extent[2],
            fill=False,
            edgecolor=BLUE,
            linewidth=1.8,
            zorder=3,
        )
    )
    for i in range(1, 6):
        x = grid_extent[0] + i * (grid_extent[1] - grid_extent[0]) / 6
        ax.plot([x, x], [grid_extent[2], grid_extent[3]], color="white", alpha=0.45, lw=0.6)
    for i in range(1, 7):
        y = grid_extent[2] + i * (grid_extent[3] - grid_extent[2]) / 8
        ax.plot([grid_extent[0], grid_extent[1]], [y, y], color="white", alpha=0.45, lw=0.6)
    ax.text(0.4025, 0.842, "Cached HIPT-based\nfeature grid", ha="center", fontsize=11.5, fontweight="bold")

    rounded_box(ax, 0.492, 0.714, 0.102, 0.102, "Local MIL bags\nat measured sites", GRAY_FILL, GRAY, 10.5, "bold")
    rounded_box(ax, 0.492, 0.572, 0.102, 0.085, "Tissue-valid\ngrid positions", GRAY_FILL, GRAY, 10.5, "bold")
    arrow(ax, (0.460, 0.752), (0.489, 0.764), BLUE)
    arrow(ax, (0.460, 0.690), (0.489, 0.615), BLUE)

    rounded_box(ax, 0.630, 0.642, 0.090, 0.145, "Trainable\nadapter", BLUE_FILL, BLUE, 12.5, "bold")
    arrow(ax, (0.596, 0.765), (0.627, 0.735), BLUE)
    arrow(ax, (0.596, 0.615), (0.627, 0.684), BLUE)

    rounded_box(
        ax,
        0.755,
        0.625,
        0.125,
        0.180,
        "CellFM-initialized\ndecoding module\nmolecular-side prior",
        PURPLE_FILL,
        PURPLE,
        11.5,
        "bold",
    )
    arrow(ax, (0.723, 0.714), (0.752, 0.714), PURPLE)

    add_image(ax, observed_crop, (0.914, 0.976, 0.720, 0.815), edge=ORANGE)
    add_image(ax, dense_crop, (0.914, 0.976, 0.570, 0.665), edge=ORANGE)
    arrow(ax, (0.882, 0.748), (0.908, 0.767), ORANGE)
    arrow(ax, (0.882, 0.680), (0.908, 0.618), ORANGE)
    ax.text(
        0.945,
        0.830,
        "Observed sites\nbag mean x assay size factor\ncount-scale reconstruction",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=TEXT,
        linespacing=1.1,
    )
    ax.text(
        0.945,
        0.553,
        "Unmeasured grid\nnormalized expression-rate field\n(candidate output)",
        ha="center",
        va="top",
        fontsize=7.2,
        color=TEXT,
        linespacing=1.1,
    )

    ax.text(0.040, 0.548, "Paired ST coordinates + counts", fontsize=11.5, fontweight="bold", color=GREEN)
    draw_spot_array(ax, (0.055, 0.515), rows=3, cols=4, spacing=0.018)
    arrow(
        ax,
        (0.128, 0.505),
        (0.535, 0.707),
        GREEN,
        dashed=True,
        connectionstyle="angle3,angleA=0,angleB=90",
    )
    ax.text(0.300, 0.535, "training supervision only", fontsize=10.5, color=GREEN, fontweight="bold")


def draw_mil_panel(ax):
    ax.text(0.012, 0.455, "b", fontsize=25, fontweight="bold", va="top")
    ax.text(0.035, 0.455, "Local MIL bags approximate assay support", fontsize=18, fontweight="bold", va="top")

    x0, y0, w, h = 0.060, 0.090, 0.160, 0.270
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=BLUE_FILL, edgecolor=BLUE, linewidth=1.6))
    for i in range(1, 8):
        x = x0 + i * w / 8
        ax.plot([x, x], [y0, y0 + h], color="#92B8DD", lw=0.7)
    for i in range(1, 8):
        y = y0 + i * h / 8
        ax.plot([x0, x0 + w], [y, y], color="#92B8DD", lw=0.7)
    center = (x0 + 0.085, y0 + 0.135)
    ax.add_patch(Circle(center, 0.074, fill=False, edgecolor=PURPLE, linewidth=2.2, alpha=0.65))
    selected = [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0), (0, 2), (1, -1), (1, 1), (2, 0)]
    for dx, dy in selected:
        ax.add_patch(Circle((center[0] + dx * 0.021, center[1] + dy * 0.021), 0.006, facecolor=PURPLE, edgecolor="white", lw=0.5))
    ax.text(x0 + w / 2, y0 + h + 0.018, "Cached feature grid", fontsize=11.5, ha="center", fontweight="bold")
    ax.text(center[0], y0 - 0.020, "finite assay support", fontsize=9.5, ha="center", color=GRAY)

    arrow(ax, (0.230, 0.225), (0.275, 0.225), PURPLE)
    rounded_box(ax, 0.285, 0.135, 0.155, 0.185, "Instance set\n$X = \\{x_1, ..., x_K\\}$\n\nlocal morphology\nfor one measurement", PURPLE_FILL, PURPLE, 11.5, "bold")
    ax.text(0.070, 0.055, "The bag approximates the tissue region summarized by one spatial measurement.", fontsize=10.5, color=TEXT)


def draw_decoder_panel(ax):
    ax.text(0.505, 0.455, "c", fontsize=25, fontweight="bold", va="top")
    ax.text(0.528, 0.455, "Adapter and CellFM-initialized molecular readout", fontsize=18, fontweight="bold", va="top")

    rounded_box(ax, 0.525, 0.225, 0.070, 0.115, "HIPT-based\ninstance $x_i$", BLUE_FILL, BLUE, 9.8, "bold")
    rounded_box(ax, 0.615, 0.225, 0.070, 0.115, "Trainable\nadapter", BLUE_FILL, BLUE, 9.8, "bold")
    rounded_box(ax, 0.705, 0.205, 0.105, 0.155, "CellFM-initialized\nmodule\nmost backbone frozen", PURPLE_FILL, PURPLE, 9.5, "bold")
    arrow(ax, (0.598, 0.282), (0.612, 0.282), BLUE)
    arrow(ax, (0.688, 0.282), (0.702, 0.282), PURPLE)

    rounded_box(ax, 0.830, 0.242, 0.060, 0.080, "instance rate\n$\\rho_i$", ORANGE_FILL, ORANGE, 9.3, "bold")
    arrow(ax, (0.813, 0.282), (0.827, 0.282), ORANGE)

    rounded_box(ax, 0.910, 0.250, 0.072, 0.090, "Observed\nbag mean x $s_n$\ncount scale", ORANGE_FILL, ORANGE, 8.5, "bold")
    rounded_box(ax, 0.910, 0.105, 0.072, 0.090, "Dense\nsingle-grid rate\ncandidate", GRAY_FILL, GRAY, 8.5, "bold")
    arrow(ax, (0.893, 0.282), (0.907, 0.295), ORANGE)
    arrow(ax, (0.892, 0.260), (0.907, 0.160), GRAY, dashed=True, connectionstyle="arc3,rad=0.18")

    ax.text(0.757, 0.176, "gene embeddings and decoder projection are fine-tuned", fontsize=8.5, ha="center", color=PURPLE)
    ax.text(0.946, 0.215, "assay size factor\nat observed sites only", fontsize=7.5, ha="center", va="center", color=ORANGE, linespacing=1.0)


def build_figure(source_path: Path, outputs):
    with Image.open(source_path) as image:
        source = image.convert("RGB").copy()

    plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(14, 7.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_overview(ax, source)
    ax.plot([0.01, 0.99], [0.475, 0.475], color="#D0D5DD", linewidth=1.2)
    ax.plot([0.490, 0.490], [0.025, 0.455], color="#D0D5DD", linewidth=1.2)
    draw_mil_panel(ax)
    draw_decoder_panel(ax)

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=270, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Redraw the Morpho-FM overview figure with explicit dual-pretraining data flow.")
    parser.add_argument("--source", type=Path, required=True, help="Existing workflow PNG used only for image thumbnails.")
    parser.add_argument("--output", type=Path, required=True, help="Primary PNG output path.")
    parser.add_argument("--mirror", type=Path, help="Optional second PNG path to overwrite with the same figure.")
    args = parser.parse_args()

    outputs = [args.output]
    if args.mirror is not None and args.mirror != args.output:
        outputs.append(args.mirror)
    build_figure(args.source, outputs)


if __name__ == "__main__":
    main()
