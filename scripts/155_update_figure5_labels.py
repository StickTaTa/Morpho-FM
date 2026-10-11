"""Synchronize Figure 5 terminology and setting colours in final raster assets."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\人生的第4篇论文")
SOURCE_TIFF = PROJECT_ROOT / "Figure" / "Figure_5" / "Figure_5.tif"
MANUSCRIPT_PNG = PROJECT_ROOT / "manuscript_npj" / "Figures" / "fig_main_05.png"

FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_ITALIC = Path(r"C:\Windows\Fonts\ariali.ttf")


def recolour_setting_key(image: Image.Image) -> Image.Image:
    """Use the red/blue setting colours already used in panels b, d and e."""

    rgb = np.asarray(image.convert("RGB")).copy()
    panel_a = rgb[:1350]

    colour_map = {
        (209, 229, 227): (228, 124, 119),  # within-section: green to red
        (242, 198, 188): (151, 199, 220),  # cross-section: pink to blue
    }
    tolerance = 8

    # Undo any earlier broad replacement outside the setting matrix and key.
    non_key = panel_a[:, :2850]
    reverse_map = {target: source for source, target in colour_map.items()}
    for source, target in reverse_map.items():
        source_array = np.asarray(source, dtype=np.int16)
        distance = np.abs(non_key.astype(np.int16) - source_array)
        mask = np.all(distance <= tolerance, axis=2)
        non_key[mask] = target
    panel_a[:, :2850] = non_key

    setting_key = panel_a[:, 2850:]
    for source, target in colour_map.items():
        source_array = np.asarray(source, dtype=np.int16)
        distance = np.abs(setting_key.astype(np.int16) - source_array)
        mask = np.all(distance <= tolerance, axis=2)
        setting_key[mask] = target

    panel_a[:, 2850:] = setting_key
    rgb[:1350] = panel_a
    return Image.fromarray(rgb, mode="RGB")


def draw_centered_mixed_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    segments: list[tuple[str, ImageFont.FreeTypeFont]],
) -> None:
    widths = [draw.textlength(text, font=font) for text, font in segments]
    x = center_x - sum(widths) / 2
    for (text, font), width in zip(segments, widths):
        draw.text((x, center_y), text, fill="black", font=font, anchor="lm")
        x += width


def update_panel_a_and_b_labels(image: Image.Image) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    regular = ImageFont.truetype(str(FONT_REGULAR), 58)

    draw.rectangle((1450, 0, 2700, 82), fill="white")
    draw.text(
        (2075, 43),
        "Prediction at measured locations",
        fill="black",
        font=regular,
        anchor="mm",
    )

    draw.rectangle((1450, 350, 2700, 457), fill="white")
    draw.text(
        (2075, 405),
        "Candidate dense molecular fields",
        fill="black",
        font=regular,
        anchor="mm",
    )

    panel_b_titles = [
        ((0, 1250, 1500, 1445), (750, 1347), "306 retained genes\nmeasured locations"),
        ((1500, 1250, 2800, 1445), (2150, 1347), "29 breast markers\nmeasured locations"),
        (
            (2800, 1250, 3997, 1445),
            (3398, 1347),
            "29 breast markers\nre-aggregation to measured locations",
        ),
    ]
    for region, center, label in panel_b_titles:
        draw.rectangle(region, fill="white")
        draw.multiline_text(
            center,
            label,
            fill="black",
            font=regular,
            anchor="mm",
            align="center",
            spacing=0,
        )
    return output


def update_panel_c_titles(image: Image.Image) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    regular = ImageFont.truetype(str(FONT_REGULAR), 58)
    italic = ImageFont.truetype(str(FONT_ITALIC), 58)

    title_regions = [
        (1068, 2300, 2048, 2488),
        (2048, 2300, 3027, 2488),
        (3027, 2300, 3997, 2488),
    ]
    for region in title_regions:
        draw.rectangle(region, fill="white")

    draw_centered_mixed_text(
        draw,
        center_x=(1068 + 2048) / 2,
        center_y=2418,
        segments=[
            ("Measured ", regular),
            ("ERBB2", italic),
            (" expression", regular),
        ],
    )
    draw.text(
        ((2048 + 3027) / 2, 2418),
        "Within-section reconstruction",
        fill="black",
        font=regular,
        anchor="mm",
    )
    draw.text(
        ((3027 + 3997) / 2, 2418),
        "Cross-section transfer (S2-to-S1)",
        fill="black",
        font=regular,
        anchor="mm",
    )
    return output


def main() -> None:
    with Image.open(SOURCE_TIFF) as source:
        dpi = source.info.get("dpi", (600, 600))
        icc_profile = source.info.get("icc_profile")
        updated = update_panel_c_titles(update_panel_a_and_b_labels(recolour_setting_key(source)))

    save_kwargs = {"dpi": dpi}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    updated.save(SOURCE_TIFF, compression="tiff_lzw", **save_kwargs)
    updated.save(MANUSCRIPT_PNG, optimize=True, **save_kwargs)


if __name__ == "__main__":
    main()
