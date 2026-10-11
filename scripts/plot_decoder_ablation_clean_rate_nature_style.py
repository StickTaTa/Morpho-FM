from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "plot_decoder_ablation_mlp_nature_style.py"

spec = importlib.util.spec_from_file_location("decoder_ablation_mlp_plot", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT}")
plot_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_mod)

plot_mod.ROOT = ROOT
plot_mod.OUTDIR = ROOT / "figures_nature_decoder_ablation_clean_rate"
plot_mod.FIG_DIR = plot_mod.OUTDIR / "figures"
plot_mod.TABLE_DIR = plot_mod.OUTDIR / "tables"

plot_mod.BRANCHES = {
    "Single | Pretrained CellFM": {
        "protocol": "single",
        "decoder": "Pretrained CellFM",
        "path": ROOT
        / "results"
        / "st_superres"
        / "cellfm_clean_rate_ablation"
        / "single_pretrained_INT28"
        / "INT28",
        "color": "#4DBBD5",
    },
    "Single | Random CellFM": {
        "protocol": "single",
        "decoder": "Random CellFM",
        "path": ROOT
        / "results"
        / "st_superres"
        / "cellfm_clean_rate_ablation"
        / "single_random_INT28"
        / "INT28",
        "color": "#91D1C2",
    },
    "Single | MLP decoder": {
        "protocol": "single",
        "decoder": "MLP decoder",
        "path": ROOT
        / "results"
        / "ablation_decoder"
        / "mlp_decoder_clean_rate_single_INT25_to_INT28"
        / "INT28",
        "color": "#7E8CE0",
    },
    "Multi | Pretrained CellFM": {
        "protocol": "multi",
        "decoder": "Pretrained CellFM",
        "path": ROOT
        / "results"
        / "st_superres"
        / "cellfm_clean_rate_ablation"
        / "multi_pretrained_INT28"
        / "INT28",
        "color": "#E64B35",
    },
    "Multi | Random CellFM": {
        "protocol": "multi",
        "decoder": "Random CellFM",
        "path": ROOT
        / "results"
        / "st_superres"
        / "cellfm_clean_rate_ablation"
        / "multi_random_INT28"
        / "INT28",
        "color": "#F39B7F",
    },
    "Multi | MLP decoder": {
        "protocol": "multi",
        "decoder": "MLP decoder",
        "path": ROOT
        / "results"
        / "ablation_decoder"
        / "mlp_decoder_clean_rate_multi_INT25_INT26_val_INT27_to_INT28"
        / "INT28",
        "color": "#B07AA1",
    },
}
plot_mod.ORDER = list(plot_mod.BRANCHES)


def main() -> None:
    plot_mod.main()
    for ext in ("pdf", "png", "svg"):
        src = plot_mod.FIG_DIR / f"decoder_ablation_mlp_supplement.{ext}"
        dst = plot_mod.FIG_DIR / f"decoder_ablation_clean_rate_supplement.{ext}"
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
