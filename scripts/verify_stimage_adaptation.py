"""Audit gate for the STimage row: prove the vectorized form equals upstream, don't assert it.

The STimage benchmark row cannot use upstream's literal multi-output Keras model at
this repository's 17,511-gene panel (see the measurement table in
`benchmark/hest_dataset.py` and the reference project's cancelled stress job). It
therefore trains a vectorized head that stacks upstream's per-gene `Dense(2)` layers
into one einsum, and a restated negative-binomial loss.

Both are claimed to be exact reformulations. This script checks that claim against
upstream's own code at three gene counts, using upstream's own
`negative_binomial_layer` and `negative_binomial_loss` as the reference:

  * head output: max absolute difference, must be <= --atol
  * loss: max absolute difference, must be <= --atol

It also times building the vectorized head at the full panel width, to be compared
against the recorded 191.3 s that upstream's `CNN_NB_multiple_genes` needs at the
same width.

Usage (remote, inside a Slurm allocation):
    python scripts/verify_stimage_adaptation.py \
        --upstream-root benchmark/STimage --n-genes 4 \
        --out-json results/benchmark_audit/stimage_vectorization_check_g4.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the STimage vectorized adapter against upstream.")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--n-genes", type=int, default=64, help="Gene count for the exactness check.")
    parser.add_argument("--feature-dim", type=int, default=2048, help="ResNet50 GAP width used by the real row.")
    parser.add_argument("--rel-atol", type=float, default=1e-5, help="Relative tolerance for both checks.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timing-genes",
        type=int,
        default=17511,
        help="Gene count at which to time the vectorized head build; 0 disables.",
    )
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if str(root / "benchmark") not in sys.path:
        sys.path.insert(0, str(root / "benchmark"))

    import tensorflow as tf

    from hest_dataset import (
        STIMAGE_FEATURE_DIM,
        bind_stimage_upstream,
        build_stimage_vectorized_head,
        verify_stimage_vectorization,
    )

    report: dict = {
        "tensorflow": tf.__version__,
        "upstream_root": str(Path(args.upstream_root).resolve()),
        "n_genes_checked": int(args.n_genes),
        "feature_dim": int(args.feature_dim),
        "rel_atol": float(args.rel_atol),
    }
    print(f"[env] tf={tf.__version__} check_genes={args.n_genes} feature_dim={args.feature_dim}", flush=True)

    try:
        bind_stimage_upstream(args.upstream_root)
        check = verify_stimage_vectorization(
            args.upstream_root,
            feature_dim=int(args.feature_dim),
            n_genes=int(args.n_genes),
            seed=int(args.seed),
            rel_atol=float(args.rel_atol),
        )
        report["exactness"] = check
        print(
            f"[exactness] head_rel={check['relative_head_difference']:.3e} "
            f"loss_rel_f64={check['relative_loss_difference_float64']:.3e} "
            f"(f32={check['relative_loss_difference_float32']:.3e}) -> "
            f"{'MATCH' if check['head_matches'] and check['loss_matches'] else 'MISMATCH'}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        report["exactness_failed"] = True
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
        # The check attaches its measurements to the exception so a failure still
        # produces a diagnosable record rather than an error string alone.
        if hasattr(exc, "report"):
            report["exactness"] = exc.report
        print(f"[exactness] FAILED: {type(exc).__name__}: {exc}", flush=True)
        _write(report, args.out_json)
        return 1

    if int(args.timing_genes) > 0:
        started = time.time()
        head = build_stimage_vectorized_head(int(args.feature_dim), int(args.timing_genes))
        build_seconds = time.time() - started
        report["vectorized_head"] = {
            "n_genes": int(args.timing_genes),
            "feature_dim": int(args.feature_dim),
            "build_seconds": round(build_seconds, 3),
            "kernel_shape": list(head.kernel.shape),
            "n_params": int(head.kernel.numpy().size + head.bias.numpy().size),
        }
        print(
            f"[timing] vectorized head at {args.timing_genes} genes built in "
            f"{build_seconds:.3f}s | kernel {list(head.kernel.shape)}",
            flush=True,
        )

    report["ok"] = True
    _write(report, args.out_json)
    return 0


def _write(report: dict, out_json: str | None) -> None:
    print(json.dumps(report, indent=2), flush=True)
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[out] wrote {out_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
