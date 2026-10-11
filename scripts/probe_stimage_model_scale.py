"""Feasibility gate for the STimage benchmark row: can upstream build at this gene count?

Upstream `stimage/_model.py::CNN_NB_multiple_genes` creates one `Dense(2)` +
one `Lambda(negative_binomial_layer)` output head *per gene*. The official tutorial
and the upstream HER2ST/OOD scripts only ever run a handful of genes (e.g. the
8-gene and 30-gene lists hard-coded in `stimage/02_Training.py`), never a whole
transcriptome. This repository's gene panel is `data/spatial_data/common_genes.txt`,
which is far larger, so model construction and a single forward pass must be probed
before any benchmark row is promised.

This script does NOT train. It reports, for a given gene count:

  * whether the upstream model builds at all,
  * how long construction takes and how many parameters it produces,
  * whether one forward pass runs, and its GPU peak memory.

Runs on the upstream source unmodified -- the model is constructed by calling
`CNN_NB_multiple_genes` itself, so a failure here is upstream's behaviour and not
an artefact of a reimplementation.

Usage (remote, inside a Slurm allocation):
    python scripts/probe_stimage_model_scale.py \
        --upstream-root benchmark/STimage --n-genes 17511 \
        --out-json results/benchmark_audit/stimage_scale_probe_17511.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe upstream STimage model scale.")
    parser.add_argument("--upstream-root", required=True, help="Directory holding the upstream `stimage/` package.")
    parser.add_argument("--n-genes", type=int, required=True)
    parser.add_argument("--tile-size", type=int, default=299)
    parser.add_argument("--cnn-base", default="resnet50")
    parser.add_argument("--fine-tuning", action="store_true")
    parser.add_argument("--forward-batch", type=int, default=2, help="0 skips the forward pass.")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    upstream_root = Path(args.upstream_root).expanduser().resolve()
    if not (upstream_root / "stimage" / "_model.py").is_file():
        raise FileNotFoundError(f"Upstream STimage package not found under {upstream_root}")

    report: dict = {
        "upstream_root": str(upstream_root),
        "n_genes": int(args.n_genes),
        "tile_size": int(args.tile_size),
        "cnn_base": str(args.cnn_base),
        "fine_tuning": bool(args.fine_tuning),
        "forward_batch": int(args.forward_batch),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    import numpy as np
    import tensorflow as tf  # noqa: E402  (import cost reported below)

    # `tf.keras` is a lazy-loaded module in TF 2.15 and exposes no `__version__`,
    # so read the standalone `keras` package the environment actually installed.
    try:
        import keras as _keras_package

        keras_version = str(_keras_package.__version__)
    except Exception:  # noqa: BLE001
        keras_version = "unknown"

    report["tensorflow"] = tf.__version__
    report["keras"] = keras_version
    report["tf_built_with_cuda"] = bool(tf.test.is_built_with_cuda())
    gpus = tf.config.list_physical_devices("GPU")
    report["gpus"] = [g.name for g in gpus]
    report["gpu_visible"] = bool(gpus)
    print(f"[env] tf={tf.__version__} keras={keras_version} gpus={report['gpus']}", flush=True)
    if not gpus:
        print("[env] WARNING: no GPU visible; this probe is meaningful only on a GPU node.", flush=True)

    sys.path.insert(0, str(upstream_root))
    from stimage._model import CNN_NB_multiple_genes  # noqa: E402

    tile_shape = (int(args.tile_size), int(args.tile_size), 3)
    started = time.time()
    try:
        model = CNN_NB_multiple_genes(
            tile_shape, int(args.n_genes), cnnbase=str(args.cnn_base), ft=bool(args.fine_tuning)
        )
    except Exception as exc:  # noqa: BLE001 - the whole point is to record how it fails
        report["build_ok"] = False
        report["build_seconds"] = round(time.time() - started, 2)
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-25:]
        print(f"[build] FAILED after {report['build_seconds']}s: {type(exc).__name__}: {exc}", flush=True)
        _write(report, args.out_json)
        return 1

    report["build_ok"] = True
    report["build_seconds"] = round(time.time() - started, 2)
    report["n_outputs"] = int(len(model.outputs))
    report["n_layers"] = int(len(model.layers))
    report["n_params"] = int(model.count_params())
    report["n_trainable_params"] = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    print(
        f"[build] ok in {report['build_seconds']}s | outputs={report['n_outputs']} "
        f"layers={report['n_layers']} params={report['n_params']:,}",
        flush=True,
    )

    if int(args.forward_batch) > 0 and gpus:
        x = np.random.default_rng(0).integers(0, 256, size=(int(args.forward_batch), *tile_shape), dtype=np.uint8)
        forward_started = time.time()
        try:
            outputs = model.predict(x.astype("float32"), batch_size=int(args.forward_batch), verbose=0)
            report["forward_ok"] = True
            report["forward_seconds"] = round(time.time() - forward_started, 2)
            report["n_output_arrays"] = int(len(outputs))
            report["output_shape_per_head"] = list(np.asarray(outputs[0]).shape)
            print(
                f"[forward] ok in {report['forward_seconds']}s | heads={len(outputs)} "
                f"head_shape={report['output_shape_per_head']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            report["forward_ok"] = False
            report["forward_seconds"] = round(time.time() - forward_started, 2)
            report["forward_error_type"] = type(exc).__name__
            report["forward_error"] = str(exc)
            print(f"[forward] FAILED after {report['forward_seconds']}s: {type(exc).__name__}: {exc}", flush=True)

        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            report["gpu_peak_bytes"] = int(info.get("peak", 0))
            report["gpu_current_bytes"] = int(info.get("current", 0))
            print(
                f"[mem] gpu peak={report['gpu_peak_bytes'] / 2**20:.0f} MiB "
                f"current={report['gpu_current_bytes'] / 2**20:.0f} MiB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            report["gpu_mem_error"] = str(exc)

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
