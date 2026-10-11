from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "phase6_structure_preservation"
DEFAULT_GMT = (
    ROOT
    / "data"
    / "spatial_data"
    / "genesets"
    / "phase6"
    / "h.all.v2025.1.Hs.symbols.gmt"
)
DEFAULT_FIGURE_DIR = Path(
    r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Phase_6_structure_preservation"
)

PROSTATE_SLIDES = ["INT25", "INT26", "INT27", "INT28"]
KIDNEY_SLIDES = ["INT13", "INT14", "INT15", "INT17", "INT18", "INT19", "INT21", "INT24"]
COHORT_SLIDES = {"Prostate": PROSTATE_SLIDES, "Kidney": KIDNEY_SLIDES}

GMT_SOURCE_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/"
    "h.all.v2025.1.Hs.symbols.gmt"
)
GMT_RELEASE = "MSigDB 2025.1.Hs Hallmark"
EXPECTED_GMT_SHA256 = "F22066AF72E215CCB7B89D88E492C07E1EEF17534C2CA7B0F9902CFECBBDD8E9"


@dataclass(frozen=True)
class Task:
    cohort: str
    train: str
    test: str
    morpho_dir: Path
    istar_dir: Path

    @property
    def task_id(self) -> str:
        return f"{self.cohort.lower()}_{self.train}_to_{self.test}"


@dataclass(frozen=True)
class PriorInput:
    setting: str
    initialization: str
    result_dir: Path
    gene_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate-controlled Phase 6 molecular-programme and spatial-fidelity analysis."
    )
    parser.add_argument(
        "command",
        choices=["self-test", "audit", "freeze", "pilot", "run", "summarize", "plot", "all"],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gmt", type=Path, default=DEFAULT_GMT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--skip-full-truth-check",
        action="store_true",
        help="Use deterministic truth probes only. The default fully checks one direction per target section.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in require_file(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json_genes(path: Path) -> list[str]:
    values = json.loads(require_file(path).read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Expected a JSON gene list: {path}")
    return [str(value) for value in values]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def canonical_gene_path(root: Path, cohort: str) -> Path:
    if cohort == "Prostate":
        name = "prostate_hest_v1_INT25_INT26_INT27_INT28_INT35.txt"
    elif cohort == "Kidney":
        name = "kidney_hest_v1_INT13_INT14_INT15_INT17_INT18_INT19_INT21_INT24.txt"
    else:
        raise ValueError(cohort)
    return root / "data" / "spatial_data" / "genesets" / name


def morpho_gene_path(root: Path, cohort: str) -> Path:
    if cohort == "Prostate":
        return root / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json"
    if cohort == "Kidney":
        return (
            root
            / "checkpoints"
            / "kidney_hest"
            / "st_superres"
            / "cellfm_single_INT18_to_INT21"
            / "common_genes.json"
        )
    raise ValueError(cohort)


def load_gene_orders(root: Path, cohort: str) -> tuple[list[str], list[str], list[str]]:
    canonical = read_lines(canonical_gene_path(root, cohort))
    morpho = read_json_genes(morpho_gene_path(root, cohort))
    istar = canonical.copy()
    if len(canonical) != 17_512 or len(set(canonical)) != len(canonical):
        raise ValueError(f"Invalid canonical panel for {cohort}: {len(canonical)} genes")
    if set(morpho) != set(canonical):
        raise ValueError(f"Morpho-FM and canonical gene sets differ for {cohort}")
    return canonical, morpho, istar


def build_tasks(root: Path) -> list[Task]:
    tasks: list[Task] = []
    for train in PROSTATE_SLIDES:
        tests = [slide for slide in PROSTATE_SLIDES if slide != train]
        tag = f"{train}_to_{'_'.join(tests)}"
        for test in tests:
            tasks.append(
                Task(
                    cohort="Prostate",
                    train=train,
                    test=test,
                    morpho_dir=root / "results" / "prostate_hest" / "morpho" / "single_superres" / tag / test,
                    istar_dir=root / "results" / "prostate_hest" / "istar" / "single" / tag / test,
                )
            )
    for train in KIDNEY_SLIDES:
        tests = [slide for slide in KIDNEY_SLIDES if slide != train]
        morpho_tag = f"train_{train}_to_{'_'.join(tests)}"
        istar_tag = f"{train}_to_{'_'.join(tests)}"
        for test in tests:
            tasks.append(
                Task(
                    cohort="Kidney",
                    train=train,
                    test=test,
                    morpho_dir=root / "results" / "kidney_hest" / "morpho" / "single_rotating" / morpho_tag / test,
                    istar_dir=root / "results" / "kidney_hest" / "istar" / "single" / istar_tag / test,
                )
            )
    if len(tasks) != 68:
        raise AssertionError(f"Expected 68 directions, found {len(tasks)}")
    return tasks


def prior_inputs(root: Path) -> list[PriorInput]:
    return [
        PriorInput(
            "Multi-slide",
            "Pretrained",
            root / "results" / "st_superres" / "cellfm_INT28",
            root / "checkpoints" / "st_superres" / "cellfm_forward_sum" / "common_genes.json",
        ),
        PriorInput(
            "Multi-slide",
            "Random",
            root / "results" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42_INT28",
            root / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "multi_seed42" / "common_genes.json",
        ),
        PriorInput(
            "Single-slide",
            "Pretrained",
            root / "results" / "prostate_hest" / "morpho" / "single_superres" / "INT25_to_INT28",
            root / "checkpoints" / "st_superres" / "cellfm_single_INT25_to_INT28" / "common_genes.json",
        ),
        PriorInput(
            "Single-slide",
            "Random",
            root / "results" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42_INT28",
            root / "checkpoints" / "st_superres" / "cellfm_randominit_ablation" / "single_seed42" / "common_genes.json",
        ),
    ]


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def h5ad_spots(root: Path, slide: str) -> tuple[list[str], np.ndarray]:
    path = root / "data" / "spatial_data" / f"{slide}.h5ad"
    with h5py.File(require_file(path), "r") as handle:
        spot_ids = decode_strings(handle["obs/_index"][:])
        coords = np.asarray(handle["obsm/spatial"][:], dtype=np.float64)
    if coords.shape != (len(spot_ids), 2):
        raise ValueError(f"Invalid coordinates in {path}: {coords.shape}")
    return spot_ids, coords


def source_indices_for_canonical(source_genes: list[str], canonical: list[str]) -> np.ndarray:
    lookup = {gene: index for index, gene in enumerate(source_genes)}
    try:
        return np.asarray([lookup[gene] for gene in canonical], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Missing canonical gene in source order: {exc.args[0]}") from exc


def aligned_truth_max_abs(
    morpho_path: Path,
    istar_path: Path,
    morpho_for_canonical: np.ndarray,
    istar_for_canonical: np.ndarray,
    full: bool,
) -> float:
    morpho = np.load(require_file(morpho_path), mmap_mode="r")
    istar = np.load(require_file(istar_path), mmap_mode="r")
    if morpho.shape != istar.shape:
        return float("inf")
    max_abs = 0.0
    if full:
        gene_slices = [slice(start, min(start + 512, morpho.shape[1])) for start in range(0, morpho.shape[1], 512)]
        for gene_slice in gene_slices:
            a = np.asarray(morpho[:, morpho_for_canonical[gene_slice]], dtype=np.float32)
            b = np.asarray(istar[:, istar_for_canonical[gene_slice]], dtype=np.float32)
            max_abs = max(max_abs, float(np.max(np.abs(a - b))))
    else:
        row_idx = np.unique(np.linspace(0, morpho.shape[0] - 1, min(128, morpho.shape[0]), dtype=np.int64))
        gene_idx = np.unique(np.linspace(0, morpho.shape[1] - 1, 512, dtype=np.int64))
        a = np.asarray(morpho[np.ix_(row_idx, morpho_for_canonical[gene_idx])], dtype=np.float32)
        b = np.asarray(istar[np.ix_(row_idx, istar_for_canonical[gene_idx])], dtype=np.float32)
        max_abs = float(np.max(np.abs(a - b)))
    return max_abs


def audit_inputs(root: Path, output_dir: Path, full_truth_check: bool = True) -> dict:
    gate_dir = output_dir / "gate6_0"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gene_orders = {cohort: load_gene_orders(root, cohort) for cohort in COHORT_SLIDES}
    index_orders = {
        cohort: (
            source_indices_for_canonical(morpho, canonical),
            source_indices_for_canonical(istar, canonical),
        )
        for cohort, (canonical, morpho, istar) in gene_orders.items()
    }
    h5_cache = {slide: h5ad_spots(root, slide) for slide in PROSTATE_SLIDES + KIDNEY_SLIDES}
    fully_checked: set[tuple[str, str]] = set()
    rows: list[dict] = []

    for task in build_tasks(root):
        canonical, morpho_genes, istar_genes = gene_orders[task.cohort]
        morpho_idx, istar_idx = index_orders[task.cohort]
        required = [
            task.morpho_dir / "pred_bag.npy",
            task.morpho_dir / "true_bag.npy",
            task.morpho_dir / "spot_ids.txt",
            task.istar_dir / "pred_bag.npy",
            task.istar_dir / "true_bag.npy",
            task.istar_dir / "locs.npy",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        row: dict[str, object] = {
            "kind": "benchmark_direction",
            "cohort": task.cohort,
            "train": task.train,
            "test": task.test,
            "task_id": task.task_id,
            "morpho_dir": str(task.morpho_dir),
            "istar_dir": str(task.istar_dir),
            "missing_files": "|".join(missing),
            "n_canonical_genes": len(canonical),
            "gene_sets_equal": set(morpho_genes) == set(istar_genes),
            "gene_order_equal": morpho_genes == istar_genes,
        }
        if missing:
            row["status"] = "FAIL_MISSING"
            rows.append(row)
            continue

        mp = np.load(task.morpho_dir / "pred_bag.npy", mmap_mode="r")
        mt = np.load(task.morpho_dir / "true_bag.npy", mmap_mode="r")
        ip = np.load(task.istar_dir / "pred_bag.npy", mmap_mode="r")
        it = np.load(task.istar_dir / "true_bag.npy", mmap_mode="r")
        morpho_spots = read_lines(task.morpho_dir / "spot_ids.txt")
        istar_locs = np.load(task.istar_dir / "locs.npy", mmap_mode="r")
        h5_spots, h5_coords = h5_cache[task.test]
        if task.cohort == "Prostate":
            istar_cache_locs_path = root / "benchmark" / "results" / "istar" / "data" / task.test / "locs.tsv"
        else:
            istar_cache_locs_path = (
                root / "benchmark" / "results" / "kidney_INT21" / "istar" / "data" / task.test / "locs.tsv"
            )
        istar_cache_locs = pd.read_csv(require_file(istar_cache_locs_path), sep="\t")
        istar_cache_spots = istar_cache_locs["SpotID"].astype(str).tolist()
        shapes_equal = mp.shape == mt.shape == ip.shape == it.shape
        expected_shape = (len(h5_spots), len(canonical))
        spots_equal = morpho_spots == h5_spots
        spot_unique = len(morpho_spots) == len(set(morpho_spots))
        locs_length_equal = istar_locs.shape[0] == len(h5_spots)
        expected_istar_grid = h5_coords[:, ::-1] / 16.0
        locs_grid_max_abs_diff = (
            float(np.max(np.abs(np.asarray(istar_locs, dtype=float) - expected_istar_grid)))
            if istar_locs.shape == expected_istar_grid.shape
            else float("inf")
        )
        cache_spots_equal = istar_cache_spots == h5_spots
        locs_content_equal = bool(locs_grid_max_abs_diff <= 1.0)
        key = (task.cohort, task.test)
        do_full = full_truth_check and key not in fully_checked
        truth_max_abs = aligned_truth_max_abs(
            task.morpho_dir / "true_bag.npy",
            task.istar_dir / "true_bag.npy",
            morpho_idx,
            istar_idx,
            full=do_full,
        )
        if do_full:
            fully_checked.add(key)
        passed = (
            shapes_equal
            and mp.shape == expected_shape
            and spots_equal
            and spot_unique
            and locs_length_equal
            and cache_spots_equal
            and locs_content_equal
            and truth_max_abs == 0.0
            and np.isfinite(np.asarray(mp[: min(8, mp.shape[0]), : min(64, mp.shape[1])])).all()
            and np.isfinite(np.asarray(ip[: min(8, ip.shape[0]), : min(64, ip.shape[1])])).all()
        )
        row.update(
            {
                "morpho_shape": f"{mp.shape[0]}x{mp.shape[1]}",
                "istar_shape": f"{ip.shape[0]}x{ip.shape[1]}",
                "expected_shape": f"{expected_shape[0]}x{expected_shape[1]}",
                "morpho_dtype": str(mp.dtype),
                "istar_dtype": str(ip.dtype),
                "morpho_spots_match_h5ad": spots_equal,
                "morpho_spots_unique": spot_unique,
                "istar_locs_length_match": locs_length_equal,
                "istar_cache_spots_match_h5ad": cache_spots_equal,
                "istar_locs_within_one_h5ad_grid_unit": locs_content_equal,
                "istar_locs_h5ad_grid_max_abs_diff": locs_grid_max_abs_diff,
                "truth_check": "full" if do_full else "deterministic_probe",
                "aligned_truth_max_abs_diff": truth_max_abs,
                "status": "PASS" if passed else "FAIL_ALIGNMENT",
            }
        )
        rows.append(row)

    canonical, _, _ = gene_orders["Prostate"]
    prior_reference: tuple[PriorInput, np.ndarray, list[str]] | None = None
    benchmark_int28 = next(
        task
        for task in build_tasks(root)
        if task.cohort == "Prostate" and task.train == "INT25" and task.test == "INT28"
    )
    _, benchmark_morpho_genes, _ = gene_orders["Prostate"]
    benchmark_idx = source_indices_for_canonical(benchmark_morpho_genes, canonical)
    for item in prior_inputs(root):
        required = [item.result_dir / "pred_bag.npy", item.result_dir / "true_bag.npy", item.gene_path]
        missing = [str(path) for path in required if not path.is_file()]
        row = {
            "kind": "prior_input",
            "cohort": "Prostate",
            "train": item.setting,
            "test": "INT28",
            "task_id": f"prior_{item.setting}_{item.initialization}".lower().replace("-", "_"),
            "morpho_dir": str(item.result_dir),
            "istar_dir": "",
            "missing_files": "|".join(missing),
            "n_canonical_genes": len(canonical),
        }
        if missing:
            row["status"] = "FAIL_MISSING"
            rows.append(row)
            continue
        genes = read_json_genes(item.gene_path)
        idx = np.asarray([genes.index(gene) for gene in canonical], dtype=np.int64)
        pred = np.load(item.result_dir / "pred_bag.npy", mmap_mode="r")
        true = np.load(item.result_dir / "true_bag.npy", mmap_mode="r")
        h5_spots, _ = h5_cache["INT28"]
        truth_max_abs = aligned_truth_max_abs(
            benchmark_int28.morpho_dir / "true_bag.npy",
            item.result_dir / "true_bag.npy",
            benchmark_idx,
            idx,
            full=full_truth_check and prior_reference is None,
        )
        truth_check = "benchmark_full" if full_truth_check and prior_reference is None else "benchmark_probe"
        if prior_reference is None:
            prior_reference = (item, true, genes)
        else:
            ref_item, _, ref_genes = prior_reference
            ref_idx = np.asarray([ref_genes.index(gene) for gene in canonical], dtype=np.int64)
            truth_max_abs = max(
                truth_max_abs,
                aligned_truth_max_abs(
                    ref_item.result_dir / "true_bag.npy",
                    item.result_dir / "true_bag.npy",
                    ref_idx,
                    idx,
                    full=full_truth_check,
                ),
            )
            truth_check = "benchmark_probe+prior_full" if full_truth_check else "deterministic_probes"
        passed = (
            pred.shape == true.shape == (len(h5_spots), len(canonical))
            and set(genes) == set(canonical)
            and truth_max_abs == 0.0
        )
        row.update(
            {
                "morpho_shape": f"{pred.shape[0]}x{pred.shape[1]}",
                "istar_shape": "",
                "expected_shape": f"{len(h5_spots)}x{len(canonical)}",
                "morpho_dtype": str(pred.dtype),
                "istar_dtype": "",
                "gene_sets_equal": set(genes) == set(canonical),
                "gene_order_equal": genes == canonical,
                "morpho_spots_match_h5ad": "truth-anchored",
                "morpho_spots_unique": "not_saved",
                "istar_locs_length_match": "not_applicable",
                "istar_cache_spots_match_h5ad": "not_applicable",
                "istar_locs_within_one_h5ad_grid_unit": "not_applicable",
                "istar_locs_h5ad_grid_max_abs_diff": "not_applicable",
                "truth_check": truth_check,
                "aligned_truth_max_abs_diff": truth_max_abs,
                "status": "PASS" if passed else "FAIL_ALIGNMENT",
            }
        )
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest_path = gate_dir / "phase6_input_alignment_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    benchmark = manifest[manifest["kind"].eq("benchmark_direction")]
    prior = manifest[manifest["kind"].eq("prior_input")]
    passed = bool(manifest["status"].eq("PASS").all())
    summary = {
        "created_at": now_iso(),
        "status": "PASS" if passed else "FAIL",
        "benchmark_directions_expected": 68,
        "benchmark_directions_passed": int(benchmark["status"].eq("PASS").sum()),
        "prostate_directions_passed": int(
            benchmark[benchmark["cohort"].eq("Prostate")]["status"].eq("PASS").sum()
        ),
        "kidney_directions_passed": int(
            benchmark[benchmark["cohort"].eq("Kidney")]["status"].eq("PASS").sum()
        ),
        "prior_inputs_expected": 4,
        "prior_inputs_passed": int(prior["status"].eq("PASS").sum()),
        "full_truth_checks": len(fully_checked) if full_truth_check else 0,
        "prostate_gene_order_requires_permutation": gene_orders["Prostate"][0] != gene_orders["Prostate"][1],
        "kidney_gene_order_requires_permutation": gene_orders["Kidney"][0] != gene_orders["Kidney"][1],
        "canonical_gene_panel_sha256": sha256_file(canonical_gene_path(root, "Prostate")),
        "manifest": str(manifest_path),
    }
    write_json(gate_dir / "phase6_input_alignment_summary.json", summary)
    if not passed:
        failed = manifest[manifest["status"].ne("PASS")][["task_id", "status", "missing_files"]]
        raise RuntimeError(f"Gate 6.0 failed:\n{failed.to_string(index=False)}")
    print(f"[Gate 6.0 PASS] {len(benchmark)}/68 directions and {len(prior)}/4 prior inputs aligned")
    return summary


def parse_gmt(path: Path, canonical: list[str], min_genes: int = 10) -> list[dict]:
    canonical_set = set(canonical)
    programmes: list[dict] = []
    for line_number, line in enumerate(require_file(path).read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            raise ValueError(f"Malformed GMT line {line_number}: {path}")
        name, description, *raw_genes = fields
        genes = list(dict.fromkeys(gene for gene in raw_genes if gene in canonical_set))
        if len(genes) >= min_genes:
            programmes.append(
                {
                    "name": name,
                    "description": description,
                    "genes": genes,
                    "n_source_genes": len(list(dict.fromkeys(raw_genes))),
                    "n_panel_genes": len(genes),
                }
            )
    if not programmes:
        raise ValueError(f"No programmes retained from {path}")
    return programmes


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] <= k:
        raise ValueError(f"Cannot build k={k} graph from coordinates {coords.shape}")
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    _, candidates = tree.query(coords, k=k + 1)
    neighbors = np.empty((coords.shape[0], k), dtype=np.int64)
    for row_index, row in enumerate(np.atleast_2d(candidates)):
        selected = [int(value) for value in row if int(value) != row_index][:k]
        if len(selected) != k:
            raise ValueError(f"Could not identify {k} non-self neighbors for row {row_index}")
        neighbors[row_index] = selected
    return neighbors


def moran_values(values: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[0] != neighbors.shape[0]:
        raise ValueError(f"Moran row mismatch: values={matrix.shape}, neighbors={neighbors.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Moran input contains non-finite values")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    lag = np.zeros_like(centered)
    for column in range(neighbors.shape[1]):
        lag += centered[neighbors[:, column]]
    lag /= float(neighbors.shape[1])
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered * lag, axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def truth_gene_statistics(
    matrix_path: Path,
    native_for_canonical: np.ndarray,
    neighbors: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.load(require_file(matrix_path), mmap_mode="r")
    n_genes = len(native_for_canonical)
    means = np.empty(n_genes, dtype=np.float64)
    variances = np.empty(n_genes, dtype=np.float64)
    moran = np.empty(n_genes, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        values = np.asarray(matrix[:, native_for_canonical[start:stop]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite truth values in {matrix_path}, columns {start}:{stop}")
        means[start:stop] = values.mean(axis=0)
        variances[start:stop] = values.var(axis=0)
        moran[start:stop] = moran_values(values, neighbors)
    return means, variances, moran


def stable_top_indices(score: np.ndarray, eligible: np.ndarray, n: int) -> np.ndarray:
    candidates = np.flatnonzero(eligible & np.isfinite(score))
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} eligible genes for a requested panel of {n}")
    order = np.argsort(-score[candidates], kind="stable")
    return candidates[order[:n]]


def freeze_analysis_spec(root: Path, output_dir: Path, gmt_path: Path) -> dict:
    audit_path = output_dir / "gate6_0" / "phase6_input_alignment_summary.json"
    audit = json.loads(require_file(audit_path).read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("Gate 6.0 must pass before Gate 6.1 can be frozen")
    gmt_hash = sha256_file(gmt_path)
    if gmt_hash != EXPECTED_GMT_SHA256:
        raise ValueError(f"Unexpected Hallmark GMT SHA256: {gmt_hash}")

    gate_dir = output_dir / "gate6_1"
    panel_dir = gate_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    canonical = read_lines(canonical_gene_path(root, "Prostate"))
    programmes = parse_gmt(gmt_path, canonical, min_genes=10)
    programme_payload = {
        "source": GMT_RELEASE,
        "source_url": GMT_SOURCE_URL,
        "sha256": gmt_hash,
        "minimum_intersection_genes": 10,
        "programmes": programmes,
    }
    write_json(gate_dir / "programmes.json", programme_payload)
    programme_rows = [
        {
            "programme": item["name"],
            "description": item["description"],
            "n_source_genes": item["n_source_genes"],
            "n_panel_genes": item["n_panel_genes"],
            "coverage": item["n_panel_genes"] / item["n_source_genes"],
        }
        for item in programmes
    ]
    pd.DataFrame(programme_rows).to_csv(gate_dir / "programme_manifest.csv", index=False)

    programme_union = {gene for item in programmes for gene in item["genes"]}
    canonical_index = {gene: index for index, gene in enumerate(canonical)}
    union_mask = np.asarray([gene in programme_union for gene in canonical], dtype=bool)
    tasks = build_tasks(root)
    panel_records: list[dict] = []
    panel_paths: dict[str, dict[str, str]] = {}

    for cohort, slides in COHORT_SLIDES.items():
        cohort_canonical, morpho_genes, _ = load_gene_orders(root, cohort)
        if cohort_canonical != canonical:
            raise ValueError("Prostate and kidney canonical gene panels must be identical for Phase 6")
        morpho_for_canonical = source_indices_for_canonical(morpho_genes, canonical)
        for slide in slides:
            reference = next(task for task in tasks if task.cohort == cohort and task.test == slide)
            spot_ids, coords = h5ad_spots(root, slide)
            if read_lines(reference.morpho_dir / "spot_ids.txt") != spot_ids:
                raise ValueError(f"Reference spot order mismatch for {cohort} {slide}")
            neighbors = knn_indices(coords, k=6)
            means, variances, moran = truth_gene_statistics(
                reference.morpho_dir / "true_bag.npy",
                morpho_for_canonical,
                neighbors,
            )
            evaluable = (variances > 1e-12) & np.isfinite(moran)
            moran_idx = stable_top_indices(moran, evaluable, n=500)
            coexpr_idx = stable_top_indices(variances, evaluable & union_mask, n=500)
            stem = f"{cohort.lower()}_{slide}"
            moran_path = panel_dir / f"{stem}_moran_panel.csv"
            coexpr_path = panel_dir / f"{stem}_coexpression_panel.csv"
            pd.DataFrame(
                {
                    "gene": [canonical[index] for index in moran_idx],
                    "canonical_index": moran_idx,
                    "truth_mean": means[moran_idx],
                    "truth_variance": variances[moran_idx],
                    "truth_moran_k6": moran[moran_idx],
                    "rank": np.arange(1, len(moran_idx) + 1),
                }
            ).to_csv(moran_path, index=False)
            pd.DataFrame(
                {
                    "gene": [canonical[index] for index in coexpr_idx],
                    "canonical_index": coexpr_idx,
                    "truth_mean": means[coexpr_idx],
                    "truth_variance": variances[coexpr_idx],
                    "truth_moran_k6": moran[coexpr_idx],
                    "rank": np.arange(1, len(coexpr_idx) + 1),
                }
            ).to_csv(coexpr_path, index=False)
            panel_paths[f"{cohort}:{slide}"] = {
                "moran": str(moran_path),
                "coexpression": str(coexpr_path),
            }
            panel_records.append(
                {
                    "cohort": cohort,
                    "test_section": slide,
                    "reference_direction": reference.task_id,
                    "n_spots": len(spot_ids),
                    "moran_panel_n": len(moran_idx),
                    "moran_panel_min_truth_moran": float(np.min(moran[moran_idx])),
                    "coexpression_panel_n": len(coexpr_idx),
                    "coexpression_panel_hallmark_union": True,
                    "moran_panel_sha256": sha256_file(moran_path),
                    "coexpression_panel_sha256": sha256_file(coexpr_path),
                }
            )

    pd.DataFrame(panel_records).to_csv(gate_dir / "panel_manifest.csv", index=False)
    spec = {
        "created_at": now_iso(),
        "status": "FROZEN",
        "analysis_version": "phase6_v1",
        "programme_source": programme_payload | {"programmes": len(programmes)},
        "input_scale": "existing observed-location count-scale matrices; no additional transform",
        "programme_score": {
            "operation": "per-gene z-score across spots, followed by unweighted programme mean",
            "constant_prediction_rule": "z-score set to zero; gene is not dropped",
            "minimum_programme_genes": 10,
            "primary_metric": "Pearson concordance of predicted and measured programme scores across spots",
        },
        "spatial_graph": {
            "coordinates": "original H5AD obsm/spatial",
            "distance": "Euclidean",
            "graph": "directed k-nearest neighbors with row-standardized weights",
            "primary_k": 6,
            "sensitivity_k": [4, 8],
        },
        "moran_panel": {
            "n_genes": 500,
            "selection": "top truth Moran's I at k=6 among finite nonconstant genes",
            "primary_metrics": ["predicted-vs-measured Pearson concordance", "absolute error"],
        },
        "coexpression_panel": {
            "n_genes": 500,
            "selection": "top truth variance within the frozen Hallmark union",
            "primary_metrics": ["upper-triangle rank concordance", "normalized matrix error"],
        },
        "statistical_unit": (
            "held-out test section; compute same-direction paired advantages first, "
            "then take the median across train directions"
        ),
        "section_direction_thresholds": {"Prostate": "3/4", "Kidney": "6/8"},
        "gate_logic": {
            "GO-MAIN": "programme and Moran endpoints pass in both tissues, with at least one matched prior endpoint positive in both settings",
            "GO-MOLECULAR": "programme and coexpression pass but Moran does not",
            "PARTIAL": "at least one endpoint or tissue is stable, but GO-MAIN/GO-MOLECULAR is not met",
            "STOP-STRUCTURE": "no structure endpoint is stable",
        },
        "panels": panel_paths,
    }
    spec_path = gate_dir / "analysis_spec.json"
    write_json(spec_path, spec)
    (gate_dir / "analysis_spec.sha256").write_text(sha256_file(spec_path) + "\n", encoding="ascii")
    print(
        f"[Gate 6.1 FROZEN] {len(programmes)} Hallmark programmes, "
        f"{len(panel_records)} target-section panels"
    )
    return spec


def zscore_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Expression matrix contains non-finite values")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    valid = scale > 1e-12
    standardized = np.zeros_like(centered)
    standardized[:, valid] = centered[:, valid] / scale[valid]
    return standardized, valid


def vector_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Pearson matrix mismatch: {a.shape}, {b.shape}")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    numerator = np.sum(a * b, axis=0)
    denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    result = np.full(a.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def scalar_pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    return float(vector_pearson(a[valid, None], b[valid, None])[0])


def scalar_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    return scalar_pearson(rankdata(a[valid]), rankdata(b[valid]))


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    standardized, _ = zscore_columns(values)
    return (standardized.T @ standardized) / float(standardized.shape[0])


def matrix_columns(path: Path, gene_to_native: dict[str, int], genes: list[str]) -> np.ndarray:
    matrix = np.load(require_file(path), mmap_mode="r")
    try:
        indices = np.asarray([gene_to_native[gene] for gene in genes], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Missing requested gene {exc.args[0]} in {path}") from exc
    values = np.asarray(matrix[:, indices], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {path}")
    return values


@dataclass
class TargetCache:
    cohort: str
    test: str
    coords: np.ndarray
    neighbors: dict[int, np.ndarray]
    programme_names: list[str]
    programme_members: list[np.ndarray]
    programme_gene_counts: list[int]
    programme_union: list[str]
    truth_programme_scores: np.ndarray
    moran_genes: list[str]
    truth_moran: dict[int, np.ndarray]
    coexpression_genes: list[str]
    truth_coexpression: np.ndarray
    truth_coexpression_upper: np.ndarray


def load_programmes(output_dir: Path) -> list[dict]:
    payload = json.loads(
        require_file(output_dir / "gate6_1" / "programmes.json").read_text(encoding="utf-8")
    )
    programmes = payload.get("programmes")
    if not isinstance(programmes, list) or not programmes:
        raise ValueError("Frozen programme list is missing")
    return programmes


def load_panel_genes(output_dir: Path, cohort: str, test: str, panel: str) -> list[str]:
    path = output_dir / "gate6_1" / "panels" / f"{cohort.lower()}_{test}_{panel}_panel.csv"
    frame = pd.read_csv(require_file(path))
    genes = frame["gene"].astype(str).tolist()
    if len(genes) != 500 or len(set(genes)) != len(genes):
        raise ValueError(f"Invalid frozen panel: {path}")
    return genes


def programme_score_matrix(
    values: np.ndarray,
    programme_members: list[np.ndarray],
) -> np.ndarray:
    standardized, _ = zscore_columns(values)
    scores = np.full((standardized.shape[0], len(programme_members)), np.nan, dtype=np.float64)
    for index, members in enumerate(programme_members):
        if len(members) >= 10:
            scores[:, index] = standardized[:, members].mean(axis=1)
    return scores


def make_target_cache(
    root: Path,
    output_dir: Path,
    cohort: str,
    test: str,
    programmes: list[dict],
    tasks: list[Task],
) -> TargetCache:
    canonical, morpho_genes, _ = load_gene_orders(root, cohort)
    canonical_set = set(canonical)
    morpho_map = {gene: index for index, gene in enumerate(morpho_genes)}
    reference = next(task for task in tasks if task.cohort == cohort and task.test == test)
    spot_ids, coords = h5ad_spots(root, test)
    if read_lines(reference.morpho_dir / "spot_ids.txt") != spot_ids:
        raise ValueError(f"Target-cache spot mismatch for {cohort} {test}")
    neighbors = {k: knn_indices(coords, k) for k in (4, 6, 8)}

    programme_union_set = {value for item in programmes for value in item["genes"]}
    programme_union = [gene for gene in canonical if gene in programme_union_set]
    union_index = {gene: index for index, gene in enumerate(programme_union)}
    truth_programme_values = matrix_columns(
        reference.morpho_dir / "true_bag.npy",
        morpho_map,
        programme_union,
    )
    _, truth_programme_valid = zscore_columns(truth_programme_values)
    programme_names: list[str] = []
    programme_members: list[np.ndarray] = []
    programme_gene_counts: list[int] = []
    for item in programmes:
        members = np.asarray(
            [union_index[gene] for gene in item["genes"] if gene in canonical_set and truth_programme_valid[union_index[gene]]],
            dtype=np.int64,
        )
        programme_names.append(str(item["name"]))
        programme_members.append(members)
        programme_gene_counts.append(len(members))
    truth_programme_scores = programme_score_matrix(truth_programme_values, programme_members)

    moran_genes = load_panel_genes(output_dir, cohort, test, "moran")
    truth_moran_values = matrix_columns(reference.morpho_dir / "true_bag.npy", morpho_map, moran_genes)
    truth_moran = {k: moran_values(truth_moran_values, graph) for k, graph in neighbors.items()}
    for k, values in truth_moran.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Frozen Moran panel is not fully evaluable for {cohort} {test}, k={k}")

    coexpression_genes = load_panel_genes(output_dir, cohort, test, "coexpression")
    truth_coexpression_values = matrix_columns(
        reference.morpho_dir / "true_bag.npy",
        morpho_map,
        coexpression_genes,
    )
    truth_coexpression = correlation_matrix(truth_coexpression_values)
    upper_index = np.triu_indices(len(coexpression_genes), k=1)
    truth_coexpression_upper = truth_coexpression[upper_index]
    return TargetCache(
        cohort=cohort,
        test=test,
        coords=coords,
        neighbors=neighbors,
        programme_names=programme_names,
        programme_members=programme_members,
        programme_gene_counts=programme_gene_counts,
        programme_union=programme_union,
        truth_programme_scores=truth_programme_scores,
        moran_genes=moran_genes,
        truth_moran=truth_moran,
        coexpression_genes=coexpression_genes,
        truth_coexpression=truth_coexpression,
        truth_coexpression_upper=truth_coexpression_upper,
    )


def evaluate_prediction(
    prediction_path: Path,
    gene_to_native: dict[str, int],
    cache: TargetCache,
) -> tuple[list[dict], list[dict], dict, np.ndarray]:
    programme_values = matrix_columns(prediction_path, gene_to_native, cache.programme_union)
    predicted_programme_scores = programme_score_matrix(programme_values, cache.programme_members)
    programme_corr = vector_pearson(predicted_programme_scores, cache.truth_programme_scores)
    programme_rows = [
        {
            "programme": name,
            "n_genes": n_genes,
            "concordance": float(value) if np.isfinite(value) else float("nan"),
        }
        for name, n_genes, value in zip(
            cache.programme_names,
            cache.programme_gene_counts,
            programme_corr,
            strict=True,
        )
    ]

    predicted_moran_values = matrix_columns(prediction_path, gene_to_native, cache.moran_genes)
    moran_rows: list[dict] = []
    for k, graph in cache.neighbors.items():
        predicted = np.nan_to_num(moran_values(predicted_moran_values, graph), nan=0.0)
        truth = cache.truth_moran[k]
        delta = predicted - truth
        moran_rows.append(
            {
                "k": k,
                "n_genes": len(truth),
                "concordance": scalar_pearson(predicted, truth),
                "rank_concordance": scalar_spearman(predicted, truth),
                "mae": float(np.mean(np.abs(delta))),
                "bias": float(np.mean(delta)),
                "truth_mean": float(np.mean(truth)),
                "predicted_mean": float(np.mean(predicted)),
                "fraction_predicted_above_truth": float(np.mean(predicted > truth)),
            }
        )

    coexpression_values = matrix_columns(prediction_path, gene_to_native, cache.coexpression_genes)
    predicted_coexpression = correlation_matrix(coexpression_values)
    upper_index = np.triu_indices(len(cache.coexpression_genes), k=1)
    predicted_upper = predicted_coexpression[upper_index]
    truth_upper = cache.truth_coexpression_upper
    denominator = np.linalg.norm(cache.truth_coexpression)
    coexpression_row = {
        "n_genes": len(cache.coexpression_genes),
        "n_pairs": len(truth_upper),
        "rank_concordance": scalar_spearman(predicted_upper, truth_upper),
        "pearson_concordance": scalar_pearson(predicted_upper, truth_upper),
        "normalized_matrix_error": float(
            np.linalg.norm(predicted_coexpression - cache.truth_coexpression) / denominator
        ),
    }
    return programme_rows, moran_rows, coexpression_row, predicted_upper


def append_evaluation_rows(
    programme_output: list[dict],
    moran_output: list[dict],
    coexpression_output: list[dict],
    prefix: dict,
    evaluation: tuple[list[dict], list[dict], dict, np.ndarray],
) -> None:
    programme_rows, moran_rows, coexpression_row, _ = evaluation
    programme_output.extend(prefix | row for row in programme_rows)
    moran_output.extend(prefix | row for row in moran_rows)
    coexpression_output.append(prefix | coexpression_row)


def run_task_set(
    root: Path,
    output_dir: Path,
    tasks_to_run: list[Task],
    destination: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(output_dir / "gate6_1" / "analysis_spec.json")
    programmes = load_programmes(output_dir)
    all_tasks = build_tasks(root)
    caches: dict[tuple[str, str], TargetCache] = {}
    gene_maps: dict[tuple[str, str], dict[str, int]] = {}
    for cohort in COHORT_SLIDES:
        canonical, morpho_genes, istar_genes = load_gene_orders(root, cohort)
        gene_maps[(cohort, "Morpho-FM")] = {gene: index for index, gene in enumerate(morpho_genes)}
        gene_maps[(cohort, "iStar")] = {gene: index for index, gene in enumerate(istar_genes)}
        if set(canonical) != set(morpho_genes) or set(canonical) != set(istar_genes):
            raise ValueError(f"Gene-set mismatch while preparing {cohort}")

    programme_output: list[dict] = []
    moran_output: list[dict] = []
    coexpression_output: list[dict] = []
    for index, task in enumerate(tasks_to_run, start=1):
        cache_key = (task.cohort, task.test)
        if cache_key not in caches:
            caches[cache_key] = make_target_cache(
                root,
                output_dir,
                task.cohort,
                task.test,
                programmes,
                all_tasks,
            )
        cache = caches[cache_key]
        for method, result_dir in (("Morpho-FM", task.morpho_dir), ("iStar", task.istar_dir)):
            prefix = {
                "cohort": task.cohort,
                "train": task.train,
                "test": task.test,
                "task_id": task.task_id,
                "method": method,
            }
            evaluation = evaluate_prediction(
                result_dir / "pred_bag.npy",
                gene_maps[(task.cohort, method)],
                cache,
            )
            append_evaluation_rows(
                programme_output,
                moran_output,
                coexpression_output,
                prefix,
                evaluation,
            )
        print(f"[{index}/{len(tasks_to_run)}] {task.cohort} {task.train}->{task.test}")

    destination.mkdir(parents=True, exist_ok=True)
    programme_frame = pd.DataFrame(programme_output)
    moran_frame = pd.DataFrame(moran_output)
    coexpression_frame = pd.DataFrame(coexpression_output)
    programme_frame.to_csv(destination / "programme_metrics.csv", index=False)
    moran_frame.to_csv(destination / "moran_metrics.csv", index=False)
    coexpression_frame.to_csv(destination / "coexpression_metrics.csv", index=False)
    return programme_frame, moran_frame, coexpression_frame


def run_pilot(root: Path, output_dir: Path) -> dict:
    task = next(
        item
        for item in build_tasks(root)
        if item.cohort == "Prostate" and item.train == "INT28" and item.test == "INT25"
    )
    programme, moran, coexpression = run_task_set(
        root,
        output_dir,
        [task],
        output_dir / "pilot",
    )
    checks = {
        "programme_finite_fraction": float(np.isfinite(programme["concordance"]).mean()),
        "programme_bounds": bool(programme["concordance"].dropna().between(-1.0, 1.0).all()),
        "moran_concordance_bounds": bool(moran["concordance"].dropna().between(-1.0, 1.0).all()),
        "moran_mae_nonnegative": bool((moran["mae"] >= 0).all()),
        "coexpression_rank_bounds": bool(
            coexpression["rank_concordance"].dropna().between(-1.0, 1.0).all()
        ),
        "coexpression_error_nonnegative": bool((coexpression["normalized_matrix_error"] >= 0).all()),
    }
    checks["status"] = "PASS" if all(value for key, value in checks.items() if key != "programme_finite_fraction") and checks["programme_finite_fraction"] >= 0.9 else "FAIL"
    checks["created_at"] = now_iso()
    checks["task"] = task.task_id
    write_json(output_dir / "pilot" / "pilot_validation.json", checks)
    if checks["status"] != "PASS":
        raise RuntimeError(f"Phase 6 pilot failed: {checks}")
    print(f"[Pilot PASS] {task.train}->{task.test}")
    return checks


def run_prior_analysis(root: Path, output_dir: Path, destination: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    programmes = load_programmes(output_dir)
    tasks = build_tasks(root)
    cache = make_target_cache(root, output_dir, "Prostate", "INT28", programmes, tasks)
    programme_output: list[dict] = []
    moran_output: list[dict] = []
    coexpression_output: list[dict] = []
    for item in prior_inputs(root):
        genes = read_json_genes(item.gene_path)
        gene_map = {gene: index for index, gene in enumerate(genes)}
        prefix = {
            "cohort": "Prostate",
            "test": "INT28",
            "setting": item.setting,
            "initialization": item.initialization,
            "seed": 42,
        }
        evaluation = evaluate_prediction(item.result_dir / "pred_bag.npy", gene_map, cache)
        append_evaluation_rows(
            programme_output,
            moran_output,
            coexpression_output,
            prefix,
            evaluation,
        )
    destination.mkdir(parents=True, exist_ok=True)
    programme_frame = pd.DataFrame(programme_output)
    moran_frame = pd.DataFrame(moran_output)
    coexpression_frame = pd.DataFrame(coexpression_output)
    programme_frame.to_csv(destination / "prior_programme_metrics.csv", index=False)
    moran_frame.to_csv(destination / "prior_moran_metrics.csv", index=False)
    coexpression_frame.to_csv(destination / "prior_coexpression_metrics.csv", index=False)
    return programme_frame, moran_frame, coexpression_frame


def build_direction_summary(
    programme: pd.DataFrame,
    moran: pd.DataFrame,
    coexpression: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["cohort", "train", "test", "task_id", "method"]
    programme_summary = (
        programme.groupby(keys, as_index=False)["concordance"]
        .median()
        .rename(columns={"concordance": "programme_concordance"})
    )
    moran_main = moran[moran["k"].eq(6)][
        keys + ["concordance", "rank_concordance", "mae", "bias", "fraction_predicted_above_truth"]
    ].rename(
        columns={
            "concordance": "moran_concordance",
            "rank_concordance": "moran_rank_concordance",
            "mae": "moran_mae",
            "bias": "moran_bias",
            "fraction_predicted_above_truth": "moran_fraction_predicted_above_truth",
        }
    )
    coexpression_summary = coexpression[
        keys + ["rank_concordance", "pearson_concordance", "normalized_matrix_error"]
    ].rename(
        columns={
            "rank_concordance": "coexpression_rank_concordance",
            "pearson_concordance": "coexpression_pearson_concordance",
            "normalized_matrix_error": "coexpression_normalized_error",
        }
    )
    return programme_summary.merge(moran_main, on=keys, validate="one_to_one").merge(
        coexpression_summary,
        on=keys,
        validate="one_to_one",
    )


SECTION_METRICS = {
    "programme": ("programme_concordance", True),
    "moran_concordance": ("moran_concordance", True),
    "moran_mae": ("moran_mae", False),
    "coexpression_rank": ("coexpression_rank_concordance", True),
    "coexpression_error": ("coexpression_normalized_error", False),
}


def build_section_summary(direction: pd.DataFrame) -> pd.DataFrame:
    direction_keys = ["cohort", "train", "test", "task_id"]
    numeric = [value for value, _ in SECTION_METRICS.values()]
    morpho = direction[direction["method"].eq("Morpho-FM")][direction_keys + numeric].rename(
        columns={column: f"{column}_morpho" for column in numeric}
    )
    istar = direction[direction["method"].eq("iStar")][direction_keys + numeric].rename(
        columns={column: f"{column}_istar" for column in numeric}
    )
    paired = morpho.merge(istar, on=direction_keys, validate="one_to_one")
    selected_columns: list[str] = []
    rename: dict[str, str] = {}
    for label, (column, higher_is_better) in SECTION_METRICS.items():
        left = f"{column}_morpho"
        right = f"{column}_istar"
        advantage = f"{label}_advantage"
        paired[advantage] = paired[left] - paired[right] if higher_is_better else paired[right] - paired[left]
        selected_columns.extend([left, right, advantage])
        rename[left] = f"{label}_morpho"
        rename[right] = f"{label}_istar"
    section = paired.groupby(["cohort", "test"], as_index=False)[selected_columns].median()
    return section.rename(columns=rename).sort_values(["cohort", "test"]).reset_index(drop=True)


def build_moran_section_by_k(moran: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "train", "test", "task_id", "k"]
    morpho = moran[moran["method"].eq("Morpho-FM")][keys + ["concordance", "mae"]].rename(
        columns={"concordance": "morpho_concordance", "mae": "morpho_mae"}
    )
    istar = moran[moran["method"].eq("iStar")][keys + ["concordance", "mae"]].rename(
        columns={"concordance": "istar_concordance", "mae": "istar_mae"}
    )
    paired = morpho.merge(istar, on=keys, validate="one_to_one")
    paired["concordance_advantage"] = paired["morpho_concordance"] - paired["istar_concordance"]
    paired["mae_advantage"] = paired["istar_mae"] - paired["morpho_mae"]
    columns = [
        "morpho_concordance",
        "istar_concordance",
        "concordance_advantage",
        "morpho_mae",
        "istar_mae",
        "mae_advantage",
    ]
    return (
        paired.groupby(["cohort", "test", "k"], as_index=False)[columns]
        .median()
        .sort_values(["cohort", "test", "k"])
        .reset_index(drop=True)
    )


def leave_one_out_positive(values: np.ndarray) -> bool:
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 2 or not np.isfinite(data).all():
        return False
    return all(float(np.median(np.delete(data, index))) > 0.0 for index in range(len(data)))


def endpoint_gate(section: pd.DataFrame, advantage_columns: list[str]) -> dict:
    details: dict[str, dict] = {}
    overall_pass = True
    for cohort, threshold in (("Prostate", 3), ("Kidney", 6)):
        subset = section[section["cohort"].eq(cohort)]
        supports = np.ones(len(subset), dtype=bool)
        medians: dict[str, float] = {}
        loo: dict[str, bool] = {}
        for column in advantage_columns:
            values = subset[column].to_numpy(dtype=float)
            supports &= values > 0.0
            medians[column] = float(np.median(values))
            loo[column] = leave_one_out_positive(values)
        cohort_pass = (
            int(supports.sum()) >= threshold
            and all(value > 0.0 for value in medians.values())
            and all(loo.values())
        )
        details[cohort] = {
            "n_sections": len(subset),
            "n_sections_jointly_positive": int(supports.sum()),
            "required_positive": threshold,
            "median_advantages": medians,
            "leave_one_section_out_positive": loo,
            "pass": cohort_pass,
        }
        overall_pass &= cohort_pass
    return {"pass": overall_pass, "cohorts": details}


def bootstrap_section_summary(section: pd.DataFrame, replicates: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    advantage_columns = [column for column in section.columns if column.endswith("_advantage")]
    for cohort, group in section.groupby("cohort"):
        for column in advantage_columns:
            values = group[column].to_numpy(dtype=np.float64)
            draws = rng.choice(values, size=(replicates, len(values)), replace=True)
            estimates = np.median(draws, axis=1)
            rows.append(
                {
                    "cohort": cohort,
                    "metric": column.removesuffix("_advantage"),
                    "n_sections": len(values),
                    "median_advantage": float(np.median(values)),
                    "bootstrap_ci_low": float(np.quantile(estimates, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_replicates": replicates,
                    "seed": 42,
                }
            )
    return pd.DataFrame(rows)


def build_prior_summary(
    programme: pd.DataFrame,
    moran: pd.DataFrame,
    coexpression: pd.DataFrame,
) -> pd.DataFrame:
    programme_summary = (
        programme.groupby(["setting", "initialization"], as_index=False)["concordance"]
        .median()
        .rename(columns={"concordance": "programme_concordance"})
    )
    programme_pretrained = programme[programme["initialization"].eq("Pretrained")][
        ["setting", "programme", "concordance"]
    ].rename(columns={"concordance": "programme_pretrained"})
    programme_random = programme[programme["initialization"].eq("Random")][
        ["setting", "programme", "concordance"]
    ].rename(columns={"concordance": "programme_random"})
    programme_pairs = programme_pretrained.merge(
        programme_random,
        on=["setting", "programme"],
        validate="one_to_one",
    )
    programme_pairs["programme_advantage"] = (
        programme_pairs["programme_pretrained"] - programme_pairs["programme_random"]
    )
    programme_paired_advantage = programme_pairs.groupby("setting")["programme_advantage"].median()
    moran_summary = moran[moran["k"].eq(6)][
        ["setting", "initialization", "concordance", "mae"]
    ].rename(columns={"concordance": "moran_concordance", "mae": "moran_mae"})
    coexpression_summary = coexpression[
        ["setting", "initialization", "rank_concordance", "normalized_matrix_error"]
    ].rename(
        columns={
            "rank_concordance": "coexpression_rank_concordance",
            "normalized_matrix_error": "coexpression_normalized_error",
        }
    )
    merged = programme_summary.merge(
        moran_summary,
        on=["setting", "initialization"],
        validate="one_to_one",
    ).merge(
        coexpression_summary,
        on=["setting", "initialization"],
        validate="one_to_one",
    )
    rows: list[dict] = []
    for setting, group in merged.groupby("setting"):
        values = group.set_index("initialization")
        row: dict[str, object] = {"setting": setting, "seed": 42}
        for label, column, higher_is_better in (
            ("programme", "programme_concordance", True),
            ("moran_concordance", "moran_concordance", True),
            ("moran_mae", "moran_mae", False),
            ("coexpression_rank", "coexpression_rank_concordance", True),
            ("coexpression_error", "coexpression_normalized_error", False),
        ):
            pretrained = float(values.loc["Pretrained", column])
            random = float(values.loc["Random", column])
            row[f"{label}_pretrained"] = pretrained
            row[f"{label}_random"] = random
            if label == "programme":
                row[f"{label}_advantage"] = float(programme_paired_advantage.loc[setting])
            else:
                row[f"{label}_advantage"] = pretrained - random if higher_is_better else random - pretrained
        rows.append(row)
    return pd.DataFrame(rows).sort_values("setting").reset_index(drop=True)


def decide_gate(
    section: pd.DataFrame,
    moran_by_k: pd.DataFrame,
    prior: pd.DataFrame,
) -> dict:
    programme_gate = endpoint_gate(section, ["programme_advantage"])
    moran_gate = endpoint_gate(section, ["moran_concordance_advantage", "moran_mae_advantage"])
    coexpression_gate = endpoint_gate(
        section,
        ["coexpression_rank_advantage", "coexpression_error_advantage"],
    )
    moran_robust = True
    moran_robust_details: dict[str, dict] = {}
    for cohort in COHORT_SLIDES:
        cohort_details: dict[str, dict] = {}
        for k in (4, 8):
            subset = moran_by_k[moran_by_k["cohort"].eq(cohort) & moran_by_k["k"].eq(k)]
            corr_median = float(subset["concordance_advantage"].median())
            mae_median = float(subset["mae_advantage"].median())
            passed = corr_median > 0.0 and mae_median > 0.0
            cohort_details[str(k)] = {
                "median_concordance_advantage": corr_median,
                "median_mae_advantage": mae_median,
                "pass": passed,
            }
            moran_robust &= passed
        moran_robust_details[cohort] = cohort_details
    moran_gate["k4_k8_robust"] = moran_robust
    moran_gate["sensitivity"] = moran_robust_details
    moran_gate["pass"] = bool(moran_gate["pass"] and moran_robust)

    prior_programme = bool((prior["programme_advantage"] > 0.0).all())
    prior_moran = bool(
        (prior["moran_concordance_advantage"] > 0.0).all()
        and (prior["moran_mae_advantage"] > 0.0).all()
    )
    prior_coexpression = bool(
        (prior["coexpression_rank_advantage"] > 0.0).all()
        and (prior["coexpression_error_advantage"] > 0.0).all()
    )
    prior_gate = {
        "single_seed": True,
        "programme_pass_both_settings": prior_programme,
        "moran_pass_both_settings": prior_moran,
        "coexpression_pass_both_settings": prior_coexpression,
        "any_primary_pass_both_settings": prior_programme or prior_moran,
    }

    if programme_gate["pass"] and moran_gate["pass"] and prior_gate["any_primary_pass_both_settings"]:
        decision = "GO-MAIN"
    elif programme_gate["pass"] and coexpression_gate["pass"] and not moran_gate["pass"]:
        decision = "GO-MOLECULAR"
    else:
        any_tissue_signal = any(
            detail["pass"]
            for gate in (programme_gate, moran_gate, coexpression_gate)
            for detail in gate["cohorts"].values()
        )
        decision = "PARTIAL" if any_tissue_signal else "STOP-STRUCTURE"
    return {
        "created_at": now_iso(),
        "decision": decision,
        "aggregation_order": (
            "compute Morpho-FM minus iStar advantage within each matched train-to-test direction, "
            "then take the median advantage within each held-out test section"
        ),
        "programme": programme_gate,
        "moran": moran_gate,
        "coexpression": coexpression_gate,
        "prior": prior_gate,
        "interpretation_boundary": "Pretrained/random results are matched single-seed observations.",
    }


def edge_metrics(predicted: np.ndarray, truth: np.ndarray, densities: tuple[float, ...] = (0.01, 0.05, 0.10)) -> list[dict]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError("Edge vectors must have the same shape")
    rows: list[dict] = []
    for density in densities:
        n_edges = max(1, int(round(density * len(truth))))
        truth_top = np.argpartition(np.abs(truth), -n_edges)[-n_edges:]
        predicted_top = np.argpartition(np.abs(predicted), -n_edges)[-n_edges:]
        recovery = len(np.intersect1d(truth_top, predicted_top, assume_unique=False)) / float(n_edges)
        sign_preservation = float(np.mean(np.sign(predicted[truth_top]) == np.sign(truth[truth_top])))
        rows.append(
            {
                "edge_density": density,
                "n_truth_edges": n_edges,
                "top_edge_recovery": recovery,
                "truth_top_edge_sign_preservation": sign_preservation,
            }
        )
    return rows


def run_conditional_edges(root: Path, output_dir: Path, destination: Path) -> pd.DataFrame:
    programmes = load_programmes(output_dir)
    tasks = build_tasks(root)
    caches: dict[tuple[str, str], TargetCache] = {}
    gene_maps: dict[tuple[str, str], dict[str, int]] = {}
    for cohort in COHORT_SLIDES:
        _, morpho_genes, istar_genes = load_gene_orders(root, cohort)
        gene_maps[(cohort, "Morpho-FM")] = {gene: index for index, gene in enumerate(morpho_genes)}
        gene_maps[(cohort, "iStar")] = {gene: index for index, gene in enumerate(istar_genes)}
    rows: list[dict] = []
    for task in tasks:
        key = (task.cohort, task.test)
        if key not in caches:
            caches[key] = make_target_cache(
                root,
                output_dir,
                task.cohort,
                task.test,
                programmes,
                tasks,
            )
        cache = caches[key]
        upper_index = np.triu_indices(len(cache.coexpression_genes), k=1)
        for method, result_dir in (("Morpho-FM", task.morpho_dir), ("iStar", task.istar_dir)):
            values = matrix_columns(
                result_dir / "pred_bag.npy",
                gene_maps[(task.cohort, method)],
                cache.coexpression_genes,
            )
            predicted_upper = correlation_matrix(values)[upper_index]
            prefix = {
                "cohort": task.cohort,
                "train": task.train,
                "test": task.test,
                "task_id": task.task_id,
                "method": method,
            }
            rows.extend(prefix | item for item in edge_metrics(predicted_upper, cache.truth_coexpression_upper))
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "conditional_edge_metrics.csv", index=False)
    keys = ["cohort", "train", "test", "task_id", "edge_density"]
    metric_columns = ["top_edge_recovery", "truth_top_edge_sign_preservation"]
    morpho = frame[frame["method"].eq("Morpho-FM")][keys + metric_columns].rename(
        columns={column: f"{column}_morpho" for column in metric_columns}
    )
    istar = frame[frame["method"].eq("iStar")][keys + metric_columns].rename(
        columns={column: f"{column}_istar" for column in metric_columns}
    )
    paired = morpho.merge(istar, on=keys, validate="one_to_one")
    for column in metric_columns:
        paired[f"{column}_advantage"] = paired[f"{column}_morpho"] - paired[f"{column}_istar"]
    summary_columns = [column for column in paired.columns if column.endswith(("_morpho", "_istar", "_advantage"))]
    section_summary = (
        paired.groupby(["cohort", "test", "edge_density"], as_index=False)[summary_columns]
        .median()
        .sort_values(["cohort", "test", "edge_density"])
    )
    section_summary.to_csv(destination / "conditional_edge_section_summary.csv", index=False)
    return frame


def run_full_analysis(root: Path, output_dir: Path) -> dict:
    pilot = json.loads(require_file(output_dir / "pilot" / "pilot_validation.json").read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS":
        raise RuntimeError("Pilot must pass before the full Phase 6 run")
    destination = output_dir / "full"
    programme, moran, coexpression = run_task_set(
        root,
        output_dir,
        build_tasks(root),
        destination,
    )
    run_prior_analysis(
        root,
        output_dir,
        destination,
    )
    return summarize_full_analysis(root, output_dir, reuse_conditional_edges=False)


def summarize_full_analysis(root: Path, output_dir: Path, reuse_conditional_edges: bool = True) -> dict:
    destination = output_dir / "full"
    programme = pd.read_csv(require_file(destination / "programme_metrics.csv"))
    moran = pd.read_csv(require_file(destination / "moran_metrics.csv"))
    coexpression = pd.read_csv(require_file(destination / "coexpression_metrics.csv"))
    prior_programme = pd.read_csv(require_file(destination / "prior_programme_metrics.csv"))
    prior_moran = pd.read_csv(require_file(destination / "prior_moran_metrics.csv"))
    prior_coexpression = pd.read_csv(require_file(destination / "prior_coexpression_metrics.csv"))
    direction = build_direction_summary(programme, moran, coexpression)
    section = build_section_summary(direction)
    moran_by_k = build_moran_section_by_k(moran)
    bootstrap = bootstrap_section_summary(section)
    prior = build_prior_summary(prior_programme, prior_moran, prior_coexpression)
    gate = decide_gate(section, moran_by_k, prior)

    direction.to_csv(destination / "direction_summary.csv", index=False)
    section.to_csv(destination / "section_summary.csv", index=False)
    moran_by_k.to_csv(destination / "moran_section_sensitivity.csv", index=False)
    bootstrap.to_csv(destination / "section_bootstrap_summary.csv", index=False)
    prior.to_csv(destination / "prior_summary.csv", index=False)
    write_json(destination / "gate6_2_decision.json", gate)
    edge_paths = [
        destination / "conditional_edge_metrics.csv",
        destination / "conditional_edge_section_summary.csv",
    ]
    if gate["coexpression"]["pass"] and (not reuse_conditional_edges or not all(path.is_file() for path in edge_paths)):
        run_conditional_edges(root, output_dir, destination)

    summary = {
        "created_at": now_iso(),
        "status": "COMPLETE",
        "decision": gate["decision"],
        "benchmark_directions": int(direction["task_id"].nunique()),
        "held_out_sections": int(section.shape[0]),
        "programmes": int(programme["programme"].nunique()),
        "moran_panel_genes_per_section": 500,
        "coexpression_panel_genes_per_section": 500,
        "conditional_edges_run": bool(gate["coexpression"]["pass"]),
        "gpu_used": False,
        "output_dir": str(destination),
    }
    write_json(destination / "phase6_run_summary.json", summary)
    print(f"[Gate 6.2 {gate['decision']}] full Phase 6 analysis complete")
    return gate


def plot_delta_panel(
    ax,
    section: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
) -> None:
    colors = {"Prostate": "#0072B2", "Kidney": "#D55E00"}
    offsets = {"Prostate": -0.10, "Kidney": 0.10}
    for x, (column, label) in enumerate(zip(columns, labels, strict=True)):
        for cohort in ("Prostate", "Kidney"):
            values = section.loc[section["cohort"].eq(cohort), column].to_numpy(dtype=float)
            positions = np.full(len(values), x + offsets[cohort])
            ax.scatter(
                positions,
                values,
                s=18,
                color=colors[cohort],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.35,
                label=cohort if x == 0 else None,
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot(
                [x + offsets[cohort] - 0.065, x + offsets[cohort] + 0.065],
                [median, median],
                color=colors[cohort],
                linewidth=2.0,
                zorder=4,
            )
    ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_ylabel("Advantage over iStar")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)


def plot_phase6(output_dir: Path, figure_dir: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    destination = output_dir / "full"
    gate = json.loads(require_file(destination / "gate6_2_decision.json").read_text(encoding="utf-8"))
    if gate["decision"] == "STOP-STRUCTURE":
        print("[Figure skipped] Gate decision is STOP-STRUCTURE")
        return []
    section = pd.read_csv(require_file(destination / "section_summary.csv"))
    prior = pd.read_csv(require_file(destination / "prior_summary.csv"))
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.7), constrained_layout=True)
    plot_delta_panel(
        axes[0, 0],
        section,
        ["programme_advantage"],
        ["Programme\nconcordance"],
        "a  Coordinated molecular programmes",
    )
    axes[0, 0].legend(frameon=False, loc="best")
    plot_delta_panel(
        axes[0, 1],
        section,
        ["moran_concordance_advantage", "moran_mae_advantage"],
        ["Moran\nconcordance", "Moran error\nreduction"],
        "b  Spatial-autocorrelation fidelity",
    )
    plot_delta_panel(
        axes[1, 0],
        section,
        ["coexpression_rank_advantage", "coexpression_error_advantage"],
        ["Rank\nconcordance", "Matrix error\nreduction"],
        "c  Co-expression consistency",
    )

    ax = axes[1, 1]
    metrics = [
        ("programme_advantage", "Programme"),
        ("moran_concordance_advantage", "Moran\nconcordance"),
        ("moran_mae_advantage", "Moran error\nreduction"),
        ("coexpression_rank_advantage", "Co-expression\nrank"),
        ("coexpression_error_advantage", "Matrix error\nreduction"),
    ]
    setting_style = {
        "Single-slide": ("o", "#009E73", -0.08),
        "Multi-slide": ("s", "#CC79A7", 0.08),
    }
    for setting, (marker, color, offset) in setting_style.items():
        row = prior[prior["setting"].eq(setting)].iloc[0]
        values = [float(row[column]) for column, _ in metrics]
        positions = np.arange(len(metrics), dtype=float) + offset
        ax.scatter(positions, values, marker=marker, s=27, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylabel("Pretrained minus random advantage")
    ax.set_title("d  Matched CellFM initialization controls", loc="left", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    handles = [
        Line2D([0], [0], marker=marker, linestyle="", markerfacecolor=color, markeredgecolor="white", markersize=5.5, label=setting)
        for setting, (marker, color, _) in setting_style.items()
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    for axis in axes.flat:
        axis.tick_params(length=2.5, width=0.65)

    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "phase6_structure_preservation_summary.png"
    pdf = figure_dir / "phase6_structure_preservation_summary.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Figure saved] {png}")
    print(f"[Figure saved] {pdf}")
    return [png, pdf]


def self_test() -> None:
    x, y = np.meshgrid(np.arange(6, dtype=float), np.arange(6, dtype=float))
    coords = np.column_stack([x.ravel(), y.ravel()])
    graph = knn_indices(coords, k=4)
    values = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    moran = moran_values(values, graph)
    standardized, valid = zscore_columns(values)
    corr = correlation_matrix(values[:, :2])
    assert graph.shape == (36, 4)
    assert np.isfinite(moran[:2]).all() and np.isnan(moran[2])
    assert valid.tolist() == [True, True, False]
    assert np.allclose(standardized[:, 2], 0.0)
    assert np.allclose(np.diag(corr), 1.0)
    assert abs(scalar_pearson(values[:, 0], values[:, 0]) - 1.0) < 1e-12

    paired_rows: list[dict] = []
    for method, method_values in (("Morpho-FM", [0.0, 10.0, 11.0]), ("iStar", [0.0, 9.0, 100.0])):
        for index, value in enumerate(method_values):
            paired_rows.append(
                {
                    "cohort": "Synthetic",
                    "train": f"train_{index}",
                    "test": "test_0",
                    "task_id": f"task_{index}",
                    "method": method,
                    "programme_concordance": value,
                    "moran_concordance": value,
                    "moran_mae": value,
                    "coexpression_rank_concordance": value,
                    "coexpression_normalized_error": value,
                }
            )
    paired_section = build_section_summary(pd.DataFrame(paired_rows)).iloc[0]
    assert paired_section["programme_advantage"] == 0.0
    assert paired_section["programme_morpho"] - paired_section["programme_istar"] == 1.0

    prior_programme_rows: list[dict] = []
    for initialization, method_values in (("Pretrained", [0.0, 10.0, 11.0]), ("Random", [0.0, 9.0, 100.0])):
        for index, value in enumerate(method_values):
            prior_programme_rows.append(
                {
                    "setting": "Synthetic",
                    "initialization": initialization,
                    "programme": f"programme_{index}",
                    "concordance": value,
                }
            )
    prior_moran = pd.DataFrame(
        [
            {"setting": "Synthetic", "initialization": initialization, "k": 6, "concordance": value, "mae": value}
            for initialization, value in (("Pretrained", 1.0), ("Random", 0.0))
        ]
    )
    prior_coexpression = pd.DataFrame(
        [
            {
                "setting": "Synthetic",
                "initialization": initialization,
                "rank_concordance": value,
                "normalized_matrix_error": 1.0 - value,
            }
            for initialization, value in (("Pretrained", 1.0), ("Random", 0.0))
        ]
    )
    paired_prior = build_prior_summary(
        pd.DataFrame(prior_programme_rows),
        prior_moran,
        prior_coexpression,
    ).iloc[0]
    assert paired_prior["programme_advantage"] == 0.0
    assert paired_prior["programme_pretrained"] - paired_prior["programme_random"] == 1.0
    print("[Self-test PASS] numerical primitives and paired aggregation")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    gmt_path = args.gmt.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    if args.command == "self-test":
        self_test()
        return
    if args.command in {"audit", "all"}:
        audit_inputs(root, output_dir, full_truth_check=not args.skip_full_truth_check)
    if args.command in {"freeze", "all"}:
        freeze_analysis_spec(root, output_dir, gmt_path)
    if args.command in {"pilot", "all"}:
        run_pilot(root, output_dir)
    if args.command in {"run", "all"}:
        run_full_analysis(root, output_dir)
    if args.command == "summarize":
        summarize_full_analysis(root, output_dir)
    if args.command in {"plot", "all"}:
        plot_phase6(output_dir, figure_dir)


if __name__ == "__main__":
    main()
