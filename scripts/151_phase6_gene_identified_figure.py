from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "150_phase6_structure_preservation.py"
DEFAULT_OUTPUT = ROOT / "results" / "phase6_structure_preservation" / "gene_identified_presentation"
DEFAULT_PHASE6_OUTPUT = ROOT / "results" / "phase6_structure_preservation"
DEFAULT_SUPP10_OUTPUT = (
    Path.home()
    / "Desktop"
    / "\u4eba\u751f\u7684\u7b2c4\u7bc7\u8bba\u6587"
    / "manuscript_npj"
    / "Figures"
    / "fig_supp_10.png"
)
DEFAULT_FIGURE_DIR = Path(
    r"C:\Users\Administrator\Desktop\人生的第4篇论文\Figure\Phase_6_structure_preservation"
)

PROSTATE_EXAMPLE = {
    "cohort": "Prostate",
    "train": "INT25",
    "test": "INT28",
    "programme": "HALLMARK_ANDROGEN_RESPONSE",
    "selection_rule": "Existing representative direction in the manuscript benchmark.",
}
KIDNEY_PROGRAMME = "HALLMARK_HYPOXIA"
COEXPRESSION_GENES_PER_PROGRAMME = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the gene-identified Phase 6 analysis and Supplementary Figure 16 draft."
    )
    parser.add_argument("command", choices=["self-test", "run", "plot", "supp10", "all"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase6-output", type=Path, default=DEFAULT_PHASE6_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--supp10-output", type=Path, default=DEFAULT_SUPP10_OUTPUT)
    return parser.parse_args()


def load_phase6_module():
    spec = importlib.util.spec_from_file_location("phase6_structure", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def standardize_in_place(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Expression matrix contains non-finite values")
    matrix -= matrix.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.einsum("ij,ij->j", matrix, matrix) / float(matrix.shape[0]))
    valid = scale > 1e-12
    matrix[:, valid] /= scale[valid]
    matrix[:, ~valid] = 0.0
    return matrix, valid


def matrix_columns_sequential(
    base,
    path: Path,
    gene_to_native: dict[str, int],
    genes: list[str],
    row_chunk: int = 128,
) -> np.ndarray:
    matrix = np.load(base.require_file(path), mmap_mode="r")
    try:
        indices = np.asarray([gene_to_native[gene] for gene in genes], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Missing requested gene {exc.args[0]} in {path}") from exc
    selected = np.empty((matrix.shape[0], len(indices)), dtype=np.float64)
    for start in range(0, matrix.shape[0], row_chunk):
        stop = min(start + row_chunk, matrix.shape[0])
        block = np.array(matrix[start:stop], dtype=np.float32, copy=True, order="C")
        selected[start:stop] = block[:, indices]
    if not np.isfinite(selected).all():
        raise ValueError(f"Non-finite values in {path}")
    return selected


def correlation_matrix_from_standardized(standardized: np.ndarray, members: np.ndarray) -> np.ndarray:
    values = standardized[:, members]
    return (values.T @ values) / float(values.shape[0])


def matrix_metrics(base, predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, bool]:
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError(f"Correlation-matrix mismatch: {predicted.shape} versus {truth.shape}")
    upper = np.triu_indices(predicted.shape[0], k=1)
    predicted_upper = predicted[upper]
    truth_upper = truth[upper]
    rank = base.scalar_spearman(predicted_upper, truth_upper)
    rank_was_constant = not np.isfinite(rank)
    if rank_was_constant:
        rank = 0.0
    denominator = float(np.linalg.norm(truth))
    if denominator <= 1e-12:
        raise ValueError("Truth correlation matrix has zero norm")
    error = float(np.linalg.norm(predicted - truth) / denominator)
    return float(rank), error, rank_was_constant


def verify_frozen_inputs(base, phase6_output: Path) -> None:
    audit = json.loads(
        base.require_file(phase6_output / "gate6_0" / "phase6_input_alignment_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if audit.get("status") != "PASS":
        raise RuntimeError("Gate 6.0 is not PASS")
    spec_path = base.require_file(phase6_output / "gate6_1" / "analysis_spec.json")
    expected_hash = base.require_file(phase6_output / "gate6_1" / "analysis_spec.sha256").read_text(
        encoding="ascii"
    ).strip()
    if base.sha256_file(spec_path) != expected_hash:
        raise RuntimeError("Frozen Gate 6.1 specification hash changed")


def make_target_context(base, root: Path, phase6_output: Path, cohort: str, test: str, programmes, tasks):
    target_tasks = [task for task in tasks if task.cohort == cohort and task.test == test]
    if not target_tasks:
        raise ValueError(f"No tasks for {cohort} {test}")
    canonical, morpho_genes, istar_genes = base.load_gene_orders(root, cohort)
    morpho_map = {gene: index for index, gene in enumerate(morpho_genes)}
    istar_map = {gene: index for index, gene in enumerate(istar_genes)}
    reference = target_tasks[0]
    programme_union_set = {gene for item in programmes for gene in item["genes"]}
    programme_union = [gene for gene in canonical if gene in programme_union_set]
    union_index = {gene: index for index, gene in enumerate(programme_union)}
    truth_raw = matrix_columns_sequential(
        base,
        reference.morpho_dir / "true_bag.npy", morpho_map, programme_union
    )
    truth_variance = truth_raw.var(axis=0)
    truth_standardized, truth_valid = standardize_in_place(truth_raw)
    score_member_genes: dict[str, list[str]] = {}
    coexpression_member_indices: dict[str, np.ndarray] = {}
    coexpression_member_genes: dict[str, list[str]] = {}
    truth_correlations: dict[str, np.ndarray] = {}
    selected_gene_set: set[str] = set()
    for programme in programmes:
        name = str(programme["name"])
        genes = [
            gene
            for gene in programme["genes"]
            if truth_valid[union_index[gene]]
        ]
        score_indices = np.asarray([union_index[gene] for gene in genes], dtype=np.int64)
        if len(score_indices) < COEXPRESSION_GENES_PER_PROGRAMME:
            raise ValueError(
                f"Only {len(score_indices)} truth-variable genes for {cohort} {test} {name}"
            )
        ranked = sorted(
            zip(genes, score_indices, strict=True),
            key=lambda item: (-float(truth_variance[item[1]]), item[0]),
        )
        selected = ranked[:COEXPRESSION_GENES_PER_PROGRAMME]
        coexpression_genes = [item[0] for item in selected]
        coexpression_indices = np.asarray([item[1] for item in selected], dtype=np.int64)
        score_member_genes[name] = genes
        coexpression_member_indices[name] = coexpression_indices
        coexpression_member_genes[name] = coexpression_genes
        selected_gene_set.update(coexpression_genes)
        truth_correlations[name] = correlation_matrix_from_standardized(
            truth_standardized, coexpression_indices
        )
    analysis_genes = [gene for gene in canonical if gene in selected_gene_set]
    analysis_index = {gene: index for index, gene in enumerate(analysis_genes)}
    analysis_member_indices = {
        name: np.asarray([analysis_index[gene] for gene in genes], dtype=np.int64)
        for name, genes in coexpression_member_genes.items()
    }
    return {
        "target_tasks": target_tasks,
        "morpho_map": morpho_map,
        "istar_map": istar_map,
        "analysis_genes": analysis_genes,
        "score_member_genes": score_member_genes,
        "coexpression_member_indices": analysis_member_indices,
        "coexpression_member_genes": coexpression_member_genes,
        "truth_correlations": truth_correlations,
    }


def evaluate_method(base, prediction_path: Path, gene_map: dict[str, int], context) -> dict[str, dict]:
    values = matrix_columns_sequential(
        base, prediction_path, gene_map, context["analysis_genes"]
    )
    standardized, _ = standardize_in_place(values)
    rows: dict[str, dict] = {}
    for name, members in context["coexpression_member_indices"].items():
        predicted = correlation_matrix_from_standardized(standardized, members)
        rank, error, rank_constant = matrix_metrics(
            base, predicted, context["truth_correlations"][name]
        )
        rows[name] = {
            "rank_concordance": rank,
            "normalized_matrix_error": error,
            "rank_constant": rank_constant,
        }
    return rows


def build_programme_score_summary(phase6_output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(
        phase6_output / "full" / "programme_metrics.csv"
    )
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
    tissue_rows: list[dict] = []
    for (cohort, programme), group in section.groupby(["cohort", "programme"], sort=False):
        tissue_rows.append(
            {
                "cohort": cohort,
                "programme": programme,
                "score_morpho": float(group["score_morpho"].median()),
                "score_istar": float(group["score_istar"].median()),
                "score_advantage": float(group["score_advantage"].median()),
                "score_positive_sections": int((group["score_advantage"] > 0.0).sum()),
                "n_sections": len(group),
            }
        )
    return section, pd.DataFrame(tissue_rows)


def aggregate_programme_coexpression(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = [
        "n_genes",
        "morpho_rank",
        "istar_rank",
        "rank_advantage",
        "morpho_error",
        "istar_error",
        "error_advantage",
    ]
    section = (
        pairs.groupby(["cohort", "test", "programme"], as_index=False)[numeric]
        .median()
        .sort_values(["cohort", "test", "programme"])
        .reset_index(drop=True)
    )
    tissue_rows: list[dict] = []
    for (cohort, programme), group in section.groupby(["cohort", "programme"], sort=False):
        tissue_rows.append(
            {
                "cohort": cohort,
                "programme": programme,
                "n_genes_min": int(group["n_genes"].min()),
                "n_genes_max": int(group["n_genes"].max()),
                "morpho_rank": float(group["morpho_rank"].median()),
                "istar_rank": float(group["istar_rank"].median()),
                "rank_advantage": float(group["rank_advantage"].median()),
                "rank_positive_sections": int((group["rank_advantage"] > 0.0).sum()),
                "morpho_error": float(group["morpho_error"].median()),
                "istar_error": float(group["istar_error"].median()),
                "error_advantage": float(group["error_advantage"].median()),
                "error_positive_sections": int((group["error_advantage"] > 0.0).sum()),
                "joint_positive_sections": int(
                    ((group["rank_advantage"] > 0.0) & (group["error_advantage"] > 0.0)).sum()
                ),
                "n_sections": len(group),
            }
        )
    tissue = pd.DataFrame(tissue_rows)

    section_summary = (
        section.groupby(["cohort", "test"], as_index=False)[numeric[1:]]
        .median()
        .sort_values(["cohort", "test"])
        .reset_index(drop=True)
    )
    section_summary["n_programmes"] = pairs["programme"].nunique()
    return section, tissue, section_summary


def leave_one_out_positive(values: np.ndarray) -> bool:
    data = np.asarray(values, dtype=np.float64)
    return len(data) >= 2 and all(
        float(np.median(np.delete(data, index))) > 0.0 for index in range(len(data))
    )


def build_gate(section_summary: pd.DataFrame) -> dict:
    details: dict[str, dict] = {}
    passed = True
    for cohort, required in (("Prostate", 3), ("Kidney", 6)):
        subset = section_summary[section_summary["cohort"].eq(cohort)]
        rank = subset["rank_advantage"].to_numpy(dtype=float)
        error = subset["error_advantage"].to_numpy(dtype=float)
        joint = (rank > 0.0) & (error > 0.0)
        cohort_pass = bool(
            joint.sum() >= required
            and np.median(rank) > 0.0
            and np.median(error) > 0.0
            and leave_one_out_positive(rank)
            and leave_one_out_positive(error)
        )
        details[cohort] = {
            "n_sections": len(subset),
            "jointly_positive_sections": int(joint.sum()),
            "required_jointly_positive": required,
            "median_rank_advantage": float(np.median(rank)),
            "median_error_advantage": float(np.median(error)),
            "rank_leave_one_section_out_positive": leave_one_out_positive(rank),
            "error_leave_one_section_out_positive": leave_one_out_positive(error),
            "pass": cohort_pass,
        }
        passed &= cohort_pass
    return {
        "status": "PASS" if passed else "FAIL",
        "scope": "Descriptive programme-specific decomposition; does not replace Gate 6.2.",
        "details": details,
    }


def choose_examples(phase6_output: Path) -> list[dict]:
    direction = pd.read_csv(phase6_output / "full" / "direction_summary.csv")
    section = pd.read_csv(phase6_output / "full" / "section_summary.csv")
    kidney_reference = float(
        section.loc[section["cohort"].eq("Kidney"), "programme_advantage"].median()
    )
    kidney = direction[direction["cohort"].eq("Kidney")].pivot(
        index=["cohort", "train", "test", "task_id"],
        columns="method",
        values="programme_concordance",
    ).reset_index()
    kidney["overall_effect"] = kidney["Morpho-FM"] - kidney["iStar"]
    kidney["distance_to_section_median"] = (kidney["overall_effect"] - kidney_reference).abs()
    closest = kidney.sort_values(
        ["distance_to_section_median", "train", "test"], kind="stable"
    ).iloc[0]
    if int((kidney["distance_to_section_median"] == closest["distance_to_section_median"]).sum()) != 1:
        raise RuntimeError("Kidney closest-to-median example is not unique")
    kidney_example = {
        "cohort": "Kidney",
        "train": str(closest["train"]),
        "test": str(closest["test"]),
        "programme": KIDNEY_PROGRAMME,
        "selection_rule": (
            "Direction whose overall programme effect is uniquely closest to the median of the "
            "eight held-out kidney-section effects; programme fixed for renal-cancer relevance."
        ),
        "kidney_section_median_reference": kidney_reference,
        "direction_overall_effect": float(closest["overall_effect"]),
        "distance_to_reference": float(closest["distance_to_section_median"]),
    }
    return [PROSTATE_EXAMPLE.copy(), kidney_example]


def load_example(base, root: Path, phase6_output: Path, programmes, tasks, example: dict) -> dict:
    cohort = example["cohort"]
    train = example["train"]
    test = example["test"]
    programme = example["programme"]
    context = make_target_context(base, root, phase6_output, cohort, test, programmes, tasks)
    task = next(
        item for item in context["target_tasks"] if item.train == train and item.test == test
    )
    score_genes = context["score_member_genes"][programme]
    truth_raw = matrix_columns_sequential(
        base, task.morpho_dir / "true_bag.npy", context["morpho_map"], score_genes
    )
    morpho_raw = matrix_columns_sequential(
        base, task.morpho_dir / "pred_bag.npy", context["morpho_map"], score_genes
    )
    istar_raw = matrix_columns_sequential(
        base, task.istar_dir / "pred_bag.npy", context["istar_map"], score_genes
    )
    score_members = np.arange(len(score_genes), dtype=np.int64)
    matrix_genes = context["coexpression_member_genes"][programme]
    score_index = {gene: index for index, gene in enumerate(score_genes)}
    matrix_members = np.asarray([score_index[gene] for gene in matrix_genes], dtype=np.int64)
    selected_genes = matrix_genes[:8]
    selected_indices = matrix_members[:8]

    truth_standardized, _ = standardize_in_place(truth_raw)
    morpho_standardized, _ = standardize_in_place(morpho_raw)
    istar_standardized, _ = standardize_in_place(istar_raw)
    scores = {
        "Truth": truth_standardized[:, score_members].mean(axis=1),
        "Morpho-FM": morpho_standardized[:, score_members].mean(axis=1),
        "iStar": istar_standardized[:, score_members].mean(axis=1),
    }
    gene_concordance = {
        "Morpho-FM": base.vector_pearson(
            morpho_standardized[:, selected_indices], truth_standardized[:, selected_indices]
        ),
        "iStar": base.vector_pearson(
            istar_standardized[:, selected_indices], truth_standardized[:, selected_indices]
        ),
    }
    score_metrics = pd.read_csv(phase6_output / "full" / "programme_metrics.csv")
    selected = score_metrics[
        score_metrics["cohort"].eq(cohort)
        & score_metrics["train"].eq(train)
        & score_metrics["test"].eq(test)
        & score_metrics["programme"].eq(programme)
    ].set_index("method")
    calculated_morpho = base.scalar_pearson(scores["Morpho-FM"], scores["Truth"])
    calculated_istar = base.scalar_pearson(scores["iStar"], scores["Truth"])
    if not np.isclose(calculated_morpho, float(selected.loc["Morpho-FM", "concordance"]), atol=1e-10):
        raise AssertionError("Representative Morpho-FM programme score does not match Phase 6")
    if not np.isclose(calculated_istar, float(selected.loc["iStar", "concordance"]), atol=1e-10):
        raise AssertionError("Representative iStar programme score does not match Phase 6")
    _, coords = base.h5ad_spots(root, test)
    return {
        **example,
        "score_n_genes": len(score_members),
        "coexpression_n_genes": len(matrix_members),
        "selected_genes": selected_genes,
        "coords": coords,
        "scores": scores,
        "score_concordance": {"Morpho-FM": calculated_morpho, "iStar": calculated_istar},
        "gene_concordance": gene_concordance,
    }


def run_analysis(root: Path, phase6_output: Path, output_dir: Path) -> dict:
    base = load_phase6_module()
    verify_frozen_inputs(base, phase6_output)
    programmes = base.load_programmes(phase6_output)
    tasks = base.build_tasks(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows: list[dict] = []
    member_rows: list[dict] = []
    completed = 0
    for cohort, slides in base.COHORT_SLIDES.items():
        for test in slides:
            context = make_target_context(base, root, phase6_output, cohort, test, programmes, tasks)
            for programme in programmes:
                name = str(programme["name"])
                member_rows.append(
                    {
                        "cohort": cohort,
                        "test": test,
                        "programme": name,
                        "n_genes": len(context["coexpression_member_genes"][name]),
                        "genes": ",".join(context["coexpression_member_genes"][name]),
                    }
                )
            for task in context["target_tasks"]:
                morpho = evaluate_method(
                    base,
                    task.morpho_dir / "pred_bag.npy",
                    context["morpho_map"],
                    context,
                )
                istar = evaluate_method(
                    base,
                    task.istar_dir / "pred_bag.npy",
                    context["istar_map"],
                    context,
                )
                for programme in programmes:
                    name = str(programme["name"])
                    paired_rows.append(
                        {
                            "cohort": cohort,
                            "train": task.train,
                            "test": task.test,
                            "task_id": task.task_id,
                            "programme": name,
                            "n_genes": len(context["coexpression_member_indices"][name]),
                            "morpho_rank": morpho[name]["rank_concordance"],
                            "istar_rank": istar[name]["rank_concordance"],
                            "rank_advantage": morpho[name]["rank_concordance"]
                            - istar[name]["rank_concordance"],
                            "morpho_error": morpho[name]["normalized_matrix_error"],
                            "istar_error": istar[name]["normalized_matrix_error"],
                            "error_advantage": istar[name]["normalized_matrix_error"]
                            - morpho[name]["normalized_matrix_error"],
                            "morpho_rank_constant": morpho[name]["rank_constant"],
                            "istar_rank_constant": istar[name]["rank_constant"],
                        }
                    )
                completed += 1
                print(f"[Programme co-expression] {completed:02d}/68 {task.task_id}", flush=True)

    pairs = pd.DataFrame(paired_rows)
    expected_keys = ["cohort", "train", "test", "task_id", "programme"]
    numeric = [
        "n_genes",
        "morpho_rank",
        "istar_rank",
        "rank_advantage",
        "morpho_error",
        "istar_error",
        "error_advantage",
    ]
    if len(pairs) != 68 * 50 or pairs.duplicated(expected_keys).any():
        raise AssertionError(f"Invalid paired table shape or keys: {pairs.shape}")
    if not np.isfinite(pairs[numeric].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite programme co-expression output")
    if not (pairs["n_genes"] >= 10).all():
        raise AssertionError("Programme with fewer than 10 genes entered the analysis")

    section_programme, tissue_coexpression, section_summary = aggregate_programme_coexpression(pairs)
    _, score_tissue = build_programme_score_summary(phase6_output)
    programme_order = {item["name"]: index for index, item in enumerate(programmes)}
    manifest = pd.DataFrame(
        [
            {
                "programme": item["name"],
                "programme_order": programme_order[item["name"]],
                "n_source_genes": item["n_source_genes"],
                "n_canonical_genes": item["n_panel_genes"],
            }
            for item in programmes
        ]
    )
    tissue = tissue_coexpression.merge(score_tissue, on=["cohort", "programme", "n_sections"])
    tissue = tissue.merge(manifest, on="programme", validate="many_to_one")
    tissue = tissue.sort_values(["programme_order", "cohort"]).reset_index(drop=True)
    gate = build_gate(section_summary)
    examples = choose_examples(phase6_output)
    example_rows: list[dict] = []
    for example in examples:
        loaded = load_example(base, root, phase6_output, programmes, tasks, example)
        example_rows.append(
            {
                **example,
                "score_n_genes": loaded["score_n_genes"],
                "coexpression_n_genes": loaded["coexpression_n_genes"],
                "selected_genes": ",".join(loaded["selected_genes"]),
                "morpho_score_concordance": loaded["score_concordance"]["Morpho-FM"],
                "istar_score_concordance": loaded["score_concordance"]["iStar"],
                "morpho_gene_concordance": ",".join(
                    f"{value:.12g}" for value in loaded["gene_concordance"]["Morpho-FM"]
                ),
                "istar_gene_concordance": ",".join(
                    f"{value:.12g}" for value in loaded["gene_concordance"]["iStar"]
                ),
            }
        )

    pairs.to_csv(output_dir / "programme_coexpression_direction_pairs.csv", index=False)
    section_programme.to_csv(output_dir / "programme_coexpression_section_summary.csv", index=False)
    tissue.to_csv(output_dir / "programme_molecular_tissue_summary.csv", index=False)
    section_summary.to_csv(output_dir / "programme_coexpression_gate_section_summary.csv", index=False)
    pd.DataFrame(member_rows).to_csv(output_dir / "programme_coexpression_members.csv", index=False)
    pd.DataFrame(example_rows).to_csv(output_dir / "suppfig16_example_manifest.csv", index=False)
    gate_payload = {
        **gate,
        "programmes": len(programmes),
        "directions": len(tasks),
        "paired_rows": len(pairs),
        "truth_defined_membership": True,
        "coexpression_genes_per_programme": COEXPRESSION_GENES_PER_PROGRAMME,
        "gpu_used": False,
        "training_or_inference_used": False,
        "output_dir": str(output_dir),
    }
    base.write_json(output_dir / "programme_specific_coexpression_gate.json", gate_payload)
    print(f"[Programme-specific co-expression gate] {gate['status']}")
    return gate_payload


def rounded_symmetric_limit(values: np.ndarray, minimum: float) -> float:
    maximum = max(float(np.nanmax(np.abs(values))), minimum)
    step = 0.05 if maximum <= 0.5 else 0.1
    return math.ceil(maximum / step) * step


def plot_section_panel(ax, section: pd.DataFrame, column: str, title: str, colors, markers) -> None:
    for cohort, offset in (("Prostate", -0.12), ("Kidney", 0.12)):
        values = section.loc[section["cohort"].eq(cohort), column].to_numpy(dtype=float)
        x = np.full(len(values), offset)
        ax.scatter(
            x,
            values,
            s=22,
            marker=markers[cohort],
            color=colors[cohort],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            label=cohort,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot([offset - 0.07, offset + 0.07], [median, median], color=colors[cohort], lw=1.8)
    ax.axhline(0.0, color="#666666", lw=0.6, ls="--")
    ax.set_xlim(-0.34, 0.34)
    ax.set_xticks([])
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    ax.grid(axis="y", color="#DDDDDD", lw=0.45)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5, length=2)


def short_programme_name(name: str) -> str:
    return name.removeprefix("HALLMARK_").replace("_", " ")


def plot_programme_block(ax, block: pd.DataFrame, cmap, norm) -> None:
    rows = [
        {"programme": name, "label": short_programme_name(name)}
        for name in block["programme"].drop_duplicates().tolist()
    ]
    for y, row in enumerate(rows):
        for x, cohort in enumerate(("Prostate", "Kidney")):
            selected = block[
                block["programme"].eq(row["programme"]) & block["cohort"].eq(cohort)
            ].iloc[0]
            ax.scatter(
                x,
                y,
                s=24,
                c=[float(selected["score_advantage"])],
                cmap=cmap,
                norm=norm,
                marker="o" if cohort == "Prostate" else "s",
                edgecolor="#333333",
                linewidth=0.28,
            )
        retained = int(
            block.loc[block["programme"].eq(row["programme"]), "n_canonical_genes"].iloc[0]
        )
        ax.text(2, y, str(retained), ha="center", va="center", fontsize=6.2, color="#333333")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=6.5)
    ax.set_xticks([0, 1, 2], ["Prostate", "Kidney", "Genes"], fontsize=6.7)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="y", color="#EEEEEE", lw=0.35)
    ax.spines[:].set_visible(False)


# Retained for audit comparison; the active entry point is the redesigned plot_figure below.
def _plot_figure_legacy(root: Path, phase6_output: Path, output_dir: Path, figure_dir: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D

    base = load_phase6_module()
    programmes = base.load_programmes(phase6_output)
    tasks = base.build_tasks(root)
    tissue = pd.read_csv(output_dir / "programme_molecular_tissue_summary.csv")
    phase6_section = pd.read_csv(phase6_output / "full" / "section_summary.csv")
    section = phase6_section[["cohort", "test", "programme_advantage"]]
    prior = pd.read_csv(phase6_output / "full" / "prior_summary.csv")
    coexpression_gate = json.loads(
        (output_dir / "programme_specific_coexpression_gate.json").read_text(encoding="utf-8")
    )
    if coexpression_gate["status"] != "FAIL":
        raise RuntimeError("Programme-only figure is reserved for a failed co-expression follow-up gate")
    examples = [
        load_example(base, root, phase6_output, programmes, tasks, example)
        for example in choose_examples(phase6_output)
    ]
    gene_rows: list[dict] = []
    for example in examples:
        for index, gene in enumerate(example["selected_genes"]):
            gene_rows.append(
                {
                    "cohort": example["cohort"],
                    "train": example["train"],
                    "test": example["test"],
                    "programme": example["programme"],
                    "gene": gene,
                    "morpho_pcc": example["gene_concordance"]["Morpho-FM"][index],
                    "istar_pcc": example["gene_concordance"]["iStar"][index],
                }
            )
    pd.DataFrame(gene_rows).to_csv(output_dir / "suppfig16_example_gene_concordance.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    cohort_colors = {"Prostate": "#0072B2", "Kidney": "#D55E00"}
    cohort_markers = {"Prostate": "o", "Kidney": "s"}
    diverging = LinearSegmentedColormap.from_list(
        "blue_white_orange", ["#2166AC", "#F7F7F7", "#B35806"]
    )
    fig = plt.figure(figsize=(7.1, 10.8), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.20, 6.15, 2.65],
        left=0.16,
        right=0.98,
        top=0.975,
        bottom=0.055,
        hspace=0.26,
    )

    a_grid = outer[0, 0].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.38)
    a_left = fig.add_subplot(a_grid[0, 0])
    plot_section_panel(
        a_left,
        section,
        "programme_advantage",
        "Morpho-FM vs iStar across held-out sections",
        cohort_colors,
        cohort_markers,
    )
    a_left.set_ylabel("Programme-score advantage", fontsize=6.8)
    handles = [
        Line2D(
            [],
            [],
            marker=cohort_markers[name],
            ls="",
            color=cohort_colors[name],
            label=f"{name} (n={4 if name == 'Prostate' else 8})",
        )
        for name in ("Prostate", "Kidney")
    ]
    a_left.legend(handles=handles, frameon=False, fontsize=6.1, loc="lower right")
    a_left.text(-0.35, 1.22, "a", transform=a_left.transAxes, fontsize=10, fontweight="bold", va="top")
    a_left.text(
        -0.21,
        1.22,
        "Coordinated molecular-programme recovery",
        transform=a_left.transAxes,
        fontsize=8.3,
        fontweight="bold",
        va="top",
    )

    a_right = fig.add_subplot(a_grid[0, 1])
    prior_order = ["Single-slide", "Multi-slide"]
    prior_colors = ["#009E73", "#CC79A7"]
    for x, (setting, color) in enumerate(zip(prior_order, prior_colors, strict=True)):
        value = float(prior.loc[prior["setting"].eq(setting), "programme_advantage"].iloc[0])
        a_right.scatter(x, value, s=34, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        a_right.text(x, value + 0.012, f"{value:+.3f}", ha="center", va="bottom", fontsize=6.2)
    a_right.axhline(0.0, color="#666666", lw=0.6, ls="--")
    a_right.set_xticks(range(2), ["Single-slide", "Multi-slide"], fontsize=6.4)
    a_right.set_ylabel("Pretrained − random", fontsize=6.8)
    a_right.set_title("Matched CellFM initialization", fontsize=7.5, fontweight="bold", pad=3)
    a_right.grid(axis="y", color="#DDDDDD", lw=0.45)
    a_right.spines[["top", "right"]].set_visible(False)
    a_right.tick_params(axis="y", labelsize=6.5, length=2)

    b_grid = outer[1, 0].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.0)
    programme_names = [item["name"] for item in programmes]
    order = pd.DataFrame({"programme": programme_names, "programme_order": range(50)})
    tissue_ordered = order.merge(tissue, on="programme", how="left", validate="one_to_many")
    score_limit = rounded_symmetric_limit(tissue["score_advantage"].to_numpy(float), 0.1)
    score_norm = TwoSlopeNorm(vmin=-score_limit, vcenter=0.0, vmax=score_limit)
    b_label_axis = fig.add_subplot(b_grid[0, 0])
    b_label_axis.set_axis_off()
    b_axis = fig.add_subplot(b_grid[0, 1])
    plot_programme_block(b_axis, tissue_ordered, diverging, score_norm)
    b_label_axis.text(-0.16, 1.04, "b", transform=b_label_axis.transAxes, fontsize=10, fontweight="bold", va="top")
    b_label_axis.text(
        -0.06,
        1.04,
        "All 50 predefined MSigDB Hallmark programmes",
        transform=b_label_axis.transAxes,
        fontsize=8.3,
        fontweight="bold",
        va="top",
    )
    b_label_axis.text(
        -0.06,
        1.005,
        "All programmes were positive in 4/4 prostate and 8/8 kidney sections",
        transform=b_label_axis.transAxes,
        fontsize=6.4,
        color="#555555",
        va="top",
    )
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=score_norm, cmap=diverging),
        ax=b_axis,
        orientation="horizontal",
        fraction=0.020,
        pad=0.035,
        aspect=35,
    )
    colorbar.ax.tick_params(labelsize=6.0, length=2, pad=1)
    colorbar.set_label("Median paired programme-score advantage over iStar", fontsize=6.4, labelpad=2)

    bottom = outer[2, 0].subgridspec(1, 2, width_ratios=[1.22, 0.78], wspace=0.46)
    c_grid = bottom[0, 0].subgridspec(
        2, 4, width_ratios=[1, 1, 1, 0.055], hspace=0.28, wspace=0.08
    )
    c_axes = []
    for row, example in enumerate(examples):
        all_scores = np.concatenate([example["scores"][name] for name in ("Truth", "Morpho-FM", "iStar")])
        limit = max(float(np.quantile(np.abs(all_scores), 0.98)), 0.25)
        norm = Normalize(vmin=-limit, vmax=limit)
        row_axes = []
        for column, method in enumerate(("Truth", "Morpho-FM", "iStar")):
            ax = fig.add_subplot(c_grid[row, column])
            coords = example["coords"]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=example["scores"][method],
                cmap=diverging,
                norm=norm,
                s=2.2,
                linewidths=0,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title(method, fontsize=6.8, fontweight="bold", pad=2)
            if method != "Truth":
                ax.text(
                    0.98,
                    0.02,
                    f"r={example['score_concordance'][method]:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.1},
                )
            row_axes.append(ax)
        cax = fig.add_subplot(c_grid[row, 3])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=diverging), cax=cax)
        colorbar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
        title = short_programme_name(example["programme"]).title()
        row_axes[0].text(
            -0.10,
            0.50,
            f"{example['cohort']}\n{title}\n{example['train']}→{example['test']} | {example['score_n_genes']} genes",
            transform=row_axes[0].transAxes,
            ha="right",
            va="center",
            fontsize=5.8,
        )
        c_axes.extend(row_axes)
    c_axes[0].text(-0.78, 1.30, "c", transform=c_axes[0].transAxes, fontsize=10, fontweight="bold", va="top")
    c_axes[0].text(
        -0.60,
        1.30,
        "Named programme-score maps",
        transform=c_axes[0].transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )

    d_grid = bottom[0, 1].subgridspec(2, 1, hspace=0.58)
    d_axes = []
    for row, example in enumerate(examples):
        ax = fig.add_subplot(d_grid[row, 0])
        genes = example["selected_genes"]
        morpho = np.asarray(example["gene_concordance"]["Morpho-FM"], dtype=float)
        istar = np.asarray(example["gene_concordance"]["iStar"], dtype=float)
        y = np.arange(len(genes))
        for index in range(len(genes)):
            ax.plot([istar[index], morpho[index]], [y[index], y[index]], color="#B8B8B8", lw=1.0, zorder=1)
        ax.scatter(istar, y, marker="s", s=17, color="#777777", edgecolor="white", linewidth=0.3, label="iStar", zorder=2)
        ax.scatter(morpho, y, marker="o", s=20, color="#0072B2", edgecolor="white", linewidth=0.3, label="Morpho-FM", zorder=3)
        ax.axvline(0.0, color="#777777", lw=0.55, ls="--")
        ax.set_yticks(y, genes, fontsize=6.2)
        ax.set_ylim(len(genes) - 0.45, -0.55)
        ax.set_xlim(-0.25, 1.03)
        ax.grid(axis="x", color="#E2E2E2", lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.2, length=2)
        title = short_programme_name(example["programme"]).title()
        ax.set_title(
            f"{example['cohort']} | {title} | {example['train']}→{example['test']}",
            fontsize=6.7,
            fontweight="bold",
            pad=3,
        )
        if row == 0:
            ax.legend(frameon=False, fontsize=5.9, loc="upper left")
        if row == 1:
            ax.set_xlabel("Per-gene PCC across locations", fontsize=6.5)
        d_axes.append(ax)
    d_axes[0].text(-0.52, 1.30, "d", transform=d_axes[0].transAxes, fontsize=10, fontweight="bold", va="top")
    d_axes[0].text(
        -0.35,
        1.30,
        "Representative programme-member genes",
        transform=d_axes[0].transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "phase6_named_programme_summary.png"
    pdf = figure_dir / "phase6_named_programme_summary.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"[Figure saved] {png}")
    print(f"[Figure saved] {pdf}")
    print("[Co-expression omitted] Programme-specific follow-up gate was FAIL")
    return [png, pdf]


def _plot_figure_composite(root: Path, phase6_output: Path, output_dir: Path, figure_dir: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    base = load_phase6_module()
    programmes = base.load_programmes(phase6_output)
    tasks = base.build_tasks(root)
    tissue = pd.read_csv(output_dir / "programme_molecular_tissue_summary.csv")
    programme_section, programme_tissue = build_programme_score_summary(phase6_output)
    phase6_section = pd.read_csv(phase6_output / "full" / "section_summary.csv")
    prior = pd.read_csv(phase6_output / "full" / "prior_summary.csv")
    coexpression_gate = json.loads(
        (output_dir / "programme_specific_coexpression_gate.json").read_text(encoding="utf-8")
    )
    if coexpression_gate["status"] != "FAIL":
        raise RuntimeError("Programme-only figure is reserved for a failed co-expression follow-up gate")

    examples = [
        load_example(base, root, phase6_output, programmes, tasks, example)
        for example in choose_examples(phase6_output)
    ]
    gene_rows: list[dict] = []
    for example in examples:
        for index, gene in enumerate(example["selected_genes"]):
            gene_rows.append(
                {
                    "cohort": example["cohort"],
                    "train": example["train"],
                    "test": example["test"],
                    "programme": example["programme"],
                    "gene": gene,
                    "morpho_pcc": example["gene_concordance"]["Morpho-FM"][index],
                    "istar_pcc": example["gene_concordance"]["iStar"][index],
                }
            )
    pd.DataFrame(gene_rows).to_csv(
        output_dir / "suppfig16_example_gene_concordance.csv", index=False
    )

    programme_names = [item["name"] for item in programmes]
    if len(programme_names) != 50 or len(set(programme_names)) != 50:
        raise RuntimeError("Expected 50 unique Hallmark programmes")
    if len(programme_section) != 600:
        raise RuntimeError(f"Expected 600 section-programme rows, observed {len(programme_section)}")
    if not np.isfinite(programme_section["score_advantage"]).all():
        raise RuntimeError("Non-finite programme-score advantage")
    if not (programme_section["score_advantage"] > 0.0).all():
        raise RuntimeError("The programme-only figure requires all 600 paired advantages to be positive")
    if len(tissue) != 100 or tissue[["cohort", "programme"]].duplicated().any():
        raise RuntimeError("Expected one tissue summary row per cohort and programme")
    for cohort, expected_sections in (("Prostate", 4), ("Kidney", 8)):
        selected = tissue.loc[tissue["cohort"].eq(cohort)]
        if len(selected) != 50:
            raise RuntimeError(f"Expected 50 programmes for {cohort}")
        if not (
            selected["score_positive_sections"].eq(expected_sections)
            & selected["n_sections"].eq(expected_sections)
        ).all():
            raise RuntimeError(f"Not every {cohort} programme is positive in all sections")

    pooled_order = (
        programme_tissue.groupby("programme")["score_advantage"]
        .median()
        .sort_values(kind="stable")
        .index.tolist()
    )
    section_order = [
        ("Prostate", test)
        for test in sorted(
            phase6_section.loc[phase6_section["cohort"].eq("Prostate"), "test"]
        )
    ] + [
        ("Kidney", test)
        for test in sorted(
            phase6_section.loc[phase6_section["cohort"].eq("Kidney"), "test"]
        )
    ]
    section_index = pd.MultiIndex.from_tuples(section_order, names=["cohort", "test"])
    score_matrix = (
        programme_section.pivot(
            index=["cohort", "test"], columns="programme", values="score_advantage"
        )
        .reindex(index=section_index, columns=pooled_order)
        .to_numpy(dtype=float)
    )
    if score_matrix.shape != (12, 50) or not np.isfinite(score_matrix).all():
        raise RuntimeError(f"Invalid programme heatmap matrix: {score_matrix.shape}")
    section_effects = (
        phase6_section.set_index(["cohort", "test"])
        .reindex(section_index)["programme_advantage"]
        .to_numpy(dtype=float)
    )

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.3,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    cohort_colors = {"Prostate": "#2C6EAE", "Kidney": "#D97706"}
    score_cmap = LinearSegmentedColormap.from_list(
        "programme_advantage", ["#F4F4F2", "#A9D5C5", "#167D6A"]
    )
    spatial_cmap = LinearSegmentedColormap.from_list(
        "programme_score", ["#2C6EAE", "#F7F7F5", "#D97706"]
    )

    fig = plt.figure(figsize=(7.1, 7.5), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.70, 3.15, 1.65],
        left=0.105,
        right=0.955,
        top=0.935,
        bottom=0.115,
        hspace=0.62,
    )

    # a, all 50 programme effects in every held-out section.
    a_grid = outer[0, 0].subgridspec(
        1, 4, width_ratios=[4.55, 0.10, 1.05, 1.20], wspace=0.42
    )
    a_heat = fig.add_subplot(a_grid[0, 0])
    heat_limit = float(np.ceil(score_matrix.max() * 10.0) / 10.0)
    heat_norm = Normalize(vmin=0.0, vmax=heat_limit)
    heat = a_heat.imshow(
        score_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=score_cmap,
        norm=heat_norm,
    )
    a_heat.axhline(3.5, color="white", lw=2.0)
    a_heat.axhline(3.5, color="#555555", lw=0.55)
    a_heat.set_yticks(np.arange(12), [test for _, test in section_order], fontsize=6.4)
    for label, (cohort, _) in zip(
        a_heat.get_yticklabels(), section_order, strict=True
    ):
        label.set_color(cohort_colors[cohort])
        label.set_fontweight("bold")

    anchor_styles = {
        "HALLMARK_ANDROGEN_RESPONSE": (
            "Androgen response",
            cohort_colors["Prostate"],
        ),
        "HALLMARK_HYPOXIA": ("Hypoxia", cohort_colors["Kidney"]),
    }
    anchor_indices = [pooled_order.index(name) for name in anchor_styles]
    a_heat.set_xticks(
        anchor_indices,
        [anchor_styles[name][0] for name in anchor_styles],
        rotation=24,
        ha="right",
    )
    for label, name in zip(a_heat.get_xticklabels(), anchor_styles, strict=True):
        label.set_color(anchor_styles[name][1])
        label.set_fontweight("bold")
        label.set_fontsize(6.2)
    for name, index in zip(anchor_styles, anchor_indices, strict=True):
        a_heat.add_patch(
            Rectangle(
                (index - 0.5, -0.5),
                1.0,
                12.0,
                fill=False,
                edgecolor=anchor_styles[name][1],
                linewidth=0.85,
                clip_on=False,
            )
        )
    a_heat.tick_params(axis="both", length=0, pad=2)
    a_heat.spines[:].set_visible(False)
    for y_position, cohort in ((0.84, "Prostate"), (0.32, "Kidney")):
        a_heat.text(
            -0.13,
            y_position,
            cohort,
            transform=a_heat.transAxes,
            rotation=90,
            color=cohort_colors[cohort],
            fontsize=6.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
    a_heat.text(
        -0.18,
        1.24,
        "a",
        transform=a_heat.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    a_heat.text(
        -0.10,
        1.24,
        "Programme-level agreement across held-out sections",
        transform=a_heat.transAxes,
        fontsize=8.4,
        fontweight="bold",
        va="top",
    )
    a_heat.text(
        -0.10,
        1.09,
        "All 50 programmes had positive paired advantages in 4/4 prostate and 8/8 kidney sections",
        transform=a_heat.transAxes,
        fontsize=6.5,
        color="#555555",
        va="top",
    )

    a_cbar = fig.add_subplot(a_grid[0, 1])
    colorbar = fig.colorbar(heat, cax=a_cbar)
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    colorbar.set_label("Programme-score advantage", fontsize=6.2, labelpad=3)

    a_section = fig.add_subplot(a_grid[0, 2], sharey=a_heat)
    section_y = np.arange(12)
    for cohort, marker in (("Prostate", "o"), ("Kidney", "s")):
        mask = np.asarray([item[0] == cohort for item in section_order])
        a_section.scatter(
            section_effects[mask],
            section_y[mask],
            s=20,
            marker=marker,
            color=cohort_colors[cohort],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        median = float(np.median(section_effects[mask]))
        limits = (-0.35, 3.35) if cohort == "Prostate" else (3.65, 11.35)
        a_section.plot(
            [median, median], limits, color=cohort_colors[cohort], lw=1.2, zorder=2
        )
    a_section.set_xlim(0.35, 0.84)
    a_section.set_xticks([0.4, 0.6, 0.8])
    a_section.set_title("Section median", fontsize=7.0, fontweight="bold", pad=4)
    a_section.set_xlabel("Advantage", fontsize=6.3, labelpad=1)
    a_section.tick_params(axis="x", labelsize=5.8, length=2, pad=1)
    a_section.tick_params(axis="y", left=False, labelleft=False)
    a_section.grid(axis="x", color="#E1E1E1", lw=0.45)
    a_section.spines[["top", "right", "left"]].set_visible(False)
    a_section.text(
        0.02,
        0.97,
        "+0.439",
        transform=a_section.transAxes,
        color=cohort_colors["Prostate"],
        fontsize=5.8,
        va="top",
    )
    a_section.text(
        0.98,
        0.42,
        "+0.670",
        transform=a_section.transAxes,
        color=cohort_colors["Kidney"],
        fontsize=5.8,
        ha="right",
        va="center",
    )

    a_prior = fig.add_subplot(a_grid[0, 3])
    prior_order = ["Single-slide", "Multi-slide"]
    prior_y = np.asarray([1, 0])
    prior_values = np.asarray(
        [
            float(
                prior.loc[
                    prior["setting"].eq(setting), "programme_advantage"
                ].iloc[0]
            )
            for setting in prior_order
        ]
    )
    for y_value, value, marker in zip(
        prior_y, prior_values, ("o", "D"), strict=True
    ):
        a_prior.plot(
            [0.0, value], [y_value, y_value], color="#B7B7B7", lw=1.0, zorder=1
        )
        a_prior.scatter(
            value,
            y_value,
            s=27,
            marker=marker,
            color="#7A3E8E",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        a_prior.text(
            value,
            y_value + 0.17,
            f"{value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
        )
    a_prior.set_xlim(0.0, 0.28)
    a_prior.set_ylim(-0.55, 1.55)
    a_prior.set_yticks(prior_y, ["Single", "Multi"], fontsize=6.1)
    a_prior.set_xticks([0.0, 0.1, 0.2])
    a_prior.set_title("Matched CellFM prior", fontsize=7.0, fontweight="bold", pad=4)
    a_prior.set_xlabel("Pretrained - random", fontsize=6.3, labelpad=1)
    a_prior.tick_params(axis="x", labelsize=5.8, length=2, pad=1)
    a_prior.tick_params(axis="y", length=0, pad=2)
    a_prior.grid(axis="x", color="#E1E1E1", lw=0.45)
    a_prior.spines[["top", "right", "left"]].set_visible(False)

    # b, two prespecified programme-score maps with matched scales within rows.
    b_grid = outer[1, 0].subgridspec(
        2,
        5,
        width_ratios=[1.22, 1.0, 1.0, 1.0, 0.045],
        hspace=0.30,
        wspace=0.10,
    )
    b_axes = []
    b_label_axes = []
    for row, example in enumerate(examples):
        all_scores = np.concatenate(
            [example["scores"][name] for name in ("Truth", "Morpho-FM", "iStar")]
        )
        limit = max(float(np.quantile(np.abs(all_scores), 0.98)), 0.25)
        norm = Normalize(vmin=-limit, vmax=limit)
        row_axes = []
        for column, method in enumerate(("Truth", "Morpho-FM", "iStar")):
            ax = fig.add_subplot(b_grid[row, column + 1])
            coords = example["coords"]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=example["scores"][method],
                cmap=spatial_cmap,
                norm=norm,
                s=3.0,
                linewidths=0,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row == 0:
                display_name = "Measured" if method == "Truth" else method
                ax.set_title(display_name, fontsize=7.2, fontweight="bold", pad=3)
            if method != "Truth":
                ax.set_xlabel(
                    f"PCC = {example['score_concordance'][method]:.3f}",
                    fontsize=6.2,
                    labelpad=1,
                )
            row_axes.append(ax)
        label_ax = fig.add_subplot(b_grid[row, 0])
        label_ax.set_axis_off()
        cax = fig.add_subplot(b_grid[row, 4])
        colorbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=spatial_cmap), cax=cax
        )
        colorbar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
        colorbar.set_label("Programme score", fontsize=5.7, labelpad=2)
        title = short_programme_name(example["programme"]).title()
        label_ax.text(
            0.96,
            0.50,
            f"{example['cohort']}\n{title}\n{example['train']} -> {example['test']}\n"
            f"{example['score_n_genes']} variable genes",
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.5,
            color=cohort_colors[example["cohort"]],
            fontweight="bold",
        )
        b_axes.extend(row_axes)
        b_label_axes.append(label_ax)
    b_label_axes[0].text(
        -0.18,
        1.22,
        "b",
        transform=b_label_axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    b_label_axes[0].text(
        0.02,
        1.22,
        "Prespecified tissue-relevant programmes at observed locations",
        transform=b_label_axes[0].transAxes,
        fontsize=8.4,
        fontweight="bold",
        va="top",
        clip_on=False,
    )

    # c, gene-wise PCC for the high-variance members shown in the mapped programmes.
    c_grid = outer[2, 0].subgridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.34
    )
    c_axes = []
    member_image = None
    for column, example in enumerate(examples):
        ax = fig.add_subplot(c_grid[0, column])
        genes = example["selected_genes"]
        morpho = np.asarray(example["gene_concordance"]["Morpho-FM"], dtype=float)
        istar = np.asarray(example["gene_concordance"]["iStar"], dtype=float)
        matrix = np.vstack([morpho, istar])
        member_image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=score_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xticks(
            np.arange(len(genes)), genes, rotation=36, ha="right", fontsize=6.2
        )
        ax.set_yticks([0, 1], ["Morpho-FM", "iStar"], fontsize=6.5)
        ax.set_xticks(np.arange(-0.5, len(genes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=0, pad=2)
        ax.spines[:].set_visible(False)
        for row_index in range(2):
            for gene_index in range(len(genes)):
                value = float(matrix[row_index, gene_index])
                text_color = "white" if value >= 0.68 else "#222222"
                ax.text(
                    gene_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color=text_color,
                )
        title = short_programme_name(example["programme"]).title()
        positive = int((morpho > istar).sum())
        median_delta = float(np.median(morpho - istar))
        ax.text(
            0.5,
            1.20,
            f"{example['cohort']} | {title}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="bold",
        )
        ax.text(
            0.5,
            1.07,
            f"Morpho-FM higher for {positive}/8 genes; median advantage {median_delta:+.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.9,
            color="#555555",
        )
        c_axes.append(ax)
    c_colorbar_axis = fig.add_subplot(c_grid[0, 2])
    colorbar = fig.colorbar(member_image, cax=c_colorbar_axis)
    colorbar.ax.tick_params(labelsize=5.6, length=2, pad=1)
    colorbar.set_label("Per-gene PCC", fontsize=6.0, labelpad=2)
    c_axes[0].text(
        -0.20,
        1.52,
        "c",
        transform=c_axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    c_axes[0].text(
        -0.09,
        1.52,
        "Gene-wise agreement for high-variance programme members",
        transform=c_axes[0].transAxes,
        fontsize=8.4,
        fontweight="bold",
        va="top",
        clip_on=False,
    )
    fig.text(
        0.985,
        0.018,
        "Member genes were selected by measured-expression variance before method comparison.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "phase6_named_programme_summary.png"
    pdf = figure_dir / "phase6_named_programme_summary.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"[Figure saved] {png}")
    print(f"[Figure saved] {pdf}")
    print("[Co-expression omitted] Programme-specific follow-up gate was FAIL")
    return [png, pdf]


def load_h5ad_background(root: Path, slide: str) -> tuple[np.ndarray, np.ndarray]:
    path = root / "data" / "spatial_data" / f"{slide}.h5ad"
    with h5py.File(path, "r") as handle:
        image = np.asarray(handle["uns/spatial/ST/images/downscaled_fullres"][:])
        scale = float(
            handle["uns/spatial/ST/scalefactors/tissue_downscaled_fullres_scalef"][()]
        )
        coords = np.asarray(handle["obsm/spatial"][:], dtype=np.float64) * scale
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid H&E image in {path}: {image.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Invalid spatial coordinates in {path}: {coords.shape}")
    return image, coords


def plot_figure(root: Path, phase6_output: Path, output_dir: Path, figure_dir: Path) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    base = load_phase6_module()
    programmes = base.load_programmes(phase6_output)
    tasks = base.build_tasks(root)
    programme_section, programme_tissue = build_programme_score_summary(phase6_output)
    phase6_section = pd.read_csv(phase6_output / "full" / "section_summary.csv")
    prior = pd.read_csv(phase6_output / "full" / "prior_summary.csv")
    coexpression_gate = json.loads(
        (output_dir / "programme_specific_coexpression_gate.json").read_text(encoding="utf-8")
    )
    if coexpression_gate["status"] != "FAIL":
        raise RuntimeError("Programme-only figure is reserved for a failed co-expression follow-up gate")

    examples = [
        load_example(base, root, phase6_output, programmes, tasks, example)
        for example in choose_examples(phase6_output)
    ]
    gene_rows: list[dict] = []
    for example in examples:
        for index, gene in enumerate(example["selected_genes"]):
            gene_rows.append(
                {
                    "cohort": example["cohort"],
                    "train": example["train"],
                    "test": example["test"],
                    "programme": example["programme"],
                    "gene": gene,
                    "morpho_pcc": example["gene_concordance"]["Morpho-FM"][index],
                    "istar_pcc": example["gene_concordance"]["iStar"][index],
                }
            )
    pd.DataFrame(gene_rows).to_csv(
        output_dir / "suppfig16_example_gene_concordance.csv", index=False
    )

    programme_names = [item["name"] for item in programmes]
    if len(programme_names) != 50 or len(set(programme_names)) != 50:
        raise RuntimeError("Expected 50 unique Hallmark programmes")
    if len(programme_section) != 600:
        raise RuntimeError(f"Expected 600 section-programme rows, observed {len(programme_section)}")
    if not np.isfinite(
        programme_section[["score_morpho", "score_istar", "score_advantage"]]
    ).all().all():
        raise RuntimeError("Non-finite programme-score result")
    if not (programme_section["score_advantage"] > 0.0).all():
        raise RuntimeError("Expected all 600 paired programme-score advantages to be positive")

    pooled_order = (
        programme_tissue.groupby("programme")["score_advantage"]
        .median()
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    section_order = [
        (cohort, slide)
        for cohort, slides in base.COHORT_SLIDES.items()
        for slide in slides
    ]
    section_index = pd.MultiIndex.from_tuples(section_order, names=["cohort", "test"])

    def absolute_matrix(column: str) -> np.ndarray:
        matrix = (
            programme_section.pivot(
                index=["cohort", "test"], columns="programme", values=column
            )
            .reindex(index=section_index, columns=pooled_order)
            .to_numpy(dtype=float)
        )
        if matrix.shape != (12, 50) or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid {column} matrix: {matrix.shape}")
        return matrix

    morpho_matrix = absolute_matrix("score_morpho")
    istar_matrix = absolute_matrix("score_istar")
    section_frame = phase6_section.set_index(["cohort", "test"]).reindex(section_index)
    if section_frame[["programme_morpho", "programme_istar"]].isna().any().any():
        raise RuntimeError("Missing section-level programme summary")

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    method_colors = {"Morpho-FM": "#E64B35", "iStar": "#4DBBD5"}
    cohort_colors = {"Prostate": "#2C6EAE", "Kidney": "#D97706"}
    neutral = "#A6A6A6"
    programme_cmap = plt.get_cmap("magma_r")
    spatial_cmap = plt.get_cmap("viridis")
    anchor_styles = {
        "HALLMARK_ANDROGEN_RESPONSE": ("Androgen response", cohort_colors["Prostate"]),
        "HALLMARK_HYPOXIA": ("Hypoxia", cohort_colors["Kidney"]),
    }
    anchor_indices = [pooled_order.index(name) for name in anchor_styles]
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Panel a: absolute, shared-scale method views plus direct paired comparisons.
    fig_a = plt.figure(figsize=(7.35, 3.35), facecolor="white")
    grid_a = fig_a.add_gridspec(
        1,
        5,
        width_ratios=[2.15, 2.15, 0.08, 1.38, 1.12],
        left=0.075,
        right=0.985,
        top=0.74,
        bottom=0.22,
        wspace=0.34,
    )
    heat_norm = Normalize(vmin=-0.55, vmax=1.0)
    heat_axes = []
    heat_image = None
    for column, (method, matrix) in enumerate(
        (("Morpho-FM", morpho_matrix), ("iStar", istar_matrix))
    ):
        ax = fig_a.add_subplot(grid_a[0, column])
        heat_image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=programme_cmap,
            norm=heat_norm,
        )
        ax.axhline(3.5, color="white", lw=2.0)
        ax.axhline(3.5, color="#404040", lw=0.55)
        ax.set_title(method, fontsize=8.3, fontweight="bold", pad=5)
        ax.set_xticks(
            anchor_indices,
            [anchor_styles[name][0] for name in anchor_styles],
            rotation=28,
            ha="right",
            fontsize=6.0,
        )
        for label, name in zip(ax.get_xticklabels(), anchor_styles, strict=True):
            label.set_color(anchor_styles[name][1])
            label.set_fontweight("bold")
        for name, index in zip(anchor_styles, anchor_indices, strict=True):
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -0.5),
                    1.0,
                    12.0,
                    fill=False,
                    edgecolor=anchor_styles[name][1],
                    linewidth=0.9,
                    clip_on=False,
                )
            )
        if column == 0:
            ax.set_yticks(
                np.arange(12), [slide for _, slide in section_order], fontsize=6.2
            )
            for label, (cohort, _) in zip(
                ax.get_yticklabels(), section_order, strict=True
            ):
                label.set_color(cohort_colors[cohort])
                label.set_fontweight("bold")
            ax.text(
                -0.18,
                0.83,
                "Prostate",
                transform=ax.transAxes,
                rotation=90,
                color=cohort_colors["Prostate"],
                fontsize=6.3,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                -0.18,
                0.31,
                "Kidney",
                transform=ax.transAxes,
                rotation=90,
                color=cohort_colors["Kidney"],
                fontsize=6.3,
                fontweight="bold",
                ha="center",
                va="center",
            )
        else:
            ax.set_yticks([])
        ax.tick_params(axis="both", length=0, pad=2)
        ax.spines[:].set_visible(False)
        heat_axes.append(ax)

    cax = fig_a.add_subplot(grid_a[0, 2])
    colorbar = fig_a.colorbar(heat_image, cax=cax)
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    colorbar.set_label("Programme-score PCC", fontsize=6.2, labelpad=3)

    section_ax = fig_a.add_subplot(grid_a[0, 3])
    y = np.arange(12)
    section_morpho = section_frame["programme_morpho"].to_numpy(dtype=float)
    section_istar = section_frame["programme_istar"].to_numpy(dtype=float)
    section_ax.hlines(y, section_istar, section_morpho, color=neutral, lw=1.25, zorder=1)
    section_ax.scatter(
        section_istar,
        y,
        s=18,
        marker="s",
        color=method_colors["iStar"],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    section_ax.scatter(
        section_morpho,
        y,
        s=23,
        marker="o",
        color=method_colors["Morpho-FM"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    section_ax.set_xlim(0.0, 1.02)
    section_ax.set_ylim(11.6, -0.6)
    section_ax.set_yticks([])
    section_ax.set_xticks([0.0, 0.5, 1.0])
    section_ax.set_xlabel("Median PCC", fontsize=6.4, labelpad=2)
    section_ax.set_title("Held-out sections", fontsize=7.2, fontweight="bold", pad=5)
    section_ax.grid(axis="x", color="#E3E3E3", lw=0.45)
    section_ax.spines[["top", "right", "left"]].set_visible(False)
    section_ax.tick_params(axis="x", labelsize=5.8, length=2)
    section_ax.text(
        0.02,
        -0.18,
        "Median gain  +0.439 prostate | +0.670 kidney",
        transform=section_ax.transAxes,
        fontsize=5.6,
        color="#4F4F4F",
    )

    prior_ax = fig_a.add_subplot(grid_a[0, 4])
    prior_order = ["Single-slide", "Multi-slide"]
    prior_y = np.arange(2)
    prior_frame = prior.set_index("setting").reindex(prior_order)
    random_values = prior_frame["programme_random"].to_numpy(dtype=float)
    pretrained_values = prior_frame["programme_pretrained"].to_numpy(dtype=float)
    prior_gains = prior_frame["programme_advantage"].to_numpy(dtype=float)
    prior_ax.hlines(prior_y, random_values, pretrained_values, color=neutral, lw=1.5)
    prior_ax.scatter(
        random_values,
        prior_y,
        s=22,
        marker="s",
        color="#777777",
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    prior_ax.scatter(
        pretrained_values,
        prior_y,
        s=28,
        marker="o",
        color=method_colors["Morpho-FM"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    for y_value, random_value, pretrained_value, gain in zip(
        prior_y, random_values, pretrained_values, prior_gains, strict=True
    ):
        prior_ax.text(
            (random_value + pretrained_value) / 2.0,
            y_value - 0.18,
            f"{gain:+.3f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
        )
    prior_ax.set_xlim(0.0, 1.02)
    prior_ax.set_ylim(1.55, -0.55)
    prior_ax.set_yticks(prior_y, ["Single", "Multi"], fontsize=6.0)
    prior_ax.set_xticks([0.0, 0.5, 1.0])
    prior_ax.set_xlabel("Median PCC", fontsize=6.4, labelpad=2)
    prior_ax.set_title("Initialization", fontsize=7.2, fontweight="bold", pad=5)
    prior_ax.grid(axis="x", color="#E3E3E3", lw=0.45)
    prior_ax.spines[["top", "right", "left"]].set_visible(False)
    prior_ax.tick_params(axis="x", labelsize=5.8, length=2)
    prior_ax.tick_params(axis="y", length=0, pad=2)

    fig_a.text(0.015, 0.955, "a", fontsize=11, fontweight="bold", va="top")
    fig_a.text(
        0.055,
        0.955,
        "Direct programme-level agreement across held-out sections",
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    fig_a.text(
        0.055,
        0.875,
        "Morpho-FM achieved higher programme-score concordance in all 600 section-programme comparisons",
        fontsize=6.7,
        color="#4F4F4F",
        va="top",
    )
    section_ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="", color=method_colors["Morpho-FM"],
                label="Morpho-FM", markersize=4.5,
            ),
            Line2D(
                [0], [0], marker="s", linestyle="", color=method_colors["iStar"],
                label="iStar", markersize=4.0,
            ),
        ],
        frameon=False,
        fontsize=5.7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        handletextpad=0.3,
        columnspacing=0.8,
    )
    panel_a_path = figure_dir / "phase6_named_programme_panel_a.png"
    panel_a_pdf = figure_dir / "phase6_named_programme_panel_a.pdf"
    fig_a.savefig(panel_a_path, dpi=600, facecolor="white")
    fig_a.savefig(panel_a_pdf, dpi=600, facecolor="white")
    plt.close(fig_a)
    saved.extend([panel_a_path, panel_a_pdf])

    # Panel b: H&E-backed spatial programme-score maps using the manuscript magma scale.
    fig_b = plt.figure(figsize=(7.35, 4.65), facecolor="white")
    grid_b = fig_b.add_gridspec(
        2,
        5,
        width_ratios=[0.92, 1.0, 1.0, 1.0, 0.045],
        left=0.045,
        right=0.975,
        top=0.83,
        bottom=0.055,
        hspace=0.28,
        wspace=0.10,
    )
    label_axes = []
    for row, example in enumerate(examples):
        image, coords = load_h5ad_background(root, example["test"])
        if len(coords) != len(example["scores"]["Truth"]):
            raise RuntimeError(
                f"H&E coordinate count does not match programme scores for {example['test']}: "
                f"{len(coords)} versus {len(example['scores']['Truth'])}"
            )
        all_scores = np.concatenate(
            [example["scores"][name] for name in ("Truth", "Morpho-FM", "iStar")]
        )
        low, high = np.quantile(all_scores[np.isfinite(all_scores)], [0.02, 0.98])
        if high <= low:
            high = low + 1.0
        norm = Normalize(vmin=float(low), vmax=float(high))
        mapped = None
        for column, method in enumerate(("Truth", "Morpho-FM", "iStar")):
            ax = fig_b.add_subplot(grid_b[row, column + 1])
            ax.imshow(image, interpolation="bilinear")
            mapped = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=example["scores"][method],
                cmap=spatial_cmap,
                norm=norm,
                s=4.5,
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title(
                    "Measured" if method == "Truth" else method,
                    fontsize=7.5,
                    fontweight="bold",
                    pad=3,
                )
            if method != "Truth":
                ax.set_xlabel(
                    f"PCC = {example['score_concordance'][method]:.3f}",
                    fontsize=6.4,
                    labelpad=1,
                )
        label_ax = fig_b.add_subplot(grid_b[row, 0])
        label_ax.set_axis_off()
        label_axes.append(label_ax)
        title = short_programme_name(example["programme"]).title()
        label_ax.text(
            0.96,
            0.50,
            f"{example['cohort']}\n{title}\n{example['train']} -> {example['test']}\n"
            f"{example['score_n_genes']} variable genes",
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=6.7,
            color=cohort_colors[example["cohort"]],
            fontweight="bold",
        )
        cax = fig_b.add_subplot(grid_b[row, 4])
        colorbar = fig_b.colorbar(mapped, cax=cax)
        colorbar.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
        colorbar.set_label("Standardized programme score", fontsize=5.8, labelpad=2)

    fig_b.text(0.015, 0.955, "b", fontsize=11, fontweight="bold", va="top")
    fig_b.text(
        0.055,
        0.955,
        "Tissue-relevant programme scores retain their histological context",
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    fig_b.text(
        0.055,
        0.885,
        "Observed-location scores are overlaid on the matched H&E image; each row uses one shared colour scale",
        fontsize=6.7,
        color="#4F4F4F",
        va="top",
    )
    panel_b_path = figure_dir / "phase6_named_programme_panel_b.png"
    panel_b_pdf = figure_dir / "phase6_named_programme_panel_b.pdf"
    fig_b.savefig(panel_b_path, dpi=600, facecolor="white")
    fig_b.savefig(panel_b_pdf, dpi=600, facecolor="white")
    plt.close(fig_b)
    saved.extend([panel_b_path, panel_b_pdf])

    # Panel c: direct gene-wise paired comparisons make the size and direction visible.
    fig_c = plt.figure(figsize=(7.35, 3.55), facecolor="white")
    grid_c = fig_c.add_gridspec(
        1,
        2,
        left=0.11,
        right=0.95,
        top=0.73,
        bottom=0.21,
        wspace=0.38,
    )
    for column, example in enumerate(examples):
        ax = fig_c.add_subplot(grid_c[0, column])
        genes = example["selected_genes"]
        morpho = np.asarray(example["gene_concordance"]["Morpho-FM"], dtype=float)
        istar = np.asarray(example["gene_concordance"]["iStar"], dtype=float)
        delta = morpho - istar
        y = np.arange(len(genes))
        for y_value, start, end in zip(y, istar, morpho, strict=True):
            ax.annotate(
                "",
                xy=(end, y_value),
                xytext=(start, y_value),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#9D9D9D",
                    "lw": 1.15,
                    "mutation_scale": 7,
                    "shrinkA": 3,
                    "shrinkB": 4,
                },
                zorder=1,
            )
        ax.scatter(
            istar,
            y,
            s=24,
            marker="s",
            color=method_colors["iStar"],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        ax.scatter(
            morpho,
            y,
            s=31,
            marker="o",
            color=method_colors["Morpho-FM"],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        for y_value, value in zip(y, delta, strict=True):
            ax.text(
                1.035,
                y_value,
                f"{value:+.2f}",
                ha="left",
                va="center",
                fontsize=5.9,
                color="#333333",
                fontweight="bold" if value > 0 else "normal",
            )
        ax.set_xlim(-0.05, 1.14)
        ax.set_ylim(len(genes) - 0.5, -0.5)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks(y, genes, fontsize=6.5)
        for label in ax.get_yticklabels():
            label.set_fontstyle("italic")
        ax.set_xlabel("Per-gene PCC", fontsize=6.6, labelpad=2)
        title = short_programme_name(example["programme"]).title()
        ax.set_title(
            f"{example['cohort']} | {title}",
            fontsize=7.7,
            fontweight="bold",
            pad=17,
        )
        positive = int((delta > 0).sum())
        ax.text(
            0.5,
            1.055,
            f"Morpho-FM higher for {positive}/8 genes | median gain {np.median(delta):+.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="#4F4F4F",
        )
        ax.text(
            0.91,
            1.01,
            "Delta",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.6,
            color="#666666",
        )
        ax.grid(axis="x", color="#E3E3E3", lw=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=6.0, length=2)

    fig_c.text(0.015, 0.955, "c", fontsize=11, fontweight="bold", va="top")
    fig_c.text(
        0.055,
        0.955,
        "Direct gene-wise agreement for high-variance programme members",
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    fig_c.text(
        0.055,
        0.885,
        "Arrows point from iStar to Morpho-FM; genes were selected by measured-expression variance before comparison",
        fontsize=6.7,
        color="#4F4F4F",
        va="top",
    )
    fig_c.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="", color=method_colors["Morpho-FM"],
                label="Morpho-FM", markersize=5.0,
            ),
            Line2D(
                [0], [0], marker="s", linestyle="", color=method_colors["iStar"],
                label="iStar", markersize=4.5,
            ),
        ],
        frameon=False,
        fontsize=6.2,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    panel_c_path = figure_dir / "phase6_named_programme_panel_c.png"
    panel_c_pdf = figure_dir / "phase6_named_programme_panel_c.pdf"
    fig_c.savefig(panel_c_path, dpi=600, facecolor="white")
    fig_c.savefig(panel_c_pdf, dpi=600, facecolor="white")
    plt.close(fig_c)
    saved.extend([panel_c_path, panel_c_pdf])

    for path in saved:
        print(f"[Figure saved] {path}")
    print("[Composite unchanged] Review the three separate panels before replacing fig_supp_10.png")
    return saved


def plot_supplementary_figure_10(output_dir: Path, output_path: Path) -> Path:
    import fitz
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    source = output_dir / "suppfig16_example_gene_concordance.csv"
    data = pd.read_csv(source)
    required_columns = {
        "cohort",
        "train",
        "test",
        "programme",
        "gene",
        "morpho_pcc",
        "istar_pcc",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise RuntimeError(f"Missing Supplementary Figure 10 columns: {sorted(missing)}")
    if len(data) != 16 or data.duplicated(["cohort", "programme", "gene"]).any():
        raise RuntimeError("Expected 16 unique programme-member gene rows")
    if not np.isfinite(data[["morpho_pcc", "istar_pcc"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite programme-member PCC")
    if not (
        data[["morpho_pcc", "istar_pcc"]].ge(-1.0)
        & data[["morpho_pcc", "istar_pcc"]].le(1.0)
    ).all().all():
        raise RuntimeError("Programme-member PCC lies outside [-1, 1]")

    panels = [
        {
            "cohort": "Prostate",
            "programme": "HALLMARK_ANDROGEN_RESPONSE",
            "title": "Prostate | Androgen response",
            "genes": ["KLK3", "KLK2", "SCD", "NKX3-1", "PLPP1", "TPD52", "ABCC4", "PMEPA1"],
            "expected_positive": 8,
        },
        {
            "cohort": "Kidney",
            "programme": "HALLMARK_HYPOXIA",
            "title": "Kidney | Hypoxia",
            "genes": ["MT2A", "IGFBP3", "CCN2", "SERPINE1", "ALDOB", "DCN", "FOS", "ENO1"],
            "expected_positive": 7,
        },
    ]

    arial = Path(r"C:\Windows\Fonts\arial.ttf")
    arial_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    arial_italic = Path(r"C:\Windows\Fonts\ariali.ttf")
    for font in (arial, arial_bold, arial_italic):
        if not font.exists():
            raise FileNotFoundError(font)
        font_manager.fontManager.addfont(str(font))

    width_mm = 170.0
    height_mm = 76.0
    dpi = 600.0
    method_colors = {"Morpho-FM": "#E64B35", "iStar": "#4DBBD5"}
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.8,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor="white",
        sharex=True,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.80,
        bottom=0.17,
        wspace=0.31,
    )

    for axis, panel in zip(axes, panels, strict=True):
        selected = (
            data.loc[
                data["cohort"].eq(panel["cohort"])
                & data["programme"].eq(panel["programme"])
            ]
            .set_index("gene")
            .reindex(panel["genes"])
        )
        if selected[["morpho_pcc", "istar_pcc"]].isna().any().any():
            raise RuntimeError(f"Missing genes for {panel['title']}")
        morpho = selected["morpho_pcc"].to_numpy(dtype=float)
        istar = selected["istar_pcc"].to_numpy(dtype=float)
        positive = int(np.sum(morpho > istar))
        if positive != panel["expected_positive"]:
            raise RuntimeError(
                f"Unexpected advantage count for {panel['title']}: {positive}/8"
            )
        if panel["cohort"] == "Kidney":
            exceptions = selected.index[morpho <= istar].tolist()
            if exceptions != ["ALDOB"]:
                raise RuntimeError(f"Unexpected kidney exception: {exceptions}")

        y = np.arange(len(panel["genes"]))
        for y_value, left, right in zip(y, istar, morpho, strict=True):
            axis.plot(
                [left, right],
                [y_value, y_value],
                color="#A6A6A6",
                linewidth=1.15,
                solid_capstyle="round",
                zorder=1,
            )
        axis.scatter(
            istar,
            y,
            s=25,
            marker="s",
            color=method_colors["iStar"],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        axis.scatter(
            morpho,
            y,
            s=31,
            marker="o",
            color=method_colors["Morpho-FM"],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        axis.set_xlim(0.0, 1.02)
        axis.set_ylim(len(panel["genes"]) - 0.45, -0.55)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_yticks(y, panel["genes"], fontsize=7.0)
        for label in axis.get_yticklabels():
            label.set_fontstyle("italic")
        axis.set_title(panel["title"], fontsize=7.8, fontweight="normal", pad=5)
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=3)
        axis.tick_params(axis="x", labelsize=6.6, length=2.0)

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=method_colors["Morpho-FM"],
                label="Morpho-FM",
                markersize=5.0,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                color=method_colors["iStar"],
                label="iStar",
                markersize=4.6,
            ),
        ],
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    figure.supxlabel("Per-gene PCC", fontsize=7.0, y=0.055)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", dpi=dpi, facecolor="white")
    plt.close(figure)

    pixmap = fitz.Pixmap(str(output_path))
    png_size = (pixmap.width, pixmap.height)
    pixmap = None
    expected_size = (
        round(width_mm / 25.4 * dpi),
        round(height_mm / 25.4 * dpi),
    )
    if any(abs(observed - expected) > 1 for observed, expected in zip(png_size, expected_size)):
        raise RuntimeError(f"Unexpected Supplementary Figure 10 dimensions: {png_size}")
    print(
        f"[verified] Supplementary Figure 10: {width_mm:.1f} x {height_mm:.1f} mm; "
        f"{png_size[0]} x {png_size[1]} px at 600 dpi"
    )
    print(f"[saved] {output_path}")
    return output_path


def self_test() -> None:
    base = load_phase6_module()
    rng = np.random.default_rng(42)
    values = rng.normal(size=(40, 12))
    standardized, valid = standardize_in_place(values.copy())
    truth = correlation_matrix_from_standardized(standardized, np.arange(12))
    rank, error, constant = matrix_metrics(base, truth, truth)
    assert valid.all()
    assert abs(rank - 1.0) < 1e-12
    assert error < 1e-12
    assert not constant

    constant_prediction = np.zeros_like(truth)
    rank, error, constant = matrix_metrics(base, constant_prediction, truth)
    assert rank == 0.0 and error > 0.0 and constant

    synthetic = pd.DataFrame(
        [
            {
                "cohort": "Synthetic",
                "train": f"train_{index}",
                "test": "test",
                "task_id": f"task_{index}",
                "programme": "p",
                "n_genes": 10,
                "morpho_rank": morpho,
                "istar_rank": istar,
                "rank_advantage": morpho - istar,
                "morpho_error": 1.0 - morpho,
                "istar_error": 1.0 - istar,
                "error_advantage": morpho - istar,
            }
            for index, (morpho, istar) in enumerate(zip([0.0, 10.0, 11.0], [0.0, 9.0, 100.0]))
        ]
    )
    section, _, summary = aggregate_programme_coexpression(synthetic)
    assert float(section.iloc[0]["rank_advantage"]) == 0.0
    assert float(summary.iloc[0]["rank_advantage"]) == 0.0
    assert float(section.iloc[0]["morpho_rank"] - section.iloc[0]["istar_rank"]) == 1.0

    non_commuting_rows: list[dict] = []
    for programme, advantages in (("p1", [-1.0, 0.0, 5.0]), ("p2", [9.0, -9.0, -7.0])):
        for index, advantage in enumerate(advantages):
            non_commuting_rows.append(
                {
                    "cohort": "Synthetic",
                    "train": f"train_{index}",
                    "test": "test",
                    "task_id": f"task_{index}",
                    "programme": programme,
                    "n_genes": 10,
                    "morpho_rank": advantage,
                    "istar_rank": 0.0,
                    "rank_advantage": advantage,
                    "morpho_error": 10.0 - advantage,
                    "istar_error": 10.0,
                    "error_advantage": advantage,
                }
            )
    _, _, ordered_summary = aggregate_programme_coexpression(pd.DataFrame(non_commuting_rows))
    assert float(ordered_summary.iloc[0]["rank_advantage"]) == -3.5
    assert float(ordered_summary.iloc[0]["rank_advantage"]) != -1.0
    task = base.build_tasks(ROOT)[0]
    canonical, morpho_genes, _ = base.load_gene_orders(ROOT, "Prostate")
    genes = canonical[::1750][:10]
    gene_map = {gene: index for index, gene in enumerate(morpho_genes)}
    expected = base.matrix_columns(task.morpho_dir / "true_bag.npy", gene_map, genes)
    observed = matrix_columns_sequential(
        base, task.morpho_dir / "true_bag.npy", gene_map, genes
    )
    assert np.array_equal(observed, expected)
    print(
        "[Self-test PASS] matrix metrics, constant prediction, paired aggregation, and sequential I/O"
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    phase6_output = args.phase6_output.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    supp10_output = args.supp10_output.expanduser().resolve()
    if args.command == "self-test":
        self_test()
        return
    if args.command in {"run", "all"}:
        run_analysis(root, phase6_output, output_dir)
    if args.command in {"plot", "all"}:
        plot_figure(root, phase6_output, output_dir, figure_dir)
    if args.command == "supp10":
        plot_supplementary_figure_10(output_dir, supp10_output)


if __name__ == "__main__":
    main()
