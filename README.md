# Morpho-FM

**Spatial molecular reconstruction from routine H&E histology using transcriptomic foundation model priors**

Jinjin Huang<sup>1</sup>, Xiao Feng<sup>1</sup>, Lianghu Qu<sup>1</sup>, Lingling Zheng<sup>1,*</sup>

<sup>1</sup> MOE Key Laboratory of Gene Function and Regulation, State Key Laboratory for Biocontrol,
Innovation Center for Evolutionary Synthetic Biology, School of Life Sciences / School of Agriculture
and Biotechnology, Sun Yat-sen University Shenzhen Campus, Shenzhen 518107, China.

<sup>*</sup> Correspondence: Lingling Zheng
([zhengll33@mail.sysu.edu.cn](mailto:zhengll33@mail.sysu.edu.cn); Tel. +86-0755-23260262)

Morpho-FM is a weakly supervised framework for predicting spatial gene expression from routine
H&E whole-slide images. It conditions a pretrained single-cell transcriptomic foundation-model
prior on local histological neighbourhoods, enabling prediction at measured spatial transcriptomics
locations, dense full-section molecular reconstruction, and re-aggregation of dense outputs back to
the original measurement support for consistency checking.

## Abstract

Routine H&E histology captures tissue architecture at clinical scale, but it does not directly
measure the transcriptional programmes that organize tumour, stromal, vascular, and immune
regions. Spatial transcriptomics provides this molecular context, yet routine use remains limited by
cost, workflow complexity, and sparse sampling. Morpho-FM addresses this gap by mapping cached
whole-slide histology features into a transcriptomic decoder derived from a pretrained single-cell
foundation model. Across harmonized prostate cancer benchmarks, Morpho-FM achieved mean
per-gene Pearson correlations of 0.2855 in rotating single-slide evaluation and 0.2979 in
multi-slide held-out validation. It reproduced this advantage across kidney cancer sections,
retained predictive signal after external transfer to clear-cell renal cell carcinoma, and recovered
ERBB2-enriched tumour compartments, boundary-associated molecular gradients, and
annotation-aligned tissue domains in Xenium and HER2ST breast cancer datasets.

## Workflow

![Morpho-FM workflow](Fig/workflow.png)

**Fig. 1 overview.** Morpho-FM uses a shared coordinate system for prediction at measured
locations and dense full-section reconstruction.

**a, End-to-end workflow.** Raw H&E whole-slide images and paired spatial transcriptomics
measurements are aligned into a common tissue coordinate system. A cached whole-slide histology
feature grid is constructed offline with a Hierarchical Image Pyramid Transformer (HIPT) encoder.
The same grid supports both spot-supervised multiple-instance learning (MIL) and dense decoding
over the tissue mask. Dense outputs can be re-aggregated to the original measurement support to
check consistency with spot-level prediction.

**b, Local disk-shaped MIL bags.** For each measured ST location, Morpho-FM selects grid
positions within the corresponding spot-radius neighbourhood. These local histology features form
a disk-shaped MIL bag tied to that measured molecular observation.

**c, Visual adapter and transcriptomic decoding.** Instance features from each MIL bag are
projected by a lightweight morphology-to-transcriptome adapter into CellFM, used here as the
single-cell transcriptomic foundation-model decoder. The decoder produces instance-level
expression rates, which are mean-aggregated to obtain the bag-level spatial expression prediction.

## Highlights

- **Transcriptomic foundation priors**: cached histology features are adapted into a pretrained
  single-cell transcriptomic decoder, with CellFM used as the main foundation-model prior.
- **Shared-coordinate prediction and reconstruction**: the same HIPT feature grid supports
  prediction at measured ST locations and dense full-section molecular reconstruction.
- **Disk-shaped spot-supervised MIL**: each ST measurement supervises a local bag of grid
  positions within the spot-radius neighbourhood.
- **Consistency-aware dense decoding**: dense tissue-wide outputs can be re-aggregated to the
  original measurement support for direct consistency checking.
- **Count-aware objective**: negative-binomial likelihood is used for overdispersed spatial
  expression counts during spot-level supervision.
- **Benchmark-ready notebooks**: tracked notebooks provide standard Morpho-FM workflows and
  comparisons to representative histology-to-transcriptomics baselines.

## Repository Layout

```text
Morpho-FM/
+-- assets/cellfm/                 # Gene vocabulary files used by CellFM-based runs
+-- benchmark/                     # Baseline benchmark notebooks and HEST dataset helper
+-- configs/st_mil.yaml            # Example training/inference configuration
+-- Fig/workflow.png               # Manuscript workflow figure used in this README
+-- notebooks/                     # Main Morpho-FM and Xenium example notebooks
+-- scripts/convert_cellfm_ckpt.py # MindSpore CellFM checkpoint conversion helper
+-- src/st_pipeline/               # Core data, model, training, inference, and super-resolution code
```

Large datasets, checkpoints, raw results, logs, and local figure-generation outputs are intentionally
kept outside version control.

## Installation

Create an isolated Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

For local development, run commands with the repository root on `PYTHONPATH`:

```bash
export PYTHONPATH=src:$PYTHONPATH
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src;$env:PYTHONPATH"
```

## Data and Model Preparation

Morpho-FM expects preprocessed spatial transcriptomics data and cell-level morphology embeddings.
The example configuration in `configs/st_mil.yaml` contains the key paths:

- `data.h5ad_path`: AnnData file containing measured spatial expression.
- `data.cell_emb_h5`: HDF5 file containing cell morphology embeddings and cell barcodes.
- `data.gene_vocab_path`: CellFM gene vocabulary file, such as `assets/cellfm/expand_gene_info.csv`.
- `model.cellfm_checkpoint`: optional CellFM checkpoint converted to PyTorch format.

If starting from a MindSpore CellFM checkpoint, convert it before training:

```bash
python scripts/convert_cellfm_ckpt.py \
  --ckpt /path/to/CellFM_80M_weight.ckpt \
  --out /path/to/CellFM_80M_weight.pt
```

For CellFM 80M weights, use `assets/cellfm/expand_gene_info.csv` as the gene vocabulary.

## Quick Start

Edit `configs/st_mil.yaml` so that all data, embedding, vocabulary, checkpoint, and output paths
point to local files. Then run training:

```bash
PYTHONPATH=src python src/st_pipeline/train/train_cli.py \
  --config configs/st_mil.yaml
```

Run spot-level prediction from a trained checkpoint:

```bash
PYTHONPATH=src python src/st_pipeline/infer/predict_cli.py \
  --config configs/st_mil.yaml \
  --checkpoint checkpoints/st_mil/best_model.pt
```

Add `--save_instance` to also save cell-level predictions.

## Notebooks

Tracked example notebooks:

- `notebooks/01_st_mil_hest_multi.ipynb`: multi-slide Morpho-FM training and prediction.
- `notebooks/02_st_mil_hest_single.ipynb`: single-slide Morpho-FM training and prediction.
- `notebooks/101_xenium_preprocess.ipynb`: Xenium preprocessing example.
- `notebooks/102_xenium_generate_cache.ipynb`: Xenium cache generation.
- `notebooks/103_xenium_train.ipynb`: Xenium model training.
- `notebooks/104_xenium_predict.ipynb`: Xenium prediction and visualization.

## Benchmark Suite

The public benchmark suite currently includes six representative histology-to-transcriptomics
methods. These notebooks are the clean, tracked entry points intended for GitHub:

| Method | Notebook | Scope |
| --- | --- | --- |
| HisToGene | `benchmark/01_HisToGene_benchmark.ipynb` | Transformer-based spot expression prediction |
| iStar | `benchmark/02_iStar_benchmark.ipynb` | Image-to-ST prediction and spatial enhancement baseline |
| mclSTExp | `benchmark/03_mclSTExp_benchmark.ipynb` | Contrastive learning baseline for ST expression prediction |
| sCellST | `benchmark/04_sCellST_benchmark.ipynb` | Cell-aware spatial transcriptomics prediction baseline |
| THItoGene | `benchmark/05_THItoGene_benchmark.ipynb` | Histology-to-gene transformer baseline |
| HiST | `benchmark/06_HiST_benchmark.ipynb` | Histology-based spatial transcriptomics baseline |

Local protocol variants, including kidney-specific, single-slice, INTxx, cached result, and
method-source directories, are useful for experiments but should stay out of the public README
unless they are intentionally cleaned and tracked.

## External Components

External repositories and pretrained weights are not vendored in this repository. If needed, place
local copies under `third_party/` and keep them untracked:

- CellFM: <https://github.com/biomed-AI/CellFM>
- HEST: <https://github.com/mahmoodlab/hest/>
- LazySlide: <https://github.com/rendeirolab/LazySlide>
- sCellST: <https://github.com/mahmoodlab/sCellST>

## Version-Control Hygiene

Before publishing or pushing changes, check the exact files that will be included:

```bash
git status --short
git diff -- README.md
```

For this README update, the only manuscript figure that needs to accompany the documentation is:

```text
Fig/workflow.png
```

Do not add local outputs such as `figures_*`, `results/`, `checkpoints/`, `logs/`, `scratch/`,
temporary notebooks, or private manuscript files.

## Citation

If you use Morpho-FM, please cite the manuscript:

```bibtex
@article{huang2026morphofm,
  title  = {Morpho-FM: spatial molecular reconstruction from routine H&E histology using transcriptomic foundation model priors},
  author = {Huang, Jinjin and Feng, Xiao and Qu, Lianghu and Zheng, Lingling},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Contact

For questions about the method or manuscript, please contact Lingling Zheng at
[zhengll33@mail.sysu.edu.cn](mailto:zhengll33@mail.sysu.edu.cn).
