# Third-Party Notices

The Apache License 2.0 in the repository root applies only to original
Morpho-FM source code. The components and resources below are excluded from
that license and remain subject to their own terms.

## Distributed source

| Component | Official source and fixed revision | License or terms | Repository path | Distributed |
| --- | --- | --- | --- | --- |
| HIPT | [mahmoodlab/HIPT](https://github.com/mahmoodlab/HIPT), `780fafaed2e5b112bc1ed6e78852af1fe6714342` | Apache License 2.0 with Commons Clause | `src/st_pipeline/superres/hipt/`, except as noted below | Yes |
| DINO/timm Vision Transformer implementation | Preserved through the HIPT revision above; upstream HIPT blob `59a1106789675c20561369065c4c30954951395e` | Apache License 2.0; original Facebook/timm notice retained | `src/st_pipeline/superres/hipt/vision_transformer.py` | Yes |
| CellFM-derived PyTorch compatibility port | [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM), `76557975c935d6822ddcfd8bb9700d0b95ec7dd7` | No explicit source-code license was present at the audited revision or at upstream commit `bfed59c0e34103231165d69b97927ecc888d623c`; redistribution terms require clarification | `src/st_pipeline/model/cellfm_layers.py`, `src/st_pipeline/model/cellfm_model.py` | Yes; unresolved release item |

The HIPT Commons Clause text is retained in
`src/st_pipeline/superres/hipt/LICENSE`. The CellFM compatibility files are
not covered by the Morpho-FM Apache License 2.0. Their redistribution status
must be resolved with the CellFM copyright holders before a final release.

## Models and vocabularies

| Component | Official source and fixed revision | License or terms | Expected local path | Distributed |
| --- | --- | --- | --- | --- |
| CellFM pretrained weights | [ShangguanNingyuan/CellFM](https://huggingface.co/ShangguanNingyuan/CellFM), revision `69361b50d1814ade0e3749c9d8eb829590c2ce3c` | CC BY-NC-ND 4.0 model-card license | User-selected path under `checkpoints/` | No |
| CellFM gene vocabularies | [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM), `76557975c935d6822ddcfd8bb9700d0b95ec7dd7` | No explicit license present in the source repository | `third_party/CellFM/csv/gene_info.csv` or `expand_gene_info.csv` | No |
| HIPT pretrained weights | [mahmoodlab/HIPT](https://github.com/mahmoodlab/HIPT), `780fafaed2e5b112bc1ed6e78852af1fe6714342` | Apache License 2.0 with Commons Clause; non-commercial academic usage described by upstream | User-selected model directory under `checkpoints/` | No |

`scripts/convert_cellfm_ckpt.py` converts a checkpoint supplied by the user on
the user's local machine. Morpho-FM does not distribute the original or
converted CellFM checkpoint. A converted full checkpoint must not be uploaded
to GitHub, Zenodo, or a Morpho-FM release.

## External software

These repositories are not vendored. External source checkouts under
`third_party/` and method-source directories under `benchmark/` are local,
ignored working copies. Morpho-FM-authored benchmark notebooks remain tracked.

| Component | Official repository | Fixed revision | License at revision | Purpose | Distributed |
| --- | --- | --- | --- | --- | --- |
| CellFM | [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM) | `76557975c935d6822ddcfd8bb9700d0b95ec7dd7` | No root license file | Vocabulary and upstream architecture reference | No |
| HEST | [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST) | `2c777630b1e8a74db6b36ef05582de3bd52ccb61` | CC BY-NC-SA 4.0 | HEST data access and preprocessing | No |
| LazySlide | [rendeirolab/LazySlide](https://github.com/rendeirolab/LazySlide) | `6b9b0e760e83f8220a47b9c03b573fa1519d8aa0` | MIT | Optional whole-slide processing | No |
| sCellST | [loicchadoutaud/sCellST](https://github.com/loicchadoutaud/sCellST) | `96c3aa2af19e928920b351343272846d1f74fac0` | CC BY-NC 4.0 | Benchmark implementation | No |
| HisToGene | [maxpmx/HisToGene](https://github.com/maxpmx/HisToGene) | `44ff75b9b68949796f27f1b4121d36ea4cba8bf3` | MIT | Benchmark implementation | No |
| iStar | [daviddaiweizhang/istar](https://github.com/daviddaiweizhang/istar) | `3cb0e5352a86df337f41c5841d0808da0003457b` | GPL-3.0 | Benchmark implementation | No |
| mclSTExp | [ZhicengShi/mclSTExp](https://github.com/ZhicengShi/mclSTExp) | `027b6aa95ddd4ef0307bf80aa103c5381b01294c` | No root license file | Benchmark implementation | No |
| THItoGene | [yrjia1015/THItoGene](https://github.com/yrjia1015/THItoGene) | `13daf4d9c2531a74937dfedeeb4cb905cf0bf6cb` | No root license file | Additional benchmark workflow | No |
| HiST | [Yelab2020/HiST](https://github.com/Yelab2020/HiST) | `1a15c41e12d659fad04db3ad5a3ad26fbe6ef665` | MIT | Benchmark implementation | No |
| ST-Net | [bryanhe/ST-Net](https://github.com/bryanhe/ST-Net) | `43022c1cb7de1540d5a74ea2338a12c82491c5ad` | MIT, declared in `setup.py` classifiers; the repository carries no LICENSE file at this revision | Benchmark implementation | No |

## Study datasets

No study dataset is redistributed in this repository or its release archive.
Users must obtain each dataset from its original source and comply with the
terms shown there:

- HEST-1k: <https://huggingface.co/datasets/MahmoodLab/hest>
- 10x Genomics Xenium breast preview dataset:
  <https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast>
- HER2ST patient H sections: <https://doi.org/10.5281/zenodo.4751624>
- External ccRCC Visium sections: GEO accession
  [GSE210038](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210038)
