# Soil Grain Size from Photos

An image-based pipeline for estimating the cumulative grain-size distribution of soil
samples from ordinary smartphone photographs. This project was developed for the
**GRID / Kaggle soil grain-size competition**.

The model predicts the percentage of material finer than 11 DIN EN ISO 14688-1 sieve
diameters, from **0.002 mm to 200 mm**. Performance is measured with a log-weighted
Earth Mover's Distance (EMD); lower is better.

> **Status:** research and competition code. The project is intentionally documented as
> an experiment log rather than a packaged production library.

## Highlights

- Physical-scale normalization using camera calibration and image metadata.
- Frozen **DINOv2-base** image embeddings combined with mm-scale texture features.
- Optional **SAM ViT-L** grain segmentation for coarse-particle estimates.
- Ensemble regression with Ridge, PLS, and cosine k-nearest-neighbour models.
- Leave-one-sample-out, cross-camera, grouped, and cross-camera grouped validation.
- Automated monotonicity and bounds checks for predicted cumulative curves.
- Reproducible Slurm scripts for CPU and GPU jobs.
- Detailed experiment history, hypotheses, scores, and known limitations in
  [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## Results and limitations

The best documented image-feature ensemble reached approximately **41 EMD in
leave-one-sample-out validation**, compared with approximately **89 EMD for the mean
training-curve baseline**. These numbers are research diagnostics, not a guarantee of
test-set performance.

The dataset is small and has a substantial domain shift: training images were taken
with Motorola and Samsung phones, while the test set contains iPhone images with
different framing, resolution, lighting, and soil appearance. Predictions should
therefore be treated as estimates, not laboratory replacements.

## Repository layout

```text
.
├── README.md                 Project overview and reproduction guide
├── RESEARCH_LOG.md           Experiments, decisions, scores, and caveats
├── src/
│   ├── common.py             Paths, labels, camera scale, metric, and curve helpers
│   ├── features.py           DINOv2 and physical-scale texture extraction
│   ├── model.py              Feature sets, regressors, and validation protocols
│   ├── compare_models.py     Cross-validation model comparison
│   ├── predict.py            Image-feature ensemble submission
│   ├── predict2.py           SAM-calibrated two-stage submission
│   ├── sam_grains.py         SAM-based grain segmentation
│   ├── grains_eval.py        SAM versus label diagnostics
│   └── eda.py                Exploratory analysis and figures
├── scripts/
│   ├── env.sh                Shared environment and project setup
│   ├── extract_features.sh   GPU feature extraction
│   ├── run_sam.sh            Two-GPU SAM extraction
│   ├── run_gpu.sh            Generic one-GPU Slurm wrapper
│   └── run_cv.sh             CPU validation/prediction wrapper
├── data/                     Local competition data; not redistributed
├── cache/                    Regenerable feature and segmentation caches
├── figures/                  EDA and diagnostic plots
├── results/                  Experiment tables
└── submissions/              Generated Kaggle submission files
```

## Requirements

- Linux with a Slurm cluster for the supplied job scripts.
- Python 3.10+ (the original experiments used a conda environment named `myenv`).
- PyTorch with CUDA for DINOv2/SAM extraction.
- NumPy, pandas, SciPy, scikit-learn, Pillow, OpenCV, transformers, and timm.
- Hugging Face model weights cached locally for offline cluster jobs.

The exact original environment is not committed because it is cluster-specific. The
paths and resource limits in `scripts/` should be adapted to your own cluster before
running them.

## Data setup

The competition data is not included in the public source repository. Obtain it
according to the competition terms and arrange it as follows:

```text
data/
├── Training-All_Photos_updated/Training-All_Photos_updated/*.jpg
├── Test_All_Photos/Test_All_Photos/*.JPG
├── Training_labels_updated.csv
├── ppm_updated.csv
└── sample_submission.csv
```

The optional SandSnap extension expects the separately downloaded, publicly available
dataset at the path configured by the relevant script. See
[`RESEARCH_LOG.md`](RESEARCH_LOG.md) for attribution and licensing details.

## Reproducing the pipeline

From the project root, configure `scripts/env.sh` for your environment and download
the DINOv2 and SAM weights before enabling offline mode.

### 1. Extract DINOv2 and texture features

```bash
sbatch scripts/extract_features.sh feats_dinov2base.npz
sbatch scripts/extract_features.sh feats_gray_degrade.npz --gray --degrade
```

This writes feature caches under `cache/`. The `--gray` option reduces colour
dependence; `--degrade` applies image-quality normalization.

### 2. Run SAM grain segmentation (optional)

```bash
sbatch scripts/run_sam.sh
```

This produces sharded files such as `cache/grains_shard0.npz`. SAM requires two GPUs
with the default resource requests.

### 3. Compare models

```bash
FEATS=feats_dinov2base.npz \
  sbatch scripts/run_cv.sh src/compare_models.py rgb
```

Validation results are appended to `results/experiments.csv`.

### 4. Generate a submission

```bash
FEATS=feats_dinov2base.npz \
  sbatch scripts/run_cv.sh src/predict.py submission.csv

# Optional SAM-calibrated two-stage variant:
FEATS=feats_dinov2base.npz \
  sbatch scripts/run_cv.sh src/predict2.py submission_physics.csv linear 1.0
```

Generated files are written to `submissions/` unless an explicit path is supplied.
The output follows the competition's `sample_submission.csv` schema.

## Running locally

The Python modules assume they are launched from the repository root. For a local
environment, replace `sbatch scripts/run_cv.sh` with:

```bash
python src/compare_models.py rgb
python src/predict.py submission.csv
```

Feature extraction and SAM segmentation still require the model dependencies,
downloaded weights, and sufficient GPU memory.

## Metric

For adjacent sieve diameters `d_i` and `d_{i+1}`, the project computes the area between
the true and predicted cumulative curves on a base-10 logarithmic diameter axis:

```text
EMD = Σ 0.5 × (|error_i| + |error_i+1|)
          × (log10(d_i+1) - log10(d_i))
```

Predictions are clipped to `[0, 100]`, projected to be non-decreasing, and forced to
equal 100% at 200 mm.

## External models and data

- **DINOv2-base** and **SAM ViT-L** are Meta models distributed through Hugging Face
  under Apache-2.0-compatible terms.
- **SandSnap**: McFall et al., *SandSnap: Measuring and mapping beach grain size using
  crowd-sourced smartphone images*, Coastal Engineering 192 (2024), 104554.
  The associated Zenodo data is CC BY 4.0. See the research log for the exact
  attribution and how it is used.

Check the terms of the competition data and each dependency before redistribution or
commercial use. This repository does not redistribute competition images or labels.

## Citation

If this work is useful, please cite the competition dataset and the SandSnap paper
when using the optional external-data extension. The full methodological history is
available in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## License

No repository-wide open-source license has been selected yet. The competition data,
external datasets, pretrained weights, and third-party packages remain subject to
their own licenses and terms.
