# Soil grain size from photos — research log

Living document: every hypothesis, experiment, score, bug and decision. Update on every change.
Newest entries are appended at the bottom of each section; nothing is deleted (superseded items are
marked as such).

---

## 0. Rules & environment

| Item | Value |
|---|---|
| Compute | Slurm, **partition `mtech` only**, **max 2 GPUs per user** (QOS `gres/gpu=2`) |
| GPUs | cn02–cn05: 2× NVIDIA A30 24 GB each. cn07 = 6× RTX A6000 (user's scripts exclude it — don't use). dgx down. No Blackwell found under mtech. |
| Env | `source /csehome/m25csa007/miniconda3/etc/profile.d/conda.sh && conda activate myenv` (py3.14, torch 2.10 cu128, transformers 5.5, timm, sklearn, opencv) |
| Model weights | downloaded on login node to `~/.cache/huggingface` (home is shared); jobs run with `HF_HUB_OFFLINE=1` |
| Gotcha | login-node `/tmp` is **not** visible from compute nodes → write outputs in the project dir |
| Submissions | Kaggle, **5 per day**, **2 final selections**. Final = 0.3·public + 0.7·private. |

### Competition rules that constrain the method (read 2026-10-04)

| Rule | Consequence for us |
|---|---|
| Foundational §4.b: no "hand labeling or human prediction of the validation dataset or test data records" | Predictions must come from an automated pipeline. Visual inspection of test photos may motivate *which method to build* (e.g. H12), but no human-chosen values (e.g. a fines level picked by eye) may enter a submission. Automatic gates (e.g. SAM measurement outside training range) are fine. |
| Specific §2.6: External data / models allowed if publicly available & free to all (Reasonableness Standard) | External labelled sediment photos are allowed — record source, licence and access for each |
| Foundational §3.6.c: open-source code must be OSI-licensed without commercial-use limits | DINOv2 (Apache-2.0), SAM (Apache-2.0), sklearn/opencv OK; check every new dependency/dataset |
| Specific §2.4: Competition Data for non-commercial / academic use | — |
| Specific §2.8: winners deliver training + inference code, docs, environment | keep README / RESEARCH_LOG / scripts reproducible |
| Specific §2.2: 5 submissions/day, 2 final picks | submit only variants that test a stated hypothesis |

Typical job:
```bash
srun -p mtech -w cn04 --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 00:20:00 bash -lc '<activate> && python script.py'
```

---

## 1. Problem & metric

* Predict cumulative % finer at 11 sieve sizes (0.002 … 200 mm, DIN EN ISO 14688-1) from photos.
* Metric: log-weighted EMD (area between true and predicted cumulative curves on a log10 axis), range 0–500, lower is better.
* **Caveat:** the formula image in the competition text was not visible. `common.emd` implements the
  trapezoid form `Σ_i (|e_i|+|e_{i+1}|)/2 · (log10 x_{i+1} − log10 x_i)`. It matches the stated 0–500
  range; absolute values may differ slightly from Kaggle's (rankings between models should not).

---

## 2. Data facts (verified)

| Fact | Detail | Consequence |
|---|---|---|
| Train | 24 samples, 127 photos (Motorola Edge 69, Motorola Edge 60 fusion 3 (H374 only), Samsung A52 55) | tiny data → simple models, strong priors |
| Test | 10 samples, 35 photos, **iPhone 14 (14) / iPhone 16 (21) only** — cameras never seen in training | camera/domain shift |
| Resolution | training files are **downscaled** (Motorola 1600×720 vs native 4000×1800; Samsung 1599×1200 vs native 9248×6936). Test files are full-res. | ppm in `ppm_updated.csv` is for native width → effective ppm = ppm × long_side/native_long_side: train ≈ 4.6 px/mm (Moto60: 12.5), test 13.9 (iPhone14) / 19.5 (iPhone16) |
| Field of view | all ≈ 290–350 mm wide | similar physical framing |
| Framing | train: whole tray incl. dark tray rim + LED lamp strip; test: close-ups filling the frame (Münster has a dark rim on one side) | must mask tray/lamp |
| EXIF | some images carry orientation tags | always `ImageOps.exif_transpose` |
| IDs | submission ids differ from file names: `Münster_BS6_9,0-10m` → `HPC_Muenster_BS6_9_0-10m`; one file prefix `iPhone_16_` | `common.test_name_to_id` |
| Labels | 200 mm always 100; only H038 (4.3 %) and H374 (7.4 %) have anything > 63 mm | model has no training signal for cobbles |
| Appearance (see `viz/domain_gap.jpg`) | training soils are **dried & aggregated** (clods / pellets even for 76 %-fines H405 and 3 %-fines H616); **warm-beige** LED light. Test sands look **loose, unaggregated, neutral grey**; test Münster/Lidl are clean cobble/gravel surfaces | appearance→grain-size mapping learned in training may not transfer |

Baselines on training labels (LOSO): equal-share curve **100.3**, mean training curve **89.1**, median curve 96.6.

---

## 3. Pipeline & file inventory

Folder layout and how to run: see [README.md](README.md) (reorganised 2026-10-04; reorganised
pipeline verified to reproduce `submission_physics.csv` byte-for-byte).

| File | Role |
|---|---|
| `src/common.py` | project paths, metric, monotone projection, camera/ppm, ID mapping, label loading, `label_groups()` (near-duplicate groups) |
| `src/features.py` | resample to 4.6 px/mm → 73 mm tiles (tray/lamp masked) → DINOv2-base CLS+mean-patch & 31 scale-space texture features in mm units. Flags `--gray` (colour-invariant), `--degrade` (blur+JPEG70 quality equalisation) |
| `src/model.py` | feature sets × regressors; CV protocols `loso`, `xcam`, `group`, `xgroup` via `folds()`. `FEATS` env var selects feature file in `cache/` |
| `src/compare_models.py` | CV grid under all 4 protocols + test diagnostics (sandy fines %, Münster F(2 mm)) → appends `results/experiments.csv` |
| `src/predict.py` | v1 ensemble → `submissions/submission.csv` |
| `src/sam_grains.py` | SAM ViT-L automatic masks (full image + 2×2 windows), compact-grain filter, dedupe → per-image area % of grains with sieve size > d (d = 2 … 125 mm) |
| `src/grains_eval.py` | SAM area % vs true coarse % on training; `load_grains()` (matches by file name) |
| `src/predict2.py` | two-stage model: calibrated SAM coarse part + ensemble fines shape; grid over calibration form & blend weight |
| `src/eda.py` | deeper EDA → `figures/eda/` |
| `cache/feats_dinov2base.npz` | RGB features |
| `cache/feats_gray_degrade.npz`, `cache/feats_degrade.npz` | colour-invariant + quality-equalised / quality-equalised features |
| `cache/grains_shard{0..3}.npz` | SAM grain stats |
| `submissions/LEADERBOARD.csv` | machine-readable submission log |
| `results/experiments.csv` | every CV grid row (features, model, 4 protocols, test diagnostics) |

---

## 4. Hypothesis register

Status: ✅ supported · ❌ rejected · 🔬 testing · ❓ open (untested) · ⚠️ adopted without ablation

| ID | Hypothesis | Rationale | How tested | Status & evidence |
|---|---|---|---|---|
| H1 | Resampling all photos to a common px/mm is needed | ppm differs 4.6 vs 14–19 between train and test | by design | ⚠️ adopted, never ablated (ablation would need a scale-naive run) |
| H2 | Tiles must exclude tray rim and lamp | they dominate texture/colour stats | visual check `viz/tiles.jpg` | ✅ after bug fix (see §7 B1) |
| H3 | Frozen DINOv2 + mm-scale texture features carry grain-size signal | generic texture embeddings; physical scale-space | LOSO CV | ✅ ensemble 41.0 vs 89.1 mean-curve baseline |
| H4 | Camera shift between training phones is moderate | xcam CV close to LOSO | xcam CV | ✅ ensemble xcam 41.1 (≈ LOSO). ⚠️ does **not** cover iPhone/lighting shift (both training phones share lamp/tray) |
| H5 | Münster and Lidl are out-of-distribution (coarser than any training sample) | photos show cobbles/gravel | feature NN distance; SAM range | ✅ Münster NN-dist 8.5 vs train max 3.6; SAM area >2 mm 91 % / 80 % vs train max 69 % |
| H6 | SAM grain-area fraction ≈ mass fraction coarser than d (Delesse) | area fraction of a section ≈ volume fraction | per-sample correlation on train | ✅ r = 0.93 (2 mm), 0.90 (6.3 mm), 0.62 (20 mm); slopes 1.1–1.25; fine soils show 8–19 % false-positive "grains" (clods) |
| H7 | Two-stage (logit-calibrated SAM coarse, w=0.5) is best | best mean of LOSO/xcam CV (39.66) | CV + LB | ❌ on LB: **58.71** (worse than physics) — CV cannot see OOD extrapolation |
| H8 | For OOD coarse samples, the physical SAM estimate beats the learned model | learned models can't extrapolate beyond training support | LB comparison: physics (linear, w=1) vs two-stage | ✅ LB **49.45** vs 58.71. The two files differ ≤ 9 EMD on sandy samples but 54/25/16 on Münster/Lidl/Audorfring → the gain must come from coarse samples |
| H9 | Area-weighting under-estimates mass of large stones (mass ∝ d³, visible area ∝ d²) → physics still too fine for Münster | surface-of-pile geometry; training slopes > 1 | volume-weighted grain stats in CV (needs per-grain SAM output) | ❌ rejected on training (E23): weighting grains by area·sieve^k for the split of the gravel fraction, MAE of C(>d)/C(>2) at 6.3/20/63 mm over 13 gravelly samples: k=0 (area) **6.3**, k=0.5 8.1, k=1 (≈volume) 13.4, k=2 22.7; leave-one-group-out picks k=0. Surface area already tracks mass; volume weighting over-counts large stones |
| H10 | Colour shift (warm-lit train vs grey test) hurts DINOv2/kNN | features colour-sensitive; test sands matched to beige training soils | EDA colour stats; `--gray --degrade` features: test-to-train NN distance | ◐ partly. Test is greyer than all training (b* 3.1 vs Motorola 5.2, Samsung 11.2) but close to Motorola. Gray+degrade cuts the DINOv2 test-sand NN distance only 4.4→3.6× (still 18/21 photos → H493) ⇒ colour is **not** the main driver of the DINOv2 shift. For texture features it helps a lot: 3.0→1.85× |
| H11 | Image-quality shift (compressed train vs pristine test) hurts fine-scale texture features | DoG 0.25–0.5 mm bands depend on sharpness/JPEG | EDA §2 per-camera stats at 4.6 px/mm | ❌ mostly rejected: sharpness (LapVar) Motorola 408 > iPhone 216–231 > Samsung 120; noise and blockiness also inside the training range |
| H12 | Aggregation-state shift (dried clods in train vs loose sand in test) → fines over-predicted for sandy test samples (models say ~35–50 % < 63 µm) | `figures/debug/domain_gap.jpg`: all training soils look aggregated; test sands resemble only H493 (52 % fines) | EDA feature space + native-resolution crops | ✅ strongly supported visually: `figures/debug/native_sand.jpg` (20 mm crops at 19.5 px/mm) shows **clean, loose sand** — individual rounded grains ≈ 0.15–0.8 mm, little visible matrix. Fines are very likely far below the predicted 35–50 % |
| H13 | LOSO CV leaks through near-duplicate samples → CV optimistic | consecutive IDs, near-identical curves | EDA §1/§5: hold out whole label groups (avg-link label EMD < 15) | ✅ groups {H366,H367,H371,H372,H374}, {H368,H615}, {H181,H183}, {H038,H637}, {H126,H549,H668}, {H493,H666}. Ensemble LOSO 41.0 → **group 44.9**; gravelly samples' errors 5–12 → 10–23. **Use `group`/`xgroup` CV for model selection from now on** |
| H14 | DINOv2 embeddings carry the domain shift; mm-scale texture features are robust | EDA PCA: sandy iPhone photos form their own cluster in emb space (PC2 45–65) next to **only H493's** 5 photos; in hand-feature space they sit inside the training cloud | PCA + NN analysis (EDA §3) | ✅ supported. Emb-based models therefore predict "H493-like" (≈ 45 % fines) for every sand. Model disagreement on test sands is huge (0–51 % fines across 24 models, `results/experiments.csv`) while their CV differs by a few points |
| H18 | A second, differently trained embedding (SigLIP) improves compositional generalisation of the sand model; SigLIP text axes give named factor coordinates | agreement across encoders = real signal; disagreement = OOD uncertainty | `src/multi_embed.py`, `src/embed_eval.py` (E24): leave-site-out + dry→wet systematic split on SandSnap; concept-axis Spearman | ◐ supported on the state split: dry→wet EMD 6.38 (DINOv2) → **5.02 late fusion DINOv2+SigLIP**, MAE F(0.2) 8.1→3.0 (only 9 wet samples); site CV ≈ unchanged (6.9–8.1). CCA subspace no gain. Embeddings disagree on test-sand F(0.2) by 4–22 pts → averaging. Concept axes: coarse_vs_fine ρ=+0.80 with gravel %, but entangled (wet_vs_dry ρ=−0.73 with gravel). loose_vs_clumped separates SandSnap (+0.09) from competition (0.00); test sands +0.015…+0.049, **Audorfring lowest (−0.039)** — agrees with the gate and the H17 LB result → candidate evidence-based gate |
| H17 | The sand matrix of the gravelly test samples (Audorfring, Lidl) is also clean, so the sand model should handle their < 2 mm part too | same sites / region as the clean test sands; old models predict 17 % fines for Audorfring | LB: `submission_sand_w100_g101.csv` vs 30.389 (differs only on Audorfring, Lidl) | ❌ LB **33.037** vs 30.389 (worse). Only Audorfring (26.5) and Lidl (4.1) changed ⇒ Audorfring likely public and its matrix contains real fines; the old ensemble's ~17 % was closer. **Keep the gate (SAM area > 2 mm < 20 %)**: the sand model applies to clean-sand samples only |
| H16 | External loose-sand data (SandSnap) can teach what clean sand looks like → better sand-range prediction for test sands | SandSnap photos look like the test sands (`figures/debug/sandsnap_examples.jpg`); rules allow public data | sand model at 12 px/mm on SandSnap only; leave-site-out CV; domain check (test NN distance vs SandSnap p95); LB | ◐ model works on beach sands (leave-site-out EMD 6.0 vs 10.4 baseline; MAE F(0.2) 8.1 pts, F(0.63) 2.9). Domain match partial: 3/7 test sands inside SandSnap p95 NN distance (Airbus BS6-3, Kleink 18-3, 2-2), 4 moderately outside (33–36 vs 27.9). **Limitation:** all SandSnap sands are clean (F(0.063) 0–0.9 %), so the model always predicts ≈ 0.5 % fines — a data-borne assumption, not a measurement. **LB: 30.389 vs 49.448** (same file except the < 2 mm part of the 7 gated sands) ⇒ ✅ the public sandy samples are far cleaner than the competition-trained models assumed; external loose-sand data fixed it |
| H15 | Sand grading can be measured physically at native test resolution (19.5 px/mm → 0.05 mm/px) | grains ≥ 0.15 mm are 3+ px; analogous to the SAM coarse measurement that helped on LB (H8) | E15 opening granulometry; E16 SAM on upscaled 10 mm native crops; sanity checks: Kleink 2-2 (visibly coarser) must come out coarser than 18-3, cobble surfaces must differ from sand | ❌ **not feasible.** Pixel resolution ≠ optical resolution: native iPhone crops are soft, edges blur over ~0.1–0.2 mm, effective resolution ≈ 0.2–0.3 mm. Both methods fail the sanity checks. Fine sand / silt / clay — the quantity we need — are invisible in these photos |

---

## 5. Experiment log

| # | Date | Experiment | Result |
|---|---|---|---|
| E1 | 10-03 | Baselines (no images) | equal-share 100.3 · mean curve 89.1 · median 96.6 (LOSO) |
| E2 | 10-03 | DINOv2-base + texture, first run | **invalid**: tile filter rejected almost all tiles → 1 central crop/image (bug B1) |
| E3 | 10-03 | Same after tile fix (40 tiles/img), grid of regressors | best: hand+emb PCA5-ridge LOSO 39.5 / xcam 42.4; hand kNN5 41.8/43.1; emb PCA5-ridge 39.7/43.2 |
| E4 | 10-03 | Ensemble [hand+emb pca5_ridge, emb pls2, hand knn5, hand+emb knn5] | LOSO 41.0 / xcam 41.1 → `submission.csv` |
| E5 | 10-03 | SAM ViT-L grain segmentation (full + 2×2 windows) on all 162 photos | per-sample correlations see H6; test Münster area >2/6.3/20/63 mm = 91/85/62/11 % |
| E6 | 10-03 | Two-stage linear (unconstrained) | w=1: LOSO 39.7 / xcam 45.7; positive 20 mm intercept capped sandy samples at 97.1 % (bug-like artefact) |
| E7 | 10-03 | Two-stage, linear (intercept ≤ 0) vs logit calibration, w ∈ {0.25…1} | best avg: logit w=0.5 LOSO 38.9 / xcam 40.4 (avg 39.66); linear w=1: 39.1/46.4 |
| E8 | 10-04 | Per-sample diff between submissions | two_stage vs physics: Münster 54.5, Lidl 25.1, Audorfring 15.9, sandy ≤ 8.6 |
| E9 | 10-04 | Feature variants `--gray --degrade` and `--degrade` extracted (gray-only run failed: transient cv2 import error, then rerun accidentally on CPU — cancelled, see B10) | test-sand NN distance (÷ train-train median): emb 4.42→3.57, hand 3.04→1.85 (gray+degrade). CV grid on these features: todo |
| E10 | 10-04 | Deeper EDA (`src/eda.py`, figures in `figures/eda/`) | label table (gravel/sand/silt/clay, D50, Cu); 6 near-duplicate groups; per-camera quality/colour; PCA shift; Spearman: texture band ratios vs %<0.63/%<2 mm ρ≈0.9, vs %<0.063 ≈0.7, vs %<0.002 ≈0.5; texture features reproducible across cameras r=0.97–0.99; group-CV error by interval: 0.02–0.63 mm intervals = 22.6 of 44.9 EMD (silt/sand split dominates); worst samples H038 128, H405 95, H616 77, F827 76, H037 71 |
| E11 | 10-04 | CV grid × 4 protocols with test diagnostics (`compare_models.py rgb`) | best group CV: hand+emb pca5_ridge 42.9, emb pca5_ridge 43.2, hand pls2 44.1; best xgroup: hand knn5 47.3. Test-sand predicted fines range 0–51 % across models (extrapolation) |
| E12 | 10-04 | Native-resolution test crops (`figures/debug/native_sand.jpg`) | test sands are clean loose sands (see H12) |
| E13 | 10-04 | Project reorganisation (src/, data/, cache/, scripts/, …) | regenerated `submission_physics.csv` is byte-identical to the submitted file |
| E15 | 10-04 | `src/sand_granulometry.py`: grayscale & binary opening granulometry on 12 × 25 mm native crops per photo, log-normal tail fit | ❌ near-identical output for every sample (gray D50 ≈ 0.8 mm, binary ≈ 0.33 mm) incl. Münster cobble surfaces and H374; Kleink 2-2 comes out finer than 18-3 (visually the opposite); non-monotone at 0.15/0.2 mm on iPhone14 (2–3 px disks). Measures surface texture/shading, not grains. Output `cache/sand_granulometry.npz` kept for the record, **not used** |
| E16 | 10-04 | `src/sand_sam.py`: SAM ViT-L on 8 × 10 mm native crops upscaled to 1024 px, 64×64 point grid (3 test photos) | ❌ only 13–30 % of crop area segmented; overlays (`figures/debug/sand_sam_compare.jpg`) show blurred native detail, only distinct grains ≳ 0.4 mm separated; 2-2 again finer than 18-3. One crop included the blue tray rim (crop mask from 4.6 px/mm soil mask misses blue rims) |
| E17 | 10-04 | SandSnap download (Zenodo 7063226, CC-BY-4.0) to `/scratch/m25csa007/external/sandsnap`; `src/sandsnap_labels.py` → 1,323 matched images, **75 sieved samples**, 11-point curves (log-linear inside d10–d90, log-normal tails) | F(0.063) 0–0.9 %, F(0.2) 0.8–96.6 %, F(0.63) 66–100 % |
| E18 | 10-04 | `src/sand_features.py` (12 px/mm, 224 px tiles, gray DINOv2 + texture 0.05–3.2 mm) on 978 SandSnap + 35 test photos (rest too small / too coarse) ; `src/sand_model.py` (PCA20 + ridge on logit F) | see H16; predictions in `cache/sand_preds.csv` |
| E19 | 10-04 | `src/predict3.py`: physics submission + sand model below 2 mm for samples with SAM area > 2 mm < 20 % (automatic gate → the 7 sandy samples) | `submission_sand_w100.csv` (mean EMD change vs physics 45.7), `submission_sand_w50.csv` (22.9) |
| E20 | 10-04 | Sand model applied to the < 2 mm fraction of **all** samples (`predict3.py 1.0 101`) | differs from 30.389 file only on Audorfring (EMD 26.5: F(0.063) 17.3→0.3, F(0.2) 29.7→12.0) and Lidl (4.1) → `submission_sand_w100_g101.csv`. Tests H17: the gravelly test samples' sand matrix is also clean |
| E21 | 10-04 | SandSnap v2: the 1,137 'unmatched' rows were in `tiles_20210823_set1.zip` (66 **new** samples, d50 0.17–0.70 mm) → 2,460 images / **141 samples**. Sand-model outputs versioned (`cache/sand_*_{v1,v2}`; v1 = what produced the 30.389 file). Leave-site-out CV (site for tiles = filename letter prefix): 115 sands, EMD **7.31** vs baseline 11.89; MAE F(0.2) 8.2, F(0.63) 5.4. Domain: 4/7 test sands inside SandSnap p95 (was 3/7) | v1 vs v2 disagree strongly on test F(0.2) (Airbus BS12 29→61, BS10 18→33, Kleink 18-3 14→25) → F(0.2) is uncertain for the test sands |
| E22 | 10-04 | `predict3.py` with `SAND_VER=v2` and `SAND_VER=v1+v2` (average of the two sand-model versions; EMD convex ⇒ average never worse than mean individual error) | `submission_sand_w100_v2.csv` (EMD vs 30.389 file mean 4.5, max 15.1), `submission_sand_w100_v1v2.csv` (mean 2.2, max 7.5) |
| E23 | 10-04 | SAM rerun saving per-grain output (`cache/grains_v2_shard*.npz`, job 42363, mtech 2 GPUs); `src/coarse_weighting.py` | H9 rejected (see register); no submission |
| E24 | 10-04 | SigLIP-so400m (Apache-2.0; local copy `/scratch/m25csa007/models/`, HF download hit httpx/brotli bug; tokenizer from tokenizer.json, no new deps) + DINOv2 layer 6; job 42401 (mtech, 1 GPU) → `cache/multi_embed.npz`; eval `logs/embed_eval.txt`; `cache/sand_preds_{late,late3,multi}` | see H18. Candidate `submission_sand_w100_v1late3.csv` (sand curve = avg of v1 and late-fused v2 DINOv2-last+mid+SigLIP), EMD vs 29.532 file mean 1.4, max 4.1 |
| E14 | 10-04 | CV grid on `feats_gray_degrade.npz`| E14 | 10-04 | CV grid on `feats_gray_degrade.npz` (`compare_models.py gray_degrade`) | CV ≈ unchanged (best group: hand pls2 43.7, emb pca5_ridge 44.1; best xgroup: emb knn5 45.1, hand knn5 45.3 vs 47.3 RGB). But test-sand fines predictions stay 41–50 % for nearly all models ⇒ **no learned model fixes H12**; training has no loose clean-sand photos. Fix must be physical (H15) |

---

## 9. External data candidates (rules §2.6: public + free → allowed)

| Dataset | Content | Labels | Licence | Access | Fit |
|---|---|---|---|---|---|
| **Photogranulometry** (Plante St-Cyr et al. 2025, arXiv 2506.17469, DOI 10.20383/103.01316) | 12,714 photos of 321 soils (Montreal; 78 % silty sands + sands, gravels, artificial gravel mixes), **moist and oven-dried**, Canon R5 45 MP, 39.4 µm/px (≈ 25 px/mm), white trays, ArUco calibration image per sample | % passing 14 sieves 0.08–80 mm (Excel); no silt/clay split | CC-BY-4.0 | FRDR Globus endpoint `f163c1b3-9c88-42f6-a7bb-5839ed6c4063`, path `/1/published/publication_1311/submitted_data`; 425 GB total; ZIP download temporarily offline → needs user's Globus login | **best**: moist loose silty sands = the domain our training lacks (H12); gives fines (< 0.08 mm) |
| SandSnap 2020–21 (McFall et al. 2024; Zenodo 7063226) — **in use (user chose SandSnap only; Photogranulometry declined for now)** | sieved beach sands, smartphone photos with coin scale (mm_px per image) | D10…D90 percentiles | CC-BY-4.0 | direct Zenodo download, 3.1 GB | clean-sand grading 0.1–2 mm; no fines |
| Soranzo et al. 2025 (Soils and Foundations 65, 101575) | 136 images of 26 samples; 6936×9248 originals published at 1599×1200 | — | — | — | **appears to be the source of the competition training set** (identical Samsung A52 sizes; authors at BOKU = sponsor). No new information |
| Close-range soil images + PSD (Data in Brief 2025, S2352340925003622) | oven-dried soils clayey silt–gravel, 2 smartphones, dark chamber | PSD | ? | page 403 to fetch | possibly the same as Soranzo et al.; to check |
| Black-cotton / red soil (IEEE DataPort) | smartphone + microscope images of sieve fractions | PSD | ? (DataPort may need subscription → Reasonableness) | — | low (single soil type) |

---

## 6. Submission log

| File | Description | CV LOSO | CV xcam | Public LB |
|---|---|---|---|---|
| `submission.csv` | E4 ensemble only | 41.0 | 41.1 | not submitted yet |
| `submission_two_stage.csv` | E7 logit calibration, w=0.5 | 38.9 | 40.4 | **58.714** |
| `submission_physics.csv` | E7 linear (a≤0) calibration, w=1 (full SAM coarse part) | 39.1 | 46.4 | **49.448** |
| `submission_sand_w100.csv` | E19 physics + SandSnap sand model (W=1) for SAM-gated sands | – | – | **30.389** |
| `submission_sand_w50.csv` | E19 physics + 50/50 blend with sand model for SAM-gated sands | – | – | not submitted (w100 already +19 pts; low information) |
| `submission_sand_w100_g101.csv` | E20 sand model for the < 2 mm part of all samples | – | – | 33.037 |
| `submission_sand_w100_v1v2.csv` | E22 gate 20 %, sand model = average of v1 (75 samples) and v2 (141 samples) | – | – | **29.532** (best) |
| `submission_sand_w100_v2.csv` | E22 gate 20 %, sand model v2 only | – | – | not submitted |
| `submission_sand_w100_v1late3.csv` | E24 gate 20 %, sand model = avg(v1, late3 fusion) | – | – | pending |

Lesson: on this test set, CV (which cannot see the OOD coarse samples or the iPhone/lighting shift)
ranks models opposite to the LB for the coarse samples. Use CV for in-distribution choices, physics +
LB for OOD behaviour.

---

Scored-file checksums (md5): physics 19b05f30…, two_stage 38cfb1f0…, sand_w100 41854a47…; sand_w100_v1v2 see `md5sum submissions/scored/*`.

## 7. Bugs & pitfalls encountered

| ID | Problem | Fix |
|---|---|---|
| B1 | Tile filter (any pixel V<45 or >245, dilated) rejected nearly all tiles because of inter-grain shadows | only large dark/bright regions count: morphological opening with ~4 mm kernel; ≤2 % bad pixels per tile |
| B2 | `import features` executed `main()` (no guard) → accidental CPU run | `if __name__ == "__main__"` |
| B3 | SAM dedupe kept a whole-soil mask first → every grain discarded | filter masks (size/compactness/in-soil) **before** dedupe |
| B4 | transformers 5.5 SAM pipeline: `crops_n_layers=1` crashes on non-square images | own 2×2 window pass, drop masks touching interior window edges |
| B5 | SAM pipeline fp16 → NMS dtype error | fp32 |
| B6 | 3 SAM processes per 24 GB A30 → OOM | 2 per GPU |
| B7 | `points_per_batch` not dividing 1024 → `StopIteration` in pipeline | use 64 (or 256) |
| B8 | 6-GPU request pending (`QOSMaxGRESPerUser`) | ≤ 2 GPUs |
| B9 | `model.py` read `sys.argv[1]` on import | only when `__main__`; `FEATS` env var |
| B10 | `srun --jobid=X --overlap` step without `--gres` had no GPU → `features.py` silently fell back to CPU (5/162 images in 4 min) | request the GPU for the step, or submit a separate job; consider failing loudly when CUDA is missing |
| B12 | `predict3.py` always wrote `submission_sand_w100.csv`; a gate-101 preview overwrote the LB-scored 30.389 file | regenerated deterministically with gate 20 (inputs unchanged); LB-scored files now copied read-only to `submissions/scored/` (md5 in log below); `predict3.py` names non-default gates `_g{GATE}` and refuses to overwrite scored files |
| B11 | One-off use of the `test` partition (1 GPU, ~1 min `nvidia-smi` on cn07) before the user's mtech-only rule | rule recorded; mtech only since |

---

## 8. Next steps (ordered)

1. ~~EDA~~ done (E10). ~~Switch to group CV~~ done (H13).
2. CV grid on `feats_gray_degrade.npz` (group/xgroup) — texture features become much more domain-robust there (H10).
3. ~~H15: native-resolution sand measurement~~ — not feasible (E15, E16). Options for the 7 sandy samples (pending user decision):
   a. **External labelled data** of loose sand photos with grain-size labels (e.g. SediNet, Buscombe 2020) to teach the model what clean sand looks like — check competition rules on external data first.
   b. ~~Hand-chosen fines level for the sandy group~~ — **prohibited** by foundational rule §4.b (human prediction of test records). Choosing between *automated* model variants on the LB remains allowed.
   c. Accept the limitation and focus on the coarse samples (items 4–5).
4. Gated model: physics (w=1) where SAM measurement is outside the training range (Münster, Lidl: SAM area > 2 mm 91 / 80 % vs train max 69 %), CV-best blend elsewhere.
5. ~~H9 volume weighting~~ — rejected on training (E23).
