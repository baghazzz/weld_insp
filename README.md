# weld_insp — do weld-defect classifiers really generalise?

Weld X-ray classifiers report ~99% accuracy on the RIAWELC benchmark but drop to ~64% macro-F1 on new radiographs
(GDXray, WeldVGG 2025). This project tests **why**, and whether Vision Transformers, DINOv2 features or anomaly
detection (PatchCore, a learned "golden path") hold up better on an **independent weld X-ray dataset (SWRD)**,
ending in a cost-based operating point that trades missed defects against false rejects.

> **Status:** work in progress. Labs 1, 2 and 4 are complete with results below. Labs 5–6 are running; Labs 3 and
> 7–9 are being moved from GDXray to SWRD as the external test set (see *Why SWRD instead of GDXray*).

## Key finding so far: the benchmark leaks

The official RIAWELC test folder (2,443 tiles) consists **entirely of byte-identical copies of training tiles, with the
same labels**. 28% of validation tiles have a same-defect twin in training (perceptual hash within 6 bits). All tiles
come from only 29 source radiographs (BAM round-robin plates `RRT-xxR`), and the
`Img1–Img4` sections of a plate are repeat scans of the same film. Published ~99% scores on this split measure
memorisation. Every model here is trained and tested on a **leakage-free split grouped by plate** (70/15/15).

Measured directly (Lab 4, same features and settings, only the split differs): an RBF-SVM and a Random Forest both
score **100.0%** test accuracy on the official split, and **61.3% / 51.8%** on unseen plates (SVM macro-F1 0.612).

<!-- BEGIN leakage -->
| comparison | % within 0 bits | % within 6 bits |
|---|---|---|
| test vs train | 100.00 | 100.00 |
| test vs train (clean) | 0.03 | 1.85 |
| val vs train | 5.29 | 27.65 |
| val vs train (clean) | 0.00 | 1.70 |
<!-- END leakage -->

## Why SWRD instead of GDXray

The original plan tested on GDXray Welds. Two problems:

1. **GDXray is not independent of RIAWELC.** RIAWELC's file names (`RRT-xxR`) are BAM Round Robin Test radiographs,
   and GDXray's weld series W0003 *is* that BAM round-robin set (W0001 is a selection from it). A "new data" test on
   GDXray would partly re-test the radiographs the models were trained on.
2. **The GDXray Welds download is no longer available** from the official link, and its annotations mark only
   *where* a defect is, not its type.

**SWRD** (Zhao et al., 2025; Beijing Institute of Technology) has 3,675 seam-weld radiographs from a different lab,
labelled with six defect types: porosity, inclusion, crack, undercut, lack of fusion and lack of penetration. Crack,
porosity and lack of penetration match RIAWELC classes directly, so SWRD allows both a defect-vs-no-defect test and a
**per-type** transfer test.

## Headline results

RIAWELC columns: 4-class, clean split, test plates never seen in training. External-test columns will report SWRD
(defect vs no defect, and per-type recall for crack, porosity and lack of penetration) once Labs 3 and 7 are rerun.

<!-- BEGIN headline -->
_Filled by notebook 09 from results/headline_table.csv after Labs 4–7 have run._
<!-- END headline -->

Anomaly detection (ROC-AUC, defect vs no defect):

<!-- BEGIN anomaly -->
_Filled by notebook 09 from results/anomaly_auc.csv._
<!-- END anomaly -->

Operating points (missed defect = 20× a false reject; 5% defect rate assumed; threshold chosen on validation):

<!-- BEGIN operating -->
_Filled by notebook 09 from results/operating_points.csv._
<!-- END operating -->

## Data

| Dataset | What | Licence | In this repo? |
|---|---|---|---|
| [RIAWELC](https://github.com/stefyste/RIAWELC) | 24,407 X-ray weld tiles (227×227, 8-bit), classes CR / PO / LP / ND | released freely, cite Totino et al. 2023 | no (downloaded by notebook 01), 4 demo tiles in `app/examples/` |
| [SWRD](https://github.com/bit628/RapidX-Annotator) | 3,675 seam-weld radiographs, 6 defect types with polygon labels; external test set | per the dataset providers' terms; cite Zhao et al. 2025 | no |
| [GDXray Welds](https://github.com/computervision-xray-testing/GDXray) | BAM weld radiographs with binary defect masks; original external test set, replaced by SWRD | research and education only, **no redistribution** | no |

Folder → class: `Difetto1` = CR (crack), `Difetto2` = PO (porosity), `Difetto4` = LP (lack of penetration),
`NoDifetto` = ND (no defect); confirmed from the per-class counts in the RIAWELC paper.

## Method

| Lab | Notebook | What it does | Runtime |
|---|---|---|---|
| 1 | `01_data_exploration` | counts, samples, intensity histograms, source IDs from file names | CPU, 1 min |
| 2 | `02_leakage_check` | pHash near-duplicate search; plate-grouped clean split | CPU, 1 min |
| 3 | `03_gdxray_prep` | external test set preparation; **being rewritten for SWRD** (crops labelled CR / PO / LP / ND) | CPU |
| 4 | `04_classical_baseline` | HOG + LBP + GLCM + intensity features; SVM, Random Forest, Naive Bayes | CPU, ~30 min |
| 5 | `05_cnn_baseline` | ResNet50 fine-tuning, 3 seeds, original vs clean split | T4 GPU, ~2 h |
| 6 | `06_vit_dinov2` | ViT-S/16 fine-tuning; frozen DINOv2-S + logistic regression | T4 GPU, ~1 h |
| 7 | `07_gdxray_generalisation` | zero-shot external test of every model (moving to SWRD, adding per-type recall); augmentation retrain | T4 GPU, ~1 h |
| 8 | `08_anomaly_misjudgment` | PatchCore on good welds only; threshold sweep; cost-based operating points | T4 GPU, ~30 min |
| 9 | `09_gradcam_package` | Grad-CAM + pointing game on external-set defect outlines; README/summary from results; Gradio demo | T4 GPU, ~15 min |

Rules followed throughout: model settings and thresholds are chosen on validation data only; the test split is
touched once; the external test set is not used until the RIAWELC models are final.

## How to rerun

1. Put this folder on Google Drive as `MyDrive/weld_insp` (upload it, or sync it with Google Drive for desktop).
2. Open `notebooks/01_data_exploration.ipynb` in Colab and run all. It mounts Drive, finds the folder, downloads
   RIAWELC (and saves `data/riawelc.zip` for next time).
3. Run notebooks 02 → 09 in order. Labs 5–9 need *Runtime → Change runtime type → T4 GPU*. Training saves a
   checkpoint every epoch to `checkpoints/`; if Colab disconnects, re-run and it resumes.
4. Locally (CPU labs 1–4): `pip install -r requirements.txt`, then run the notebooks from `notebooks/` in Jupyter.

Shared code lives in `src/weld_insp/` (dataset class, transforms, leakage tools, features, metrics, training loop,
PatchCore, cost analysis). Notebooks are generated from `tools/notebook_sources/*.py` by
`python tools/build_notebooks.py`.

Demo: `python app/app.py --share` (needs `checkpoints/resnet50_clean_seed42/best.pth`).

## Limits

- SWRD has three defect types RIAWELC never saw (inclusion, undercut, lack of fusion); these can only be scored as
  "defect vs no defect", not by type.
- SWRD comes from one lab; a third dataset would be needed to claim generalisation across many sites.
- 29 source plates is a small number of independent samples; results on 6 test plates carry real uncertainty.
- The cost model (20× miss cost, 5% defect rate) is an assumption, not plant data.

## References

- Totino, Spagnolo, Perri (2023). RIAWELC: a novel dataset of radiographic images for automatic weld defects classification. IJECER.
- Palma-Ramírez et al. (2024). Deep convolutional neural network for weld defect classification in radiographic images. Heliyon.
- WeldVGG (2025). Sensors 25(19):6183.
- Zhao et al. (2025). SWRD: a dataset of radiographic image of seam weld for defect detection. J. Nondestructive Evaluation 44(2).
- Mery et al. (2015). GDXray: the database of X-ray images for nondestructive testing. J. Nondestructive Evaluation 34.
- Roth et al. (2022). Towards total recall in industrial anomaly detection (PatchCore). CVPR.
- Oquab et al. (2023). DINOv2: learning robust visual features without supervision.
