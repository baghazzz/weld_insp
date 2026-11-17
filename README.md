# weld_insp — do weld-defect classifiers really generalise?

Weld X-ray classifiers report ~99% accuracy on the RIAWELC benchmark but drop to ~64% macro-F1 on new radiographs
(GDXray, WeldVGG 2025). This project tests **why**, and whether Vision Transformers, DINOv2 features or anomaly
detection (PatchCore, a learned "golden path") hold up better, ending in a cost-based operating point that trades
missed defects against false rejects.

## Key finding so far: the benchmark leaks

The official RIAWELC test folder (2,443 tiles) consists **entirely of byte-identical copies of training tiles, with the
same labels**. 28% of validation tiles have a same-defect twin in training (perceptual hash within 6 bits). All tiles
come from only 29 source radiographs (BAM round-robin plates `RRT-xxR`, the same source GDXray uses), and the
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

## Headline results

RIAWELC columns: 4-class, clean split, test plates never seen in training. "Binary" and GDXray columns: defect vs
no defect (GDXray masks carry no defect type). "Unseen radiographs" excludes GDXray images that are RIAWELC
train/val plates.

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
| [GDXray Welds](https://github.com/computervision-xray-testing/GDXray) | BAM weld radiographs with defect masks (W0001/W0002) | research and education only, **no redistribution** | no (downloaded by notebook 03) |

Folder → class: `Difetto1` = CR (crack), `Difetto2` = PO (porosity), `Difetto4` = LP (lack of penetration),
`NoDifetto` = ND (no defect); confirmed from the per-class counts in the RIAWELC paper.

## Method

| Lab | Notebook | What it does | Runtime |
|---|---|---|---|
| 1 | `01_data_exploration` | counts, samples, intensity histograms, source IDs from file names | CPU, 1 min |
| 2 | `02_leakage_check` | pHash near-duplicate search; plate-grouped clean split | CPU, 1 min |
| 3 | `03_gdxray_prep` | GDXray download, plate matching against RIAWELC, crops and binary labels | CPU, ~20 min |
| 4 | `04_classical_baseline` | HOG + LBP + GLCM + intensity features; SVM, Random Forest, Naive Bayes | CPU, ~30 min |
| 5 | `05_cnn_baseline` | ResNet50 fine-tuning, 3 seeds, original vs clean split | T4 GPU, ~2 h |
| 6 | `06_vit_dinov2` | ViT-S/16 fine-tuning; frozen DINOv2-S + logistic regression | T4 GPU, ~1 h |
| 7 | `07_gdxray_generalisation` | zero-shot GDXray test of every model; augmentation retrain | T4 GPU, ~1 h |
| 8 | `08_anomaly_misjudgment` | PatchCore on good welds only; threshold sweep; cost-based operating points | T4 GPU, ~30 min |
| 9 | `09_gradcam_package` | Grad-CAM + pointing game on GDXray masks; README/summary from results; Gradio demo | T4 GPU, ~15 min |

Rules followed throughout: model settings and thresholds are chosen on validation data only; the test split is
touched once; GDXray is not used until the RIAWELC models are final.

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

- GDXray labels are defect / no defect only, so GDXray cannot test the per-type (CR/PO/LP) decision.
- GDXray and RIAWELC share a source (BAM). Radiographs that match RIAWELC plates are reported separately, but even
  "unseen" GDXray radiographs come from the same lab and scanner type: this is a mild domain shift, not a new factory.
- 29 source plates is a small number of independent samples; results on 6 test plates carry real uncertainty.
- The cost model (20× miss cost, 5% defect rate) is an assumption, not plant data.

## References

- Totino, Spagnolo, Perri (2023). RIAWELC: a novel dataset of radiographic images for automatic weld defects classification. IJECER.
- Palma-Ramírez et al. (2024). Deep convolutional neural network for weld defect classification in radiographic images. Heliyon.
- WeldVGG (2025). Sensors 25(19):6183.
- Mery et al. (2015). GDXray: the database of X-ray images for nondestructive testing. J. Nondestructive Evaluation 34.
- Roth et al. (2022). Towards total recall in industrial anomaly detection (PatchCore). CVPR.
- Oquab et al. (2023). DINOv2: learning robust visual features without supervision.
