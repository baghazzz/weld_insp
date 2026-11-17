# Do weld-defect classifiers generalise? A leakage audit of RIAWELC and a zero-shot test on SWRD

_Two-page summary. Tables marked "filled by notebook 09" are written from results/*.csv; edit only the prose._

## 1. Question

Published weld-defect classifiers reach ~99% accuracy on RIAWELC but ~64% macro-F1 on GDXray radiographs. We test transfer on SWRD, an independent seam-weld dataset, because GDXray shares its source radiographs with RIAWELC. Is the
drop a property of the models, or of how they were tested? And do newer models (ViT, DINOv2) or anomaly detection
trained only on good welds generalise better?

## 2. Data and the leakage finding

RIAWELC: 24,407 tiles (227×227) of crack, porosity, lack of penetration and no defect, cut from 29 BAM round-robin
radiographs. Three facts change how any result on it should be read:

- The official test folder is **2,443 byte-identical copies of training tiles**, same labels.
- 28% of validation tiles have a same-defect twin in training (pHash ≤ 6 bits); the `Img1–4` sections of a plate are
  repeat scans of the same film.
- A leakage-free split must therefore group by **source plate**. We assigned whole plates to train / val / test
  (19 / 4 / 6 plates, each class within ~4 points of 70/15/15).

<!-- BEGIN leakage -->
| comparison | % within 0 bits | % within 6 bits |
|---|---|---|
| test vs train | 100.00 | 100.00 |
| test vs train (clean) | 0.03 | 1.85 |
| val vs train | 5.29 | 27.65 |
| val vs train (clean) | 0.00 | 1.70 |
<!-- END leakage -->

## 3. Models compared

Classical (HOG, LBP, GLCM, intensity → SVM, Random Forest, Naive Bayes); ResNet50 fine-tuned (3 seeds); ViT-S/16
fine-tuned; frozen DINOv2-S with a logistic-regression head; PatchCore trained on no-defect tiles only. All on the same
clean split, seed 42, settings chosen on validation.

<!-- BEGIN headline -->
_Filled by notebook 09._
<!-- END headline -->

## 4. SWRD (external test) result

_Write 3–4 sentences: the RIAWELC → SWRD drop per model (defect vs no defect, and per-type recall), which class the models
fall back to, whether augmentation changed anything, and whether you reproduce WeldVGG's ~0.64._

<!-- BEGIN anomaly -->
_Filled by notebook 09._
<!-- END anomaly -->

## 5. Operating-point recommendation

Assumptions: a missed defect costs 20× a false reject; 5% of production welds are defective; threshold set on
validation data and applied unchanged to test and SWRD.

<!-- BEGIN operating -->
_Filled by notebook 09._
<!-- END operating -->

_Four sentences for a quality manager: which model, which threshold, what share of defects it misses, what share of
good welds it rejects, and what must happen before deployment (re-set the threshold on the plant's own radiographs)._

## Limitations and next step

1. SWRD's inclusion, undercut and lack-of-fusion classes have no RIAWELC counterpart and are scored only as defect vs no defect.
2. SWRD is one lab; a third dataset would be needed to claim broad generalisation.
3. 29 plates (6 in test) is a small number of independent samples; differences of a few points may be noise.

**Next step:** _e.g. a second external dataset from a different lab, or fine-tuning on a small labelled sample of the
target domain to measure how many target images close the gap._

## Resume bullets (generated)

<!-- BEGIN bullets -->
_Filled by notebook 09._
<!-- END bullets -->
