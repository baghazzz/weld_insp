"""weld_insp: weld defect classification on RIAWELC, tested on GDXray.

Modules
  config     paths (Colab / local), class names, seeds
  data       get RIAWELC, image table, PyTorch Dataset and transforms
  leakage    perceptual hashes, near-duplicate search, leakage-free group split
  features   HOG / LBP / GLCM / intensity descriptors for the classical baseline
  metrics    accuracy, per-class P/R/F1, macro-F1, confusion matrix, result log
  train      timm models, resumable training loop, prediction, feature extraction
  gdxray     GDXray download, crops, labels, overlap check with RIAWELC
  patchcore  anomaly detection trained on good welds only
  costs      threshold sweep, missed defects vs false rejects, cost-based operating point
"""
__version__ = "1.0.0"
