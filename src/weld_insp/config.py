"""Project paths and constants, shared by every notebook.

The same code runs in three places:
  * Google Colab, with the project folder on Drive at MyDrive/weld_insp
  * A local PC, e.g. D:\\weld_insp
  * Any folder named by the WELD_ROOT environment variable

`get_paths()` works out where you are and returns every folder the labs use.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- classes
# Folder names in the RIAWELC release are Italian ("Difetto" = defect).
# Mapping confirmed from the per-class counts in Table I of Totino et al. (2023):
#   CR 4962/1908/765, PO 4108/1580/632, LP 2893/1113/446, ND 3900/1500/600
FOLDER_TO_CLASS = {
    "Difetto1": "CR",   # crack
    "Difetto2": "PO",   # porosity
    "Difetto4": "LP",   # lack of penetration
    "NoDifetto": "ND",  # no defect
}
CLASSES = ["CR", "PO", "LP", "ND"]           # fixed order = model output order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEFECT_CLASSES = ["CR", "PO", "LP"]
CLASS_NAMES = {"CR": "crack", "PO": "porosity", "LP": "lack of penetration", "ND": "no defect"}

# Official RIAWELC split folders -> short names used in our tables
SPLIT_FOLDERS = {"training": "train", "validation": "val", "testing": "test"}

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED = 42

RIAWELC_REPO = "https://github.com/stefyste/RIAWELC.git"
GDXRAY_WELDS_URL = (
    "https://www.dropbox.com/scl/fi/im896nbhllnbnol585fsq/Welds.zip"
    "?rlkey=u584im2jtrdxzmhrg2lcqtavv&st=sswtvnmd&dl=1"
)


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Paths:
    root: Path          # project folder (Drive or D:\weld_insp)
    data: Path          # zips and prepared data live here (persistent)
    results: Path       # csv / npy / figures (persistent)
    figures: Path
    checkpoints: Path   # model weights (persistent)
    work: Path          # fast scratch: /content on Colab, data/ locally
    riawelc: Path       # extracted RIAWELC images (training/ validation/ testing/)
    gdxray: Path        # extracted GDXray Welds folder
    gdxray_crops: Path  # our 224x224 GDXray crops (persistent)

    def show(self) -> None:
        for k, v in self.__dict__.items():
            print(f"{k:13s} {v}")


def find_root() -> Path:
    """Locate the project folder."""
    if os.environ.get("WELD_ROOT"):
        return Path(os.environ["WELD_ROOT"])
    if in_colab():
        return Path("/content/drive/MyDrive/weld_insp")
    # Local: walk up from the current folder until we find src/weld_insp
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "src" / "weld_insp").is_dir():
            return p
    return here


def get_paths(mount_drive: bool = True) -> Paths:
    if in_colab() and mount_drive and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")
    root = find_root()
    work = Path("/content") if in_colab() else root / "data"
    p = Paths(
        root=root,
        data=root / "data",
        results=root / "results",
        figures=root / "results" / "figures",
        checkpoints=root / "checkpoints",
        work=work,
        riawelc=work / "riawelc",
        gdxray=work / "gdxray",
        gdxray_crops=root / "data" / "gdxray_crops",
    )
    for d in (p.data, p.results, p.figures, p.checkpoints, p.work):
        d.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = SEED) -> None:
    """Fix Python, NumPy and (if installed) PyTorch seeds so runs repeat."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
