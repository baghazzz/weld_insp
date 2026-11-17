"""Getting RIAWELC, building the image table, and the PyTorch Dataset."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import (CLASS_TO_IDX, FOLDER_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD,
                     IMG_SIZE, RIAWELC_REPO, SPLIT_FOLDERS)

# ------------------------------------------------------------------ download

def _has_images(folder: Path) -> bool:
    return all((folder / s).is_dir() for s in SPLIT_FOLDERS) and any(folder.rglob("*.png"))


def _extract_rar(first_part: Path, dest: Path) -> None:
    """Extract a multi-part RAR. Tries the unrar / 7z programs, then Python."""
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (["unrar", "x", "-o+", "-inul", str(first_part), str(dest) + "/"],
                ["7z", "x", "-y", f"-o{dest}", str(first_part)]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=True)
            return
    # Pure-Python fallback (pip install unrar-cffi). Slow (~15 min) but works anywhere.
    from unrar.cffi import rarfile  # type: ignore
    rf = rarfile.RarFile(str(first_part))
    for info in rf.infolist():
        if info.filename.lower().endswith(".png"):
            out = dest / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(rf.read(info))


def get_riawelc(paths, keep_zip: bool = True) -> Path:
    """Make sure the RIAWELC images are extracted at paths.riawelc and return that folder.

    Order of preference:
      1. images already extracted (fast path on re-runs)
      2. data/riawelc.zip exists (Task sheet: upload riawelc.zip to data/)
      3. clone github.com/stefyste/RIAWELC, unpack the RAR parts, and save
         data/riawelc.zip so the next session can use path 2.
    """
    target = paths.riawelc
    if _has_images(target):
        return target
    zip_path = paths.data / "riawelc.zip"
    if zip_path.exists():
        local_zip = paths.work / "riawelc.zip"
        if local_zip != zip_path:
            shutil.copy(zip_path, local_zip)       # Drive -> /content is much faster to unzip
        with zipfile.ZipFile(local_zip) as z:
            z.extractall(paths.work)
        _flatten(paths.work, target)
        return target
    repo = paths.work / "RIAWELC_repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", RIAWELC_REPO, str(repo)], check=True)
    parts = sorted((repo / "Dataset_partitioned").glob("*.part01.rar"))
    tmp = paths.work / "riawelc_raw"
    _extract_rar(parts[0], tmp)
    _flatten(tmp, target)
    if keep_zip:
        print("Saving data/riawelc.zip for next time ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for f in target.rglob("*.png"):
                z.write(f, Path("riawelc") / f.relative_to(target))
    return target


def _flatten(src: Path, target: Path) -> None:
    """Find the folder that holds training/validation/testing and move it to `target`."""
    if _has_images(target):
        return
    for cand in [src, *src.rglob("*")]:
        if cand.is_dir() and _has_images(cand):
            if cand != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cand), str(target))
            return
    raise FileNotFoundError(f"No training/validation/testing folders found under {src}")

# --------------------------------------------------------------- image table

# Example: RRT-12R_Img1_A80_S2_[6][104].png
#   RRT-12R  source radiograph (BAM Round Robin Test plate = GDXray series W0003)
#   Img1     section of that radiograph
#   A80      80 x 80 pixel window (upscaled to 224/227 afterwards)
#   S2       step / scale index of the tiling routine
#   [6][104] row and column of the window in the tiling grid
NAME_RE = re.compile(
    r"^(?P<plate>[^_]+)_(?P<section>Img\d+)_(?P<window>A\d+)_(?P<step>S\d+)_"
    r"\[(?P<row>\d+)\]\[(?P<col>\d+)\](?P<copy>.*)\.png$")


def parse_name(name: str) -> dict:
    m = NAME_RE.match(name)
    if not m:
        return {"plate": name.split("_")[0], "section": None, "window": None, "step": None,
                "row": np.nan, "col": np.nan, "is_copy": "copia" in name.lower()}
    d = m.groupdict()
    return {"plate": d["plate"], "section": d["section"], "window": d["window"],
            "step": d["step"], "row": int(d["row"]), "col": int(d["col"]),
            "is_copy": "copia" in d["copy"].lower()}


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build_image_table(riawelc_dir: Path, with_md5: bool = True, with_size: bool = True) -> pd.DataFrame:
    rows = []
    for split_folder, split in SPLIT_FOLDERS.items():
        for cls_folder, label in FOLDER_TO_CLASS.items():
            for f in sorted((riawelc_dir / split_folder / cls_folder).glob("*.png")):
                r = {"path": f.relative_to(riawelc_dir).as_posix(), "split": split,
                     "label": label, "folder": cls_folder, "file": f.name}
                r.update(parse_name(f.name))
                r["radiograph"] = f"{r['plate']}_{r['section']}"
                if with_size:
                    with Image.open(f) as im:
                        r["width"], r["height"] = im.size
                if with_md5:
                    r["md5"] = md5(f)
                rows.append(r)
    return pd.DataFrame(rows)


def load_gray(path, size: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("L")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im)

# -------------------------------------------------------------- torch dataset

def build_transforms(train: bool = False, augment: bool = False, img_size: int = IMG_SIZE):
    """Eval: resize + ImageNet normalise. Train with augment=True adds the Lab 7
    robustness augmentations: Gaussian noise, blur, brightness/contrast jitter."""
    import torch
    from torchvision import transforms as T

    ops = [T.Resize((img_size, img_size)), T.Grayscale(num_output_channels=3)]
    if train and augment:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.5)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        ]
    ops.append(T.ToTensor())
    if train and augment:
        ops.append(T.RandomApply([AddGaussianNoise(0.0, 0.05)], p=0.5))
    ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean, self.std = mean, std

    def __call__(self, x):
        import torch
        return (x + torch.randn_like(x) * self.std + self.mean).clamp(0, 1)

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


def _torch_dataset_base():
    from torch.utils.data import Dataset
    return Dataset


class WeldDataset(_torch_dataset_base()):
    """Reads rows of a split CSV. Grayscale -> 3 channels, resize, ImageNet normalise.

    df needs a `path` column (relative to `image_root`) and, for labelled data,
    a `label` column holding CR / PO / LP / ND.
    """

    def __init__(self, df: pd.DataFrame, image_root, transform=None, label_col: str = "label"):
        self.df = df.reset_index(drop=True)
        self.root = Path(image_root)
        self.transform = transform or build_transforms(train=False)
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("L")
        x = self.transform(img)
        lab = row[self.label_col] if self.label_col in row.index else None
        y = CLASS_TO_IDX.get(lab, -1)   # -1 = unlabelled (e.g. a binary GDXray crop)
        return x, y


def make_loader(df, image_root, train: bool, augment: bool = False, batch_size: int = 32,
                num_workers: int = 2, seed: int = 42):
    import torch
    from torch.utils.data import DataLoader
    ds = WeldDataset(df, image_root, build_transforms(train=train, augment=augment))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g,
                      worker_init_fn=_seed_worker, drop_last=False)


def _seed_worker(worker_id):
    import random
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def load_split(paths, which: str = "clean") -> pd.DataFrame:
    """Return the image table with the `split` column set to the chosen split.

    which="original": the official RIAWELC train/val/test folders.
    which="clean":    the leakage-free split from Lab 2 (results/clean_split.csv).
    """
    if which == "original":
        return pd.read_csv(paths.results / "image_table.csv")
    f = paths.results / "clean_split.csv"
    if not f.exists():
        raise FileNotFoundError("Run Lab 2 first: results/clean_split.csv is missing.")
    return pd.read_csv(f)
