"""GDXray Welds: download, inspect, cut 224x224 crops, and label them (Lab 3).

What GDXray Welds contains (Mery et al., 2015):
  W0001  10 radiographs selected from the BAM weld collection, with defects annotated
  W0002  ground-truth binary images (white = defect pixels) for the W0001 radiographs
  W0003  67 digitised radiographs of the BAM round-robin test (no defect masks)
  (newer releases add W0004)

Important: RIAWELC file names start with RRT-xxR (= BAM Round Robin Test plates), so
RIAWELC was cut from the same BAM radiographs as GDXray W0003/W0001. GDXray is therefore
NOT fully independent of RIAWELC. `match_to_riawelc` measures the overlap.

The GDXray annotations mark WHERE a defect is, not WHICH type (crack / porosity / ...).
So GDXray crops get binary labels: DEFECT or ND. Lab 7 scores the 4-class models by
collapsing CR/PO/LP -> DEFECT.
"""
from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import GDXRAY_WELDS_URL, IMG_SIZE

# Crop rule parameters (written down in Lab 3; change only with a reason)
WINDOW = 80          # RIAWELC tiles are 80x80 windows ("A80" in file names), upscaled
STRIDE = 40          # 50% overlap between neighbouring windows
CORE = 60            # defect must fall inside the central 60x60 of the window
MIN_DEFECT_PX = 16   # ... with at least this many annotated pixels -> DEFECT
ND_MARGIN = 10       # ND only if no defect pixel within the window grown by 10 px
MAX_ND_PER_IMAGE = 300


GDXRAY_URLS = [
    GDXRAY_WELDS_URL,
    "https://www.dropbox.com/scl/fi/im896nbhllnbnol585fsq/Welds.zip?rlkey=u584im2jtrdxzmhrg2lcqtavv&dl=1",
    "https://dl.dropboxusercontent.com/scl/fi/im896nbhllnbnol585fsq/Welds.zip?rlkey=u584im2jtrdxzmhrg2lcqtavv&dl=1",
]


def download_welds(paths) -> Path:
    """Download GDXray Welds (~209 MB) to data/gdxray_welds.zip once, then extract
    to paths.gdxray. Returns the folder holding W0001, W0002, ...

    Dropbox sometimes answers a script with a web page instead of the file, so every
    download is checked: a file that is not a real zip is deleted and the next link
    is tried. If all fail, download Welds.zip in your browser from the GDXray README
    and upload it to data/ as gdxray_welds.zip."""
    if any(paths.gdxray.rglob("W0001")):              # already extracted
        return find_series_root(paths.gdxray)
    zip_path = paths.data / "gdxray_welds.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        print(f"Removing a bad earlier download ({zip_path.stat().st_size:,} bytes, not a zip)")
        zip_path.unlink()
    if not zip_path.exists():
        tmp = paths.work / "gdxray_download.zip"
        for url in GDXRAY_URLS:
            print("Downloading GDXray Welds (209 MB) from", url.split("?")[0], "...")
            subprocess.run(["curl", "-L", "--fail", "-sS", "-A", "Mozilla/5.0", "-o", str(tmp), url])
            if tmp.exists() and zipfile.is_zipfile(tmp):
                if tmp != zip_path:
                    import shutil
                    shutil.copy(tmp, zip_path)
                break
            head = tmp.read_bytes()[:200] if tmp.exists() else b""
            print("  not a zip, got:", head[:120])
        else:
            raise RuntimeError("GDXray download failed. Download Welds.zip in your browser from "
                               "github.com/computervision-xray-testing/GDXray and upload it to "
                               f"{zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(paths.gdxray)
    return find_series_root(paths.gdxray)


def find_series_root(folder: Path) -> Path:
    for d in [folder, *folder.rglob("*")]:
        if d.is_dir() and (d / "W0001").is_dir():
            return d
    raise FileNotFoundError(f"No W0001 folder under {folder}")


def read_radiograph(path) -> tuple[np.ndarray, dict]:
    """Return the image as 8-bit grayscale plus info about the original file."""
    im = Image.open(path)
    info = {"file": Path(path).name, "mode": im.mode, "width": im.size[0], "height": im.size[1]}
    a = np.asarray(im)
    if a.ndim == 3:
        a = np.asarray(im.convert("L"))
    info["bit_depth"] = 16 if a.dtype == np.uint16 or a.max() > 255 else 8
    if a.dtype != np.uint8:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return a, info


def inventory(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("*.png")):
        _, info = read_radiograph(f)
        info["series"] = f.parent.name
        rows.append(info)
    return pd.DataFrame(rows)


def _index(name: str) -> int:
    m = re.search(r"_(\d+)\.", name)
    return int(m.group(1)) if m else -1


def load_masks(root: Path) -> dict:
    """Ground-truth mask per W0001 image index, from W0002 binary images, or else
    from bounding boxes in a W0001 ground_truth*.txt (rows: index x1 x2 y1 y2)."""
    masks = {}
    imgs = {_index(f.name): f for f in sorted((root / "W0001").glob("*.png"))}
    w2 = root / "W0002"
    if w2.is_dir():
        for f in sorted(w2.glob("*.png")):
            m = np.asarray(Image.open(f).convert("L")) > 127
            masks[_index(f.name)] = m
    if not masks:
        for gt in (root / "W0001").glob("ground_truth*.txt"):
            boxes = np.loadtxt(gt, ndmin=2)
            for idx, x1, x2, y1, y2 in boxes[:, :5]:
                idx = int(idx)
                if idx not in imgs:
                    continue
                if idx not in masks:
                    w, h = Image.open(imgs[idx]).size
                    masks[idx] = np.zeros((h, w), bool)
                masks[idx][int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True
    # sanity: a mask much brighter than half the image is probably inverted
    for k, m in masks.items():
        if m.mean() > 0.5:
            masks[k] = ~m
    return masks


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, as RIAWELC used."""
    import cv2
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)


def weld_band(mask: np.ndarray, window: int = WINDOW) -> tuple[int, int]:
    """Rows that contain the weld seam, estimated from where defects were annotated
    (2nd-98th percentile of defect rows, +- half a window). ND crops come only from
    this band, so they show weld bead, not plate background."""
    rows = np.nonzero(mask.any(1))[0]
    if len(rows) == 0:
        return 0, mask.shape[0]
    lo, hi = np.percentile(rows, [2, 98])
    return max(0, int(lo) - window // 2), min(mask.shape[0], int(hi) + window // 2)


def cut_crops(root: Path, out_dir: Path, window: int = WINDOW, use_clahe: bool = True,
              seed: int = 42) -> pd.DataFrame:
    """Slide a WINDOW x WINDOW box over each annotated radiograph, label it with the
    rule below, resize to 224x224 and save as PNG.

    DEFECT : >= MIN_DEFECT_PX annotated pixels inside the central CORE x CORE region
    ND     : no annotated pixel inside the window grown by ND_MARGIN on every side,
             and the window lies in the weld band
    other  : ambiguous (defect touching the edge) -> discarded
    """
    rng = np.random.default_rng(seed)
    stride = max(1, window // 2)
    core = int(round(window * CORE / WINDOW))
    min_px = max(4, int(round(MIN_DEFECT_PX * (window / WINDOW) ** 2)))
    out_dir.mkdir(parents=True, exist_ok=True)
    masks = load_masks(root)
    rows = []
    for f in sorted((root / "W0001").glob("*.png")):
        idx = _index(f.name)
        if idx not in masks:
            print(f"no annotation for {f.name}, skipped")
            continue
        img, _ = read_radiograph(f)
        mask = masks[idx]
        if mask.shape != img.shape:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
        if use_clahe:
            img = clahe(img)
        b0, b1 = weld_band(mask, window)
        H, W = img.shape
        off = (window - core) // 2
        nd_rows = []
        for y in range(0, H - window + 1, stride):
            for x in range(0, W - window + 1, stride):
                core_px = int(mask[y + off:y + off + core, x + off:x + off + core].sum())
                grown = mask[max(0, y - ND_MARGIN):y + window + ND_MARGIN,
                             max(0, x - ND_MARGIN):x + window + ND_MARGIN].sum()
                if core_px >= min_px:
                    lab = "DEFECT"
                elif grown == 0 and b0 <= y and y + window <= b1:
                    lab = "ND"
                else:
                    continue
                r = dict(source=f.name, x=x, y=y, label=lab, defect_px=core_px)
                (nd_rows if lab == "ND" else rows).append((r, img[y:y + window, x:x + window]))
        if len(nd_rows) > MAX_ND_PER_IMAGE:
            keep = rng.choice(len(nd_rows), MAX_ND_PER_IMAGE, replace=False)
            nd_rows = [nd_rows[i] for i in sorted(keep)]
        rows.extend(nd_rows)
    table = []
    for r, crop in rows:
        name = f"{Path(r['source']).stem}_w{window}_x{r['x']}_y{r['y']}_{r['label']}.png"
        Image.fromarray(crop).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC).save(out_dir / name)
        table.append({**r, "window": window, "path": name})
    return pd.DataFrame(table)


def _highpass(a: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Image minus its blurred self: keeps grain and edges, removes smooth shading, so
    that two unrelated smooth tiles cannot score a high correlation."""
    import cv2
    a = a.astype(np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), sigma)


def find_riawelc_plates(root: Path, ria_table: pd.DataFrame, ria_dir: Path,
                        sizes=(40, 60, 80, 100, 120, 160), tiles_per_plate: int = 6,
                        series=("W0001",), seed: int = 42) -> pd.DataFrame:
    """Is each GDXray radiograph one of the RIAWELC source plates, and at what scale?

    For every RIAWELC plate we take a few high-contrast tiles, shrink each one to a
    candidate window size, and slide it over the GDXray radiograph with normalised
    cross-correlation (NCC, OpenCV matchTemplate). NCC is 1.0 for an exact match, and
    both are high-pass filtered first so only grain and edges are compared;
    unrelated weld texture then scores well below 0.5. Unlike pHash on a fixed crop grid,
    this finds a tile anywhere in the radiograph, so grid offsets do not matter.

    Returns one row per (radiograph, plate): best NCC, the window size that gave it,
    and the median NCC over that plate's tiles at that size."""
    import cv2
    rng = np.random.default_rng(seed)
    uniq = ria_table.drop_duplicates("md5") if "md5" in ria_table else ria_table
    templates = {}
    for plate, g in uniq.groupby("plate"):
        g = g[g.label != "ND"] if (g.label != "ND").sum() >= tiles_per_plate else g
        pick = g.sample(min(len(g), tiles_per_plate * 3), random_state=seed)
        arrs = [np.asarray(Image.open(ria_dir / p).convert("L")) for p in pick.path]
        arrs = sorted(arrs, key=lambda a: -a.std())[:tiles_per_plate]   # most textured tiles
        templates[plate] = arrs
    rows = []
    for s_name in series:
        for f in sorted((root / s_name).glob("*.png")):
            img, _ = read_radiograph(f)
            versions = {"raw": _highpass(img), "clahe": _highpass(clahe(img))}
            for plate, arrs in templates.items():
                best = (-1.0, None, None)
                for size in sizes:
                    if size >= min(img.shape):
                        continue
                    scores = []
                    for a in arrs:
                        t = _highpass(np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR)))
                        sc = max(float(cv2.matchTemplate(v, t, cv2.TM_CCOEFF_NORMED).max())
                                 for v in versions.values())
                        scores.append(sc)
                    if max(scores) > best[0]:
                        best = (max(scores), size, float(np.median(scores)))
                rows.append({"radiograph": f.name, "series": s_name, "plate": plate,
                             "best_ncc": round(best[0], 3), "window_px": best[1],
                             "median_ncc": round(best[2], 3) if best[2] is not None else np.nan})
    return pd.DataFrame(rows)


def summarise_plate_matches(m: pd.DataFrame, plate_split: pd.Series, ncc_threshold: float = 0.75,
                            min_margin: float = 0.10) -> pd.DataFrame:
    """Best plate per radiograph. A radiograph counts as a RIAWELC plate if its best NCC
    is >= ncc_threshold AND beats the second-best plate by >= min_margin (one plate must
    stand out, not just a generally smooth image). It is 'seen' if that plate sits in the
    clean train or validation split. Tested on synthetic radiographs built from RIAWELC
    tiles: true plates scored 0.91-0.96, with clear margins."""
    best = m.loc[m.groupby("radiograph").best_ncc.idxmax()].set_index("radiograph")
    second = m.groupby("radiograph").best_ncc.apply(lambda s: s.nlargest(2).iloc[-1])
    best["runner_up_ncc"] = second
    best["margin"] = (best.best_ncc - best.runner_up_ncc).round(3)
    best["is_riawelc_plate"] = (best.best_ncc >= ncc_threshold) & (best.margin >= min_margin)
    best["plate_split"] = best.plate.map(plate_split).where(best.is_riawelc_plate)
    best["seen_in_training"] = best.plate_split.isin(["train", "val"])
    return best


def locate(img: np.ndarray, tile: np.ndarray, size: int) -> tuple[int, int, float]:
    """Where in `img` does `tile` (resized to size x size) match best? -> (x, y, ncc)."""
    import cv2
    t = _highpass(np.asarray(Image.fromarray(tile).resize((size, size), Image.BILINEAR)))
    best = (0, 0, -1.0)
    for v in (_highpass(img), _highpass(clahe(img))):
        r = cv2.matchTemplate(v, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[2]:
            best = (int(loc[0]), int(loc[1]), float(mx))
    return best
