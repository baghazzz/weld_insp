"""Hand-made image descriptors for the classical baseline (Lab 4).

Feature families (and their column ranges, see FEATURE_FAMILIES):
  HOG        shape of edges: histograms of gradient direction in small cells
  LBP        micro-texture: how each pixel compares with its neighbours
  GLCM       texture statistics of grey-level co-occurrence (contrast, homogeneity,
             energy, correlation) at 3 distances, averaged over 4 directions
  INTENSITY  32-bin brightness histogram + mean, std, 5/50/95th percentiles
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

HOG_SIZE = 128            # image is resized to 128x128 for HOG (keeps ~1.7k dims)
HOG_PPC = 16
LBP_SETTINGS = [(8, 1), (16, 2)]
GLCM_DIST = [1, 3, 5]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]
N_HIST = 32


def extract_one(path: str) -> np.ndarray:
    from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

    g = np.asarray(Image.open(path).convert("L").resize((224, 224), Image.BILINEAR))
    small = np.asarray(Image.fromarray(g).resize((HOG_SIZE, HOG_SIZE), Image.BILINEAR))
    f_hog = hog(small, orientations=9, pixels_per_cell=(HOG_PPC, HOG_PPC),
                cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    f_lbp = []
    for P, R in LBP_SETTINGS:
        lbp = local_binary_pattern(g, P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=np.arange(P + 3), density=True)
        f_lbp.append(h)
    f_lbp = np.concatenate(f_lbp)
    q = (g // 8).astype(np.uint8)                         # 32 grey levels for GLCM
    glcm = graycomatrix(q, GLCM_DIST, GLCM_ANGLES, levels=32, symmetric=True, normed=True)
    f_glcm = np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in GLCM_PROPS])
    h, _ = np.histogram(g, bins=N_HIST, range=(0, 256), density=True)
    f_int = np.concatenate([h, [g.mean() / 255, g.std() / 255],
                            np.percentile(g, [5, 50, 95]) / 255])
    return np.concatenate([f_hog, f_lbp, f_glcm, f_int]).astype(np.float32)


def feature_families() -> dict:
    """Column slices of each feature family in the vector from extract_one."""
    n_cells = HOG_SIZE // HOG_PPC
    n_hog = (n_cells - 1) ** 2 * 4 * 9
    n_lbp = sum(P + 2 for P, _ in LBP_SETTINGS)
    n_glcm = len(GLCM_PROPS) * len(GLCM_DIST)
    n_int = N_HIST + 5
    edges = np.cumsum([0, n_hog, n_lbp, n_glcm, n_int])
    names = ["HOG", "LBP", "GLCM", "INTENSITY"]
    return {n: slice(int(a), int(b)) for n, a, b in zip(names, edges[:-1], edges[1:])}


FEATURE_FAMILIES = feature_families()


def extract_all(paths, workers: int = 4, chunksize: int = 64) -> np.ndarray:
    paths = [str(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        feats = list(ex.map(extract_one, paths, chunksize=chunksize))
    return np.vstack(feats)
