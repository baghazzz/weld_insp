"""Near-duplicate (leakage) detection with perceptual hashes, and a leakage-free split.

A perceptual hash (pHash) turns an image into 64 bits that change little when the
image changes little. The Hamming distance between two hashes is the number of
bits that differ: 0 = same picture, ~32 = unrelated pictures.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _phash_one(path: str, hash_size: int = 8) -> int:
    import imagehash
    from PIL import Image
    h = imagehash.phash(Image.open(path).convert("L"), hash_size=hash_size)
    bits = h.hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def compute_phashes(paths: list, cache: Path | None = None, workers: int = 4) -> np.ndarray:
    """64-bit pHash (hash_size 8) for every image, as uint64. Cached to `cache` (.npy)."""
    if cache is not None and Path(cache).exists():
        arr = np.load(cache)
        if len(arr) == len(paths):
            return arr
        print(f"Cache {cache} has {len(arr)} hashes but {len(paths)} images; recomputing.")
    paths = [str(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(_phash_one, paths, chunksize=256))
    arr = np.array(hashes, dtype=np.uint64)
    if cache is not None:
        np.save(cache, arr)
    return arr


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):          # NumPy >= 2.0
        return np.bitwise_count(x)
    x = x.astype(np.uint64)                   # classic SWAR popcount
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def nearest_hamming(query: np.ndarray, ref: np.ndarray, chunk: int = 512):
    """For each query hash, the Hamming distance to (and index of) its nearest ref hash."""
    best_d = np.empty(len(query), dtype=np.int32)
    best_i = np.empty(len(query), dtype=np.int64)
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk, None]
        d = _popcount(q ^ ref[None, :]).astype(np.int16)
        best_i[s:s + chunk] = d.argmin(1)
        best_d[s:s + chunk] = d.min(1)
    return best_d, best_i


def leakage_table(dist: np.ndarray, thresholds=(0, 2, 4, 6, 8)) -> pd.DataFrame:
    return pd.DataFrame({
        "threshold_bits": thresholds,
        "n_within": [int((dist <= t).sum()) for t in thresholds],
        "pct_within": [round(100 * float((dist <= t).mean()), 2) for t in thresholds],
    })

# ---------------------------------------------------------- grouping / split

class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        root = a
        while self.p[root] != root:
            root = self.p[root]
        while self.p[a] != root:          # path compression
            self.p[a], a = root, self.p[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def hash_clusters(hashes: np.ndarray, threshold: int, chunk: int = 512) -> np.ndarray:
    """Link any two images within `threshold` bits; return a cluster id per image."""
    uf = UnionFind(len(hashes))
    for s in range(0, len(hashes), chunk):
        d = _popcount(hashes[s:s + chunk, None] ^ hashes[None, :])
        ii, jj = np.nonzero(d <= threshold)
        for i, j in zip(ii + s, jj):
            if i < j:
                uf.union(i, j)
    return uf.labels()


def merge_groups(*group_arrays) -> np.ndarray:
    """Union several groupings: two images end up together if ANY grouping links them."""
    n = len(group_arrays[0])
    uf = UnionFind(n)
    for g in group_arrays:
        first = {}
        for i, key in enumerate(g):
            if key in first:
                uf.union(first[key], i)
            else:
                first[key] = i
    return uf.labels()


def group_split(df: pd.DataFrame, group_col: str, label_col: str = "label",
                fractions=(0.70, 0.15, 0.15), n_trials: int = 3000, seed: int = 42) -> pd.Series:
    """Assign whole groups to train / val / test so that, per class, the counts are as
    close as possible to `fractions`. Random search over group orders, greedy filling,
    keep the best. Returns a Series of split names aligned with df.index."""
    rng = np.random.default_rng(seed)
    names = ["train", "val", "test"]
    counts = df.groupby([group_col, label_col]).size().unstack(fill_value=0)
    groups = counts.index.to_numpy()
    C = counts.to_numpy(dtype=float)                       # groups x classes
    target = np.outer(fractions, C.sum(0))                 # splits x classes
    best, best_err = None, np.inf
    for _ in range(n_trials):
        order = rng.permutation(len(groups))
        # put the biggest groups first, random among equals, so greedy works
        order = order[np.argsort(-C[order].sum(1), kind="stable")] if rng.random() < 0.5 else order
        have = np.zeros_like(target)
        assign = np.empty(len(groups), dtype=int)
        for gi in order:
            # choose the split whose relative deficit is largest after adding this group
            gaps = []
            for s in range(3):
                after = have[s] + C[gi]
                gaps.append(np.abs(after - target[s]).sum() - np.abs(have[s] - target[s]).sum())
            s = int(np.argmin(gaps))
            assign[gi] = s
            have[s] += C[gi]
        err = (np.abs(have - target) / np.maximum(target, 1)).mean()
        if np.all(have.sum(1) > 0) and np.all(have > 0) and err < best_err:
            best, best_err = assign.copy(), err
    if best is None:
        raise RuntimeError("Could not place every class in every split; too few groups.")
    mapping = {g: names[a] for g, a in zip(groups, best)}
    print(f"group_split: mean relative error per class/split = {best_err:.3f}")
    return df[group_col].map(mapping)
