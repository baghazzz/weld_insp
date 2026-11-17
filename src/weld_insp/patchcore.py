"""A compact PatchCore (Roth et al., CVPR 2022) for the "golden path" lab (Lab 8).

Idea: describe every small patch of every GOOD weld image with mid-level CNN
features, keep a representative subset (the "memory bank"), and score a new image
by how far its most unusual patch is from anything in the bank. No defect image is
ever used for training.

This file implements the method directly (~100 lines) so every step is visible and
it runs on CPU for testing. Lab 8 can cross-check it against anomalib's Patchcore.
"""
from __future__ import annotations

import numpy as np


class PatchCore:
    def __init__(self, backbone: str = "wide_resnet50_2", layers=(2, 3), feat_dim: int = 384,
                 coreset_ratio: float = 0.01, pretrained: bool = True, device=None, seed: int = 42):
        import timm
        import torch
        self.torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                       out_indices=layers).to(self.device).eval()
        self.feat_dim = feat_dim
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.bank = None

    # ---------------------------------------------------------------- features
    def _patch_features(self, x):
        """(B,3,H,W) -> (B, h*w, feat_dim) locally-aware patch features, and (h, w)."""
        torch, F = self.torch, self.torch.nn.functional
        with torch.no_grad():
            feats = self.model(x.to(self.device))
            feats = [F.avg_pool2d(f.float(), 3, 1, 1) for f in feats]      # 3x3 neighbourhood
            h, w = feats[0].shape[-2:]
            feats = [feats[0]] + [F.interpolate(f, size=(h, w), mode="bilinear",
                                                align_corners=False) for f in feats[1:]]
            f = torch.cat(feats, 1)                                        # B, C, h, w
            B, C = f.shape[:2]
            f = f.permute(0, 2, 3, 1).reshape(B, h * w, C)
            f = F.adaptive_avg_pool1d(f, self.feat_dim)                    # reduce channels
        return f, (h, w)

    # ------------------------------------------------------------------- fit
    def fit(self, loader, max_batches: int | None = None):
        torch = self.torch
        chunks = []
        for i, (x, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            f, _ = self._patch_features(x)
            chunks.append(f.reshape(-1, f.shape[-1]).cpu())
        feats = torch.cat(chunks)
        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        idx = self._greedy_coreset(feats, n_keep)
        self.bank = feats[idx].to(self.device)
        print(f"PatchCore memory bank: {len(self.bank):,} of {len(feats):,} patches")
        return self

    def _greedy_coreset(self, feats, n_keep: int, proj_dim: int = 128):
        """k-center greedy on a random projection: repeatedly add the patch that is
        farthest from everything chosen so far, so the bank covers all of 'normal'."""
        torch = self.torch
        g = torch.Generator().manual_seed(self.seed)
        P = torch.randn(feats.shape[1], proj_dim, generator=g) / proj_dim ** 0.5
        z = (feats @ P).to(self.device)
        start = int(torch.randint(len(z), (1,), generator=g))
        chosen = [start]
        dmin = torch.cdist(z, z[start:start + 1]).squeeze(1)
        for _ in range(n_keep - 1):
            nxt = int(dmin.argmax())
            chosen.append(nxt)
            dmin = torch.minimum(dmin, torch.cdist(z, z[nxt:nxt + 1]).squeeze(1))
        return torch.tensor(chosen)

    # ----------------------------------------------------------------- score
    def score(self, loader, max_batches: int | None = None, return_maps: bool = False):
        """Image score = distance of the most anomalous patch to its nearest bank patch."""
        torch = self.torch
        scores, ys, maps = [], [], []
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            f, (h, w) = self._patch_features(x)
            d = torch.cdist(f.reshape(-1, f.shape[-1]), self.bank).min(1).values
            d = d.reshape(len(x), h * w)
            scores.append(d.max(1).values.cpu().numpy())
            ys.append(np.asarray(y))
            if return_maps:
                maps.append(d.reshape(len(x), h, w).cpu().numpy())
        out = (np.concatenate(ys), np.concatenate(scores))
        return (*out, np.concatenate(maps)) if return_maps else out

    def save(self, path):
        self.torch.save({"bank": self.bank.cpu(), "feat_dim": self.feat_dim}, path)

    def load(self, path):
        ck = self.torch.load(path, map_location=self.device, weights_only=True)
        self.bank = ck["bank"].to(self.device)
        self.feat_dim = ck["feat_dim"]
        return self
