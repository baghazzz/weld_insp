"""Train / evaluate helpers shared by Labs 5, 6 and 7, so every deep model uses the same
split, seed, loaders and metrics (a fair comparison)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import set_seed
from .data import load_split, make_loader
from .train import build_model, load_best, predict, train_model


@dataclass
class Experiment:
    paths: object                 # config.Paths
    image_root: Path              # extracted RIAWELC folder
    epochs: int = 10
    batch_size: int = 32
    quick: bool = False           # smoke-test mode: tiny subsets, random weights, few batches

    @property
    def pretrained(self) -> bool:
        return not self.quick

    @property
    def max_batches(self):
        return 3 if self.quick else None

    def splits(self, which: str = "clean") -> dict:
        df = load_split(self.paths, which)
        if self.quick:
            df = df.sample(frac=1, random_state=42).groupby(["split", "label"]).head(24)
        return {s: df[df.split == s] for s in ["train", "val", "test"]}

    def run_dir(self, arch, which, seed, tag=""):
        return self.paths.checkpoints / f"{arch.replace('/', '_')}_{which}_seed{seed}{tag}"

    def train(self, arch="resnet50", which="clean", seed=42, lr=1e-4, augment=False, tag=""):
        """Train one model (resumes if its checkpoint folder already has epochs)."""
        set_seed(seed)
        d = self.splits(which)
        tl = make_loader(d["train"], self.image_root, train=True, augment=augment,
                         batch_size=self.batch_size, seed=seed)
        vl = make_loader(d["val"], self.image_root, train=False, batch_size=self.batch_size)
        model = build_model(arch, pretrained=self.pretrained)
        rd = self.run_dir(arch, which, seed, tag)
        hist, chosen = train_model(model, tl, vl, rd, epochs=self.epochs, lr=lr, patience=3,
                                   max_batches=self.max_batches)
        return hist, chosen, rd

    def load(self, run_dir, arch="resnet50"):
        return load_best(build_model(arch, pretrained=False), run_dir)

    def evaluate(self, run_dir, arch="resnet50", which="clean", split="test", df=None, image_root=None):
        """-> y_true, softmax probabilities, predicted class. Pass df/image_root to score
        other data (e.g. GDXray crops)."""
        model = self.load(run_dir, arch)
        df = self.splits(which)[split] if df is None else df
        loader = make_loader(df, image_root or self.image_root, train=False, batch_size=self.batch_size)
        y, prob = predict(model, loader, max_batches=self.max_batches)
        return y, prob, prob.argmax(1)
