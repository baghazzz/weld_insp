"""Model building, a resumable training loop, and prediction (Labs 5-7)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES
from .metrics import summary


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(name: str = "resnet50", num_classes: int = len(CLASSES), pretrained: bool = True):
    """Any timm model with a fresh `num_classes` head.
    resnet50                        -> Lab 5
    vit_small_patch16_224           -> Lab 6 (fine-tuned ViT)
    vit_small_patch14_dinov2.lvd142m -> Lab 6 (frozen DINOv2 backbone, see dino_features)
    """
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_epoch(model, loader, optimizer=None, scaler=None, dev=None):
    import torch
    import torch.nn.functional as F
    train = optimizer is not None
    model.train(train)
    tot_loss, ys, ps = 0.0, [], []
    use_amp = dev.type == "cuda"
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            tot_loss += loss.item() * len(y)
            ys.append(y.cpu().numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    s = summary(y, p)
    return tot_loss / len(y), s["accuracy"], s["macro_f1"]


def train_model(model, train_loader, val_loader, run_dir: Path, epochs: int = 10, lr: float = 1e-4,
                weight_decay: float = 0.05, patience: int = 3, max_batches: int | None = None,
                keep_epochs: bool = False):
    """Train with AdamW. After every epoch: save a checkpoint (last.pth, with optimizer
    state, overwritten each epoch) to run_dir on Drive and append to history.csv;
    best.pth holds the epoch with the best validation macro-F1. keep_epochs=True also
    keeps epoch_XX.pth for every epoch (~95 MB each for ResNet50). If the run_dir already holds checkpoints (Colab
    disconnected), training resumes from the last one.

    The 'chosen' epoch is the one with the best validation macro-F1; training stops
    early when it has not improved for `patience` epochs.
    """
    import torch
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    hist_f = run_dir / "history.csv"
    history = pd.read_csv(hist_f).to_dict("records") if hist_f.exists() else []
    start = 0
    last = run_dir / "last.pth"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["epoch"] + 1
        print(f"Resuming from epoch {start} ({last})")

    if max_batches:  # quick smoke tests: only a few batches per epoch
        train_loader = _limit(train_loader, max_batches)
        val_loader = _limit(val_loader, max_batches)

    best = max((h["val_macro_f1"] for h in history), default=-1)
    stale = 0 if not history else len(history) - 1 - int(np.argmax([h["val_macro_f1"] for h in history]))
    for ep in range(start, epochs):
        if stale >= patience:
            print(f"Early stop: no val macro-F1 gain for {patience} epochs.")
            break
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = _run_epoch(model, train_loader, opt, scaler, dev)
        va_loss, va_acc, va_f1 = _run_epoch(model, val_loader, dev=dev)
        row = dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, train_macro_f1=tr_f1,
                   val_loss=va_loss, val_acc=va_acc, val_macro_f1=va_f1, seconds=round(time.time() - t0, 1))
        history.append(row)
        pd.DataFrame(history).to_csv(hist_f, index=False)
        if keep_epochs:
            torch.save(model.state_dict(), run_dir / f"epoch_{ep:02d}.pth")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep}, last)
        if va_f1 > best:
            best, stale = va_f1, 0
            torch.save(model.state_dict(), run_dir / "best.pth")
        else:
            stale += 1
        print(f"epoch {ep:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} macroF1 {va_f1:.4f} | {row['seconds']}s")
    hist = pd.DataFrame(history)
    chosen = int(hist.loc[hist.val_macro_f1.idxmax(), "epoch"])
    (run_dir / "chosen.json").write_text(json.dumps({"chosen_epoch": chosen}))
    return hist, chosen


class _limit:
    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n:
                break
            yield b

    def __len__(self):
        return min(self.n, len(self.loader))


def load_best(model, run_dir: Path):
    import torch
    model.load_state_dict(torch.load(Path(run_dir) / "best.pth", map_location="cpu", weights_only=True))
    return model.to(device()).eval()


def predict(model, loader, max_batches: int | None = None):
    """Return (y_true, softmax probabilities)."""
    import torch
    dev = device()
    model.to(dev).eval()
    ys, ps = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                logits = model(x.to(dev))
            ps.append(torch.softmax(logits.float(), 1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def extract_backbone_features(model, loader, max_batches: int | None = None):
    """Pooled features from a timm model created with num_classes=0 (e.g. frozen DINOv2)."""
    import torch
    dev = device()
    model.to(dev).eval()
    feats, ys = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                f = model(x.to(dev))
            feats.append(f.float().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def plot_history(hist: pd.DataFrame, chosen: int, title: str = "", save=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist.epoch, hist.train_loss, "o-", label="train")
    ax[0].plot(hist.epoch, hist.val_loss, "o-", label="validation")
    ax[0].set_ylabel("cross-entropy loss")
    ax[1].plot(hist.epoch, hist.train_acc, "o-", label="train accuracy")
    ax[1].plot(hist.epoch, hist.val_acc, "o-", label="val accuracy")
    ax[1].plot(hist.epoch, hist.val_macro_f1, "s--", label="val macro-F1")
    for a in ax:
        a.axvline(chosen, color="grey", ls=":", label=f"chosen epoch {chosen}")
        a.set_xlabel("epoch")
        a.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
