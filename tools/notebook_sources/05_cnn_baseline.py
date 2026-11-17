# %% [markdown]
# # Lab 5 — CNN baseline (ResNet50)
# **Concepts:** 9 CNN, transfer learning and fine-tuning · 10 overfitting, early stopping, random seeds. **Runtime: T4 GPU**
# (Runtime → Change runtime type → T4 GPU).
#
# **Transfer learning.** ResNet50 was trained on 1.2 million ImageNet photos; its early layers already detect edges,
# blobs and textures. We keep those weights, replace the last layer (1000 ImageNet classes) with a 4-class layer, and
# **fine-tune** the whole network on weld tiles with a small learning rate.
#
# **Overfitting.** Training loss keeps falling while validation loss stops falling (or rises): the model is
# memorising training tiles. We keep the epoch with the best *validation* macro-F1 (**early stopping**) and stop
# after 3 epochs without improvement.
#
# **Seeds.** Weight init of the new head, data order and dropout are random. Fixing the seed makes a run repeatable;
# training with several seeds shows how much of a difference between models is just luck.
#
# Checkpoints go to `checkpoints/` on Drive after every epoch. If Colab disconnects, re-run the notebook: training
# resumes from the last saved epoch.

# %% SETUP

# %%
if IN_COLAB:
    !pip -q install timm torchmetrics
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from weld_insp.data import get_riawelc, make_loader
from weld_insp.experiment import Experiment
from weld_insp.train import build_model, plot_history, count_trainable, device
from weld_insp.metrics import summary, classification_report_df, plot_confusion, log_result
print("device:", device(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
RIA = get_riawelc(P)

LR = 1e-4
EX = Experiment(P, RIA, epochs=2 if QUICK else 10, batch_size=16 if QUICK else 32, quick=QUICK)
PRETRAINED = EX.pretrained        # QUICK mode runs offline with random weights

# %% [markdown]
# ## Task 1 — the Dataset and the training run
# `WeldDataset` (in `src/weld_insp/data.py`) reads a row of the split CSV, opens the PNG as grayscale, copies it to
# 3 channels (ResNet expects RGB), resizes to 224×224 and normalises with the ImageNet mean and std. The tiles are
# 227×227, so the resize is a slight shrink.

# %%
d = EX.splits("clean")
print({k: len(v) for k, v in d.items()})
x, y = next(iter(make_loader(d["train"], RIA, train=True, batch_size=8)))
print("batch:", tuple(x.shape), x.dtype, "| labels:", y.tolist())
m = build_model("resnet50", pretrained=PRETRAINED)
print(f"ResNet50: {count_trainable(m) / 1e6:.1f} M trainable parameters; new head: {m.get_classifier()}")
del m

# %%
hist, chosen, RUN = EX.train("resnet50", "clean", seed=SEED, lr=LR)

# %% [markdown]
# ## Task 2 — learning curves and the test score

# %%
plot_history(hist, chosen, title="ResNet50, clean split, seed 42", save=P.figures / "05_resnet50_curves.png")
plt.show()
hist.round(4)

# %% [markdown]
# **Chosen epoch** = highest validation macro-F1 (dotted line). Write one sentence on why: e.g. after it, training
# loss keeps dropping but validation loss rises or flattens, which is the start of overfitting.

# %%
y, prob, pred = EX.evaluate(RUN)
np.save(P.results / "resnet50_clean_seed42_test_probs.npy", prob)
s = summary(y, pred)
log_result(P.results, {"lab": 5, "model": "ResNet50", "seed": SEED, "train_split": "clean",
                       "eval_set": "RIAWELC test (clean)", "chosen_epoch": chosen, **s})
display(classification_report_df(y, pred))
plot_confusion(y, pred, title="ResNet50, clean test split", save=P.figures / "05_resnet50_confusion.png"); plt.show()
s

# %% [markdown]
# ### Compare with the published ≈ 98% (Palma-Ramírez et al., 2024)
# If you are more than 2 points below, two likely causes:
# 1. **Leakage removed.** Published numbers use RIAWELC's own folders, where Lab 2 showed the test set is copies of
#    training tiles. Our clean split tests on plates the model has never seen.
# 2. **Protocol differences.** They used a balanced 1,600-image subset, 5-fold cross-validation and a 400-image test
#    set; we train on all ~15k tiles with no class balancing or augmentation.
#
# **Test cause 1:** same settings, trained and tested on the original split.

# %%
hist_o, chosen_o, RUN_O = EX.train("resnet50", "original", seed=SEED, lr=LR)
y_o, prob_o, pred_o = EX.evaluate(RUN_O, which="original")
s_o = summary(y_o, pred_o)
log_result(P.results, {"lab": 5, "model": "ResNet50", "seed": SEED, "train_split": "original",
                       "eval_set": "RIAWELC test (original)", "chosen_epoch": chosen_o, **s_o})
pd.DataFrame([{"split": "original (leaky)", **s_o}, {"split": "clean (by plate)", **s}])[["split", "accuracy", "macro_f1"]]

# %% [markdown]
# ## Task 3 — against the classical baseline, and seed-to-seed spread

# %%
runs = [(SEED, s["macro_f1"], s["accuracy"])]
for seed in [7, 123]:
    _, _, rd = EX.train("resnet50", "clean", seed=seed, lr=LR)
    yy, _, pp = EX.evaluate(rd)
    ss = summary(yy, pp)
    log_result(P.results, {"lab": 5, "model": "ResNet50", "seed": seed, "train_split": "clean",
                           "eval_set": "RIAWELC test (clean)", **ss})
    runs.append((seed, ss["macro_f1"], ss["accuracy"]))
seeds = pd.DataFrame(runs, columns=["seed", "test_macro_f1", "test_acc"])
print(f"ResNet50 test macro-F1 over 3 seeds: {seeds.test_macro_f1.mean():.4f} ± {seeds.test_macro_f1.std():.4f} (mean ± std)")
seeds

# %%
base = pd.read_csv(P.results / "baseline_metrics.csv")
base = base[base.split == "clean"][["model", "test_acc", "test_macro_f1"]]
cnn = pd.DataFrame([{"model": "ResNet50 (seed 42)", "test_acc": s["accuracy"], "test_macro_f1": s["macro_f1"]},
                    {"model": "ResNet50 (mean of 3 seeds)", "test_acc": seeds.test_acc.mean(),
                     "test_macro_f1": seeds.test_macro_f1.mean()}])
comp = pd.concat([base, cnn], ignore_index=True)
comp.to_csv(P.results / "cnn_metrics.csv", index=False)
best_classical = base.test_macro_f1.max()
gap, spread = seeds.test_macro_f1.mean() - best_classical, seeds.test_macro_f1.std()
print(f"Gap to best classical model: {gap:+.4f} macro-F1; seed spread (std): {spread:.4f} -> "
      f"{'gap is larger than the seed spread' if abs(gap) > 2 * spread else 'gap is within ~2 std of seed noise'}")
comp

# %% [markdown]
# ## Notes — fill in from your outputs
# - **Did:** fine-tuned ImageNet ResNet50 on the clean (plate-grouped) split, 3 seeds, plus one run on the original split.
# - **Found:** _clean-split test accuracy / macro-F1 vs the original split; which class is weakest; gap to SVM/RF vs seed spread._
# - **Surprised me:** _…_
