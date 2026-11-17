# %% [markdown]
# # Lab 8 — Anomaly detection and misjudgment analysis
# **Concepts:** 14 anomaly detection and PatchCore · 15 threshold, missed defects and false rejects. **Runtime: T4 GPU.**
#
# **A modern golden path.** At Tata Motors the sealant check compared each panel with a 3-sigma "golden path" of good
# parts. PatchCore does the same with deep features: it learns only what *good* welds look like and flags anything
# unlike them. It needs no defect examples, so it cannot overfit to the defect types or plates in the training set.
#
# **PatchCore in four steps** (code: `src/weld_insp/patchcore.py`):
# 1. Pass each good tile through an ImageNet WideResNet50 and keep mid-level feature maps (layers 2 and 3): one
#    feature vector per 8×8-pixel patch, smoothed over its 3×3 neighbourhood.
# 2. Pool all patch vectors of all good tiles: the "memory" of normal weld texture.
# 3. Keep a 1% **coreset** (greedy: repeatedly add the patch farthest from those already kept), so memory is small but
#    still covers all kinds of normal texture.
# 4. Score a new tile: for each of its patches, distance to the nearest memory patch; the tile score is the largest.
#
# **Why our own implementation, not anomalib:** anomalib's API changes between versions and installing it downgrades
# Colab's PyTorch stack. The method above is the published PatchCore; an optional anomalib cross-check is at the end.

# %% SETUP

# %%
if IN_COLAB:
    !pip -q install timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from weld_insp.data import get_riawelc, make_loader
from weld_insp.experiment import Experiment
from weld_insp.patchcore import PatchCore
from weld_insp.metrics import defect_score, to_binary, log_result
from weld_insp.costs import sweep, cost_per_1000, operating_point
from weld_insp.config import CLASS_TO_IDX
RIA = get_riawelc(P)
EX = Experiment(P, RIA, batch_size=16 if QUICK else 32, quick=QUICK)
VIT = "vit_small_patch16_224.augreg_in21k_ft_in1k"
d = EX.splits("clean")
gd = pd.read_csv(P.results / "gdxray_labels.csv")
GDIR = P.gdxray_crops
N_GOOD = 40 if QUICK else 1500          # good training tiles for the memory bank (memory/time trade-off)

# %% [markdown]
# ## Task 1 — train on "no defect" only, score everything

# %%
good = d["train"][d["train"].label == "ND"].sample(min(N_GOOD, (d["train"].label == "ND").sum()), random_state=SEED)
print(f"Memory bank built from {len(good)} ND training tiles; no defect tile is used.")
pc = PatchCore(pretrained=EX.pretrained, coreset_ratio=0.01)
bank_f = P.checkpoints / "patchcore_bank.pt"
if bank_f.exists() and not QUICK:
    pc.load(bank_f)
else:
    pc.fit(make_loader(good, RIA, train=False, batch_size=EX.batch_size), max_batches=EX.max_batches)
    if not QUICK:
        pc.save(bank_f)

def pc_scores(df, root):
    _, s = pc.score(make_loader(df, root, train=False, batch_size=EX.batch_size), max_batches=EX.max_batches)
    return s
sets = {"RIAWELC val": (d["val"], RIA), "RIAWELC test": (d["test"], RIA), "GDXray": (gd, GDIR)}
y_of = {"RIAWELC val": to_binary(d["val"].label.map(CLASS_TO_IDX).values),
        "RIAWELC test": to_binary(d["test"].label.map(CLASS_TO_IDX).values),
        "GDXray": (gd.label == "DEFECT").astype(int).values}
S = {"PatchCore": {k: pc_scores(df, root) for k, (df, root) in sets.items()}}

# %% [markdown]
# **Best supervised model** for comparison = the deep model with the higher RIAWELC *validation* macro-F1 (chosen
# without looking at test or GDXray). Its anomaly-style score is P(defect) = 1 − P(ND).

# %%
cand = {"ResNet50": "resnet50", "ViT fine-tuned": VIT}
val_f1 = {n: pd.read_csv(EX.run_dir(a, "clean", SEED) / "history.csv").val_macro_f1.max() for n, a in cand.items()}
SUP = max(val_f1, key=val_f1.get)
S[SUP] = {}
for k, (df, root) in sets.items():
    _, prob, _ = EX.evaluate(EX.run_dir(cand[SUP], "clean", SEED), arch=cand[SUP], df=df, image_root=root)
    S[SUP][k] = defect_score(prob)
print("supervised model:", SUP, val_f1)

def yy(k, model):
    n = len(S[model][k]); return y_of[k][:n]

auc = pd.DataFrame([{"model": m, "eval_set": k, "roc_auc": round(roc_auc_score(yy(k, m), S[m][k]), 4)}
                    for m in S for k in ["RIAWELC test", "GDXray"] if len(set(yy(k, m))) > 1])
seen = gd.seen_in_training.astype(bool).values
for m in S:
    n = len(S[m]["GDXray"]); u = ~seen[:n]
    if u.sum() and len(set(y_of["GDXray"][:n][u])) > 1:
        auc.loc[len(auc)] = [m, "GDXray unseen radiographs", round(roc_auc_score(y_of["GDXray"][:n][u], S[m]["GDXray"][u]), 4)]
for _, r in auc.iterrows():
    log_result(P.results, {"lab": 8, "model": r.model, "train_split": "clean", "eval_set": r.eval_set + " (defect vs ND)", "roc_auc": r.roc_auc})
auc.to_csv(P.results / "anomaly_auc.csv", index=False)
auc.pivot(index="model", columns="eval_set", values="roc_auc")

# %% [markdown]
# ROC-AUC = the chance that a random defective tile scores higher than a random good tile (0.5 = coin flip,
# 1.0 = perfect ranking). It does not depend on any threshold.
#
# ## Task 2 — sweep the threshold

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
sweeps = {}
for col, k in enumerate(["RIAWELC test", "GDXray"]):
    for m, c in zip(S, ["tab:blue", "tab:red"]):
        y, s = yy(k, m), S[m][k]
        sw = sweep(y, s)
        sweeps[(m, k)] = sw
        ax[0, col].plot(sw.threshold, sw.missed_defect_rate, c=c, label=f"{m}: missed defects")
        ax[0, col].plot(sw.threshold, sw.false_reject_rate, c=c, ls="--", label=f"{m}: false rejects")
        pr, rc, _ = precision_recall_curve(y, s)
        ax[1, col].plot(rc, pr, c=c, label=m)
    ax[0, col].set_title(f"{k}: error rates vs threshold"); ax[0, col].set_xlabel("threshold (each model's own score scale)")
    ax[1, col].set_title(f"{k}: precision vs recall, defect class"); ax[1, col].set_xlabel("recall"); ax[1, col].set_ylabel("precision")
    ax[0, col].legend(fontsize=7); ax[1, col].legend(fontsize=7)
plt.tight_layout(); plt.savefig(P.figures / "08_threshold_sweep.png", dpi=130); plt.show()

# %% [markdown]
# Raising the threshold makes the inspector more lenient: fewer good welds rejected, more defects missed.
# The two scores live on different scales (PatchCore: a feature distance; supervised: a probability), so compare
# curve *shapes*, or compare at equal missed-defect rate.
#
# ## Task 3 — choose an operating point from costs
# **Assumptions (they are assumptions, change them to your plant's numbers):**
# - A missed weld defect costs **20×** a false reject (rework or re-inspection). Also tried: 5× and 100×.
# - Cost unit = one false reject. Cost per 1,000 welds = 1000 × [p·MDR·ratio + (1 − p)·FRR].
# - **p, the share of defective welds**, is 5% in production (assumed); RIAWELC and GDXray test sets are far richer
#   in defects because tiles were cut around defects, so their own prevalence would mislead.
# - The threshold is **chosen on RIAWELC validation scores**, then applied unchanged to test and GDXray.

# %%
P_DEFECT = 0.05
rows = []
for m in S:
    sw_val = sweep(yy("RIAWELC val", m), S[m]["RIAWELC val"])
    for ratio in [5, 20, 100]:
        op = operating_point(sw_val, cost_ratio=ratio, defect_rate=P_DEFECT)
        for k in ["RIAWELC val", "RIAWELC test", "GDXray"]:   # val = where the threshold was chosen
            y, s = yy(k, m), S[m][k]
            pred = s >= op["threshold"]
            mdr = float((~pred & (y == 1)).sum() / max((y == 1).sum(), 1))
            frr = float((pred & (y == 0)).sum() / max((y == 0).sum(), 1))
            rows.append({"model": m, "cost_ratio": ratio, "defect_rate_assumed": P_DEFECT,
                         "threshold_from_val": round(op["threshold"], 4), "eval_set": k,
                         "missed_defect_rate": round(mdr, 4), "false_reject_rate": round(frr, 4),
                         "cost_per_1000": round(1000 * (P_DEFECT * mdr * ratio + (1 - P_DEFECT) * frr), 1)})
ops = pd.DataFrame(rows)
ops.to_csv(P.results / "operating_points.csv", index=False)
ops

# %%
fig, ax = plt.subplots(1, len(S), figsize=(6 * len(S), 3.8), squeeze=False)
for a, m in zip(ax[0], S):
    sw_val = sweep(yy("RIAWELC val", m), S[m]["RIAWELC val"])
    for ratio in [5, 20, 100]:
        c = cost_per_1000(sw_val, ratio, P_DEFECT)
        a.plot(sw_val.threshold, c, label=f"miss = {ratio}× false reject")
        a.plot(sw_val.threshold[c.idxmin()], c.min(), "ko", ms=4)
    a.set_title(f"{m}: cost per 1,000 welds (validation)"); a.set_xlabel("threshold"); a.set_ylabel("cost (false-reject units)")
    a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(P.figures / "08_cost_curves.png", dpi=130); plt.show()

# %% [markdown]
# **Does the chosen threshold move with the cost ratio?** _Read the dots: a higher miss cost pushes the optimum to a
# lower threshold (reject more to miss less)._
#
# %%
# Recommended model = lowest cost on VALIDATION at the 20x assumption; then report its test and GDXray rates
v20 = ops[(ops.cost_ratio == 20) & (ops.eval_set == "RIAWELC val")]
REC = v20.loc[v20.cost_per_1000.idxmin(), "model"]
print("Recommended (lowest validation cost at 20x):", REC)
ops[(ops.model == REC) & (ops.cost_ratio == 20)]

# %% [markdown]
# **To the quality manager (four sentences, fill from the table above):**
# _We recommend [model] at threshold [t], set on validation data assuming a missed defect costs 20× a rework.
# On held-out RIAWELC plates it misses [x]% of defective welds and rejects [y]% of good ones.
# On GDXray radiographs from another source the miss rate rises to [x′]%, so [what that means for deployment].
# Before go-live the threshold should be re-set on a sample of the plant's own radiographs._

# %% [markdown]
# ### Optional: cross-check with anomalib
# Set `USE_ANOMALIB = True` to install anomalib **in this notebook only** and compare its PatchCore ROC-AUC on
# RIAWELC test with ours. anomalib's API changes between versions; this cell follows the 1.x/2.x `Engine` API.

# %%
USE_ANOMALIB = False
if USE_ANOMALIB:
    !pip -q install anomalib
    import shutil
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    root = Path("/content/anomalib_data")
    for sub, df in [("good", good), ("test_good", d["test"][d["test"].label == "ND"]),
                    ("test_defect", d["test"][d["test"].label != "ND"])]:
        (root / sub).mkdir(parents=True, exist_ok=True)
        for p in df.path:
            shutil.copy(RIA / p, root / sub / Path(p).name)
    dm = Folder(name="riawelc", root=root, normal_dir="good", abnormal_dir="test_defect", normal_test_dir="test_good")
    engine = Engine()
    engine.fit(model=Patchcore(), datamodule=dm)
    print(engine.test(model=engine.model, datamodule=dm))

# %% [markdown]
# ## Notes — fill in from your outputs
# - **Did:** built a PatchCore memory bank from good RIAWELC tiles only, scored RIAWELC test and GDXray, compared ROC-AUC with the best supervised model, swept thresholds and chose cost-based operating points on validation data.
# - **Found:** _PatchCore vs supervised AUC on RIAWELC and on GDXray (does the golden path transfer better?); the missed-defect and false-reject rates at the 20× point._
# - **Surprised me:** _…_
