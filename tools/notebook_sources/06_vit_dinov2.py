# %% [markdown]
# # Lab 6 — Vision Transformer and DINOv2
# **Concept:** 11 Vision Transformer and DINOv2. **Runtime: T4 GPU.** Same clean split, seed and metrics as Lab 5.
#
# **ViT.** Cut the 224×224 image into 16×16 patches (196 "tokens"), embed each, and let every patch attend to every
# other patch (self-attention). Unlike a CNN, it has no built-in notion that nearby pixels belong together; it learns
# that from data, which is why ViTs usually need large pretraining and a *lower* fine-tuning learning rate.
#
# **DINOv2.** A ViT trained by Meta on 142 M images *without labels* (self-supervised: two views of an image must give
# matching features). Its features are general enough that a **frozen** backbone plus a simple **linear classifier**
# often works well. Here we train only that linear layer: about 1,500 numbers instead of ~22 million.

# %% SETUP

# %%
if IN_COLAB:
    !pip -q install timm
import time, joblib, timm, torch
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from weld_insp.data import get_riawelc, make_loader
from weld_insp.experiment import Experiment
from weld_insp.train import build_model, plot_history, count_trainable, extract_backbone_features
from weld_insp.metrics import summary, classification_report_df, plot_confusion, log_result
from weld_insp.config import CLASSES, CLASS_TO_IDX
RIA = get_riawelc(P)
EX = Experiment(P, RIA, epochs=2 if QUICK else 10, batch_size=16 if QUICK else 32, quick=QUICK)

VIT = "vit_small_patch16_224.augreg_in21k_ft_in1k"
DINO = "vit_small_patch14_dinov2.lvd142m"

# %% [markdown]
# ## Task 1 — fine-tune a ViT-Small

# %%
hist, chosen, RUN_VIT = EX.train(VIT, "clean", seed=SEED, lr=3e-5)
plot_history(hist, chosen, title="ViT-S/16 fine-tuned, clean split", save=P.figures / "06_vit_curves.png"); plt.show()

# %%
y, prob, pred = EX.evaluate(RUN_VIT, arch=VIT)
np.save(P.results / "vit_clean_seed42_test_probs.npy", prob)
s_vit = summary(y, pred)
log_result(P.results, {"lab": 6, "model": "ViT-S/16 fine-tuned", "seed": SEED, "train_split": "clean",
                       "eval_set": "RIAWELC test (clean)", "chosen_epoch": chosen, **s_vit})
display(classification_report_df(y, pred))
plot_confusion(y, pred, title="ViT fine-tuned, clean test", save=P.figures / "06_vit_confusion.png"); plt.show()

# %% [markdown]
# ## Task 2 — frozen DINOv2 features + linear classifier
# `num_classes=0` removes the head, so the model returns one 384-number feature vector per image (the CLS token).
# DINOv2 was trained at 518 px with 14 px patches; `img_size=224` makes timm resize its position embeddings so it
# accepts our 224×224 tiles (16×16 patches of 14 px).

# %%
dino = timm.create_model(DINO, pretrained=EX.pretrained, num_classes=0, img_size=224)
for p in dino.parameters():
    p.requires_grad = False
d = EX.splits("clean")
feat_f = P.results / "dinov2_features.npy"
clean_all = pd.concat([d["train"], d["val"], d["test"]])
if feat_f.exists() and not QUICK:
    F_all = np.load(feat_f)
else:
    t0 = time.time()
    F_all, _ = extract_backbone_features(dino, make_loader(clean_all, RIA, train=False, batch_size=64))
    print(f"extracted {F_all.shape} in {time.time() - t0:.0f}s")
    if not QUICK:
        np.save(feat_f, F_all)
        clean_all[["path", "label", "split"]].to_csv(P.results / "dinov2_features_index.csv", index=False)
n_tr, n_va = len(d["train"]), len(d["val"])
Ftr, Fva, Fte = F_all[:n_tr], F_all[n_tr:n_tr + n_va], F_all[n_tr + n_va:]
ytr, yva, yte = (d[s].label.map(CLASS_TO_IDX).values for s in ["train", "val", "test"])
print(Ftr.shape, Fva.shape, Fte.shape)

# %%
grid = {}
for C in [0.01, 0.1, 1, 10]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Ftr, ytr)
    grid[C] = (clf, summary(yva, clf.predict(Fva))["macro_f1"])
    print(f"C={C:<5} val macro-F1 {grid[C][1]:.4f}")
best_C = max(grid, key=lambda c: grid[c][1]); lin = grid[best_C][0]
pred_d = lin.predict(Fte)
s_dino = summary(yte, pred_d)
np.save(P.results / "dinov2_linear_test_probs.npy", lin.predict_proba(Fte))
joblib.dump(lin, P.checkpoints / "dinov2_linear.joblib")
log_result(P.results, {"lab": 6, "model": "DINOv2-S + linear", "seed": SEED, "train_split": "clean",
                       "eval_set": "RIAWELC test (clean)", "C": best_C, **s_dino})
display(classification_report_df(yte, pred_d))
plot_confusion(yte, pred_d, title=f"DINOv2 frozen + logistic regression (C={best_C})",
               save=P.figures / "06_dinov2_confusion.png"); plt.show()

# %%
n_lin = lin[-1].coef_.size + lin[-1].intercept_.size
params = pd.DataFrame([
    {"model": "ResNet50 fine-tuned", "trained_parameters": count_trainable(build_model("resnet50", pretrained=False))},
    {"model": "ViT-S/16 fine-tuned", "trained_parameters": count_trainable(build_model(VIT, pretrained=False))},
    {"model": "DINOv2-S frozen + linear", "trained_parameters": n_lin},
])
params["trained_parameters"] = params.trained_parameters.map("{:,}".format)
params

# %% [markdown]
# ## Task 3 — do the classes separate in DINOv2 feature space?
# PCA keeps the two directions of largest variance (linear). t-SNE places points so that close neighbours stay
# close (non-linear, good for seeing clusters, but distances between far-apart clusters mean little).

# %%
rng = np.random.default_rng(SEED)
sub = rng.choice(len(Fte), min(len(Fte), 2000), replace=False)
Z_pca = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(Fte[sub]))
Z_tsne = TSNE(2, random_state=SEED, perplexity=min(30, len(sub) - 1), init="pca").fit_transform(Fte[sub])
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, Z, name in [(ax[0], Z_pca, "PCA"), (ax[1], Z_tsne, "t-SNE")]:
    for k, c in enumerate(CLASSES):
        m = yte[sub] == k
        a.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.6, label=c)
    a.set_title(f"DINOv2 features of clean test tiles, {name}"); a.legend(markerscale=3)
plt.tight_layout(); plt.savefig(P.figures / "06_dinov2_projection.png", dpi=130); plt.show()

# %% [markdown]
# _Describe: which classes form their own cluster, which overlap (e.g. CR vs LP — both are dark lines)?_
#
# ### Headline table, RIAWELC columns (clean split)

# %%
res = pd.read_csv(P.results / "all_results.csv")
res = res[res.eval_set == "RIAWELC test (clean)"]
def pick(model_prefix, seed=None):
    r = res[res.model.str.startswith(model_prefix)]
    if seed is not None and "seed" in r:
        r = r[r.seed == seed]
    return r.iloc[-1] if len(r) else None
classical = res[res.lab == 4].sort_values("macro_f1").iloc[-1]
rows = [("SVM / Random Forest (best: " + classical.model + ")", classical),
        ("ResNet50", pick("ResNet50", SEED)), ("ViT fine-tuned", pick("ViT")), ("DINOv2 + linear", pick("DINOv2"))]
headline = pd.DataFrame([{"Model": n, "RIAWELC test accuracy": r.accuracy if r is not None else np.nan,
                          "RIAWELC macro-F1": r.macro_f1 if r is not None else np.nan,
                          "GDXray macro-F1 (Lab 7)": np.nan} for n, r in rows])
headline.to_csv(P.results / "headline_table.csv", index=False)
headline.round(4)

# %% [markdown]
# ## Notes — fill in from your outputs
# - **Did:** fine-tuned ViT-S/16 (lr 3e-5) and trained a logistic regression on frozen DINOv2-S features, same clean split and seed as Lab 5.
# - **Found:** _which of the four models is best on RIAWELC; how close DINOv2+linear gets with ~1,500 trained parameters._
# - **Surprised me:** _…_
