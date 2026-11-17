# %% [markdown]
# # Lab 7 — Zero-shot generalisation on GDXray
# **Concepts:** 6 domain shift · 12 zero-shot transfer · 13 data augmentation. **Runtime: T4 GPU.**
# Start only when Labs 4–6 are final: from here on, nothing about the RIAWELC models may change because of GDXray.
#
# **Zero-shot transfer** = run the RIAWELC-trained models on GDXray with no GDXray training at all.
#
# **Comparing like with like.** GDXray labels say *defect* or *no defect* (Lab 3), so every model is scored as a
# defect detector: predicted CR, PO or LP → DEFECT. To separate "the domain changed" from "the task changed", we also
# score the RIAWELC clean test split the same binary way. The drop that matters is RIAWELC-binary → GDXray-binary.

# %% SETUP

# %%
if IN_COLAB:
    !pip -q install timm
import joblib, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from weld_insp.data import get_riawelc, make_loader
from weld_insp.experiment import Experiment
from weld_insp.features import extract_all
from weld_insp.train import extract_backbone_features, plot_history
from weld_insp.metrics import summary, binary_summary, to_binary, defect_score, log_result, plot_confusion
from weld_insp.config import CLASSES, CLASS_TO_IDX
RIA = get_riawelc(P)
EX = Experiment(P, RIA, epochs=2 if QUICK else 10, batch_size=16 if QUICK else 32, quick=QUICK)
VIT = "vit_small_patch16_224.augreg_in21k_ft_in1k"
DINO = "vit_small_patch14_dinov2.lvd142m"

gd = pd.read_csv(P.results / "gdxray_labels.csv")
GDIR = P.gdxray_crops
y_gd = (gd.label == "DEFECT").astype(int).values
unseen = ~gd.seen_in_training.astype(bool).values
test = EX.splits("clean")["test"]
y_ria = to_binary(test.label.map(CLASS_TO_IDX).values)
print(f"GDXray crops: {len(gd)} ({y_gd.sum()} DEFECT, {(1 - y_gd).sum()} ND); unseen radiographs: {unseen.sum()} crops")
print(f"RIAWELC clean test: {len(test)} tiles")

# %% [markdown]
# ## Task 1 — run every final model on GDXray, unchanged

# %%
scores = {}   # model -> dict(ria=(pred4, score), gd=(pred4, score))
# Classical: features from the same extractor as Lab 4
F_gd = extract_all([GDIR / p for p in gd.path], workers=os.cpu_count())
table = pd.read_csv(P.results / "image_table.csv")
X_all = np.load(P.results / "features.npy")
row_of = pd.Series(np.arange(len(table)), index=table.path)
F_ria = X_all[row_of[test.path].values]
for name, f in [("SVM", "classical_svm.joblib"), ("Random Forest", "classical_rf.joblib")]:
    m = joblib.load(P.checkpoints / f)
    out = {}
    for k, F in [("ria", F_ria), ("gd", F_gd)]:
        p4 = m.predict(F)
        sc = defect_score(m.predict_proba(F)) if name != "SVM" else to_binary(p4).astype(float)
        out[k] = (p4, sc)
    scores[name] = out

# %%
def deep_scores(run_dir, arch):
    out = {}
    for k, df, root in [("ria", test, RIA), ("gd", gd, GDIR)]:
        _, prob, p4 = EX.evaluate(run_dir, arch=arch, df=df, image_root=root)
        out[k] = (p4, defect_score(prob))
    return out
scores["ResNet50"] = deep_scores(EX.run_dir("resnet50", "clean", SEED), "resnet50")
scores["ViT fine-tuned"] = deep_scores(EX.run_dir(VIT, "clean", SEED), VIT)

dino = timm.create_model(DINO, pretrained=EX.pretrained, num_classes=0, img_size=224)
lin = joblib.load(P.checkpoints / "dinov2_linear.joblib")
out = {}
for k, df, root in [("ria", test, RIA), ("gd", gd, GDIR)]:
    F, _ = extract_backbone_features(dino, make_loader(df, root, train=False, batch_size=64), max_batches=EX.max_batches)
    pr = lin.predict_proba(F)
    out[k] = (pr.argmax(1), defect_score(pr))
scores["DINOv2 + linear"] = out

# %%
def row(name, k, y, mask=None):
    p4, sc = scores[name][k]
    n = min(len(p4), len(y))                       # QUICK mode scores only a few batches
    p4, sc, y = p4[:n], sc[:n], y[:n]
    if mask is not None:
        mask = mask[:n]; p4, sc, y = p4[mask], sc[mask], y[mask]
    # decision = the model's own argmax class, collapsed to DEFECT / ND
    s2 = summary(y, to_binary(p4), classes=["ND", "DEFECT"])
    # ROC-AUC needs a continuous score; the SVM was trained without probabilities, so none for it
    auc = binary_summary(y, sc)["roc_auc"] if name != "SVM" else np.nan
    return {"accuracy": s2["accuracy"], "macro_f1": s2["macro_f1"], "recall_DEFECT": s2.get("recall_DEFECT"),
            "recall_ND": s2.get("recall_ND"), "roc_auc": auc, "n": s2["n"]}

rows = []
for name in scores:
    for k, label, y, mask in [("ria", "RIAWELC clean test (binary)", y_ria, None),
                              ("gd", "GDXray all crops", y_gd, None),
                              ("gd", "GDXray unseen radiographs", y_gd, unseen)]:
        if mask is not None and mask.sum() == 0:
            continue
        r = {"model": name, "eval_set": label, **row(name, k, y, mask)}
        rows.append(r)
        log_result(P.results, {"lab": 7, "model": name, "train_split": "clean", **{k2: v for k2, v in r.items() if k2 != "model"}})
gen = pd.DataFrame(rows)
gen.to_csv(P.results / "gdxray_generalisation.csv", index=False)
gen.round(4)

# %%
pd.DataFrame({name: scores[name]["gd"][1] for name in scores if len(scores[name]["gd"][1]) == len(gd)}).assign(
    path=gd.path, label=gd.label, seen=gd.seen_in_training).to_csv(P.results / "gdxray_scores.csv", index=False)

head = pd.read_csv(P.results / "headline_table.csv")
def gd_f1(model, subset):
    r = gen[(gen.model == model) & (gen.eval_set == subset)]
    return r.macro_f1.iloc[0] if len(r) else np.nan
for i, m in head.Model.items():
    mm = ("Random Forest" if "Random Forest" in m else "SVM") if m.startswith("SVM") else m
    head.loc[i, "RIAWELC binary macro-F1"] = gd_f1(mm, "RIAWELC clean test (binary)")
    head.loc[i, "GDXray macro-F1 (Lab 7)"] = gd_f1(mm, "GDXray all crops")
    head.loc[i, "GDXray macro-F1, unseen radiographs"] = gd_f1(mm, "GDXray unseen radiographs")
head.to_csv(P.results / "headline_table.csv", index=False)
head.round(4)

# %%
plot = gen.pivot(index="model", columns="eval_set", values="macro_f1")
ax = plot.plot.bar(figsize=(9, 4), rot=0)
ax.axhline(0.64, ls="--", c="grey", lw=1, label="WeldVGG 2025: ≈0.64 on GDXray")
ax.set_ylabel("macro-F1 (defect vs no defect)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="lower left")
ax.set_title("Same models, RIAWELC vs GDXray")
plt.tight_layout(); plt.savefig(P.figures / "07_riawelc_vs_gdxray.png", dpi=130); plt.show()

# %% [markdown]
# ## Task 2 — what breaks?
# Rows = true GDXray label, columns = what the 4-class model predicted. A defect called ND is a missed defect.

# %%
gd_rank = gen[gen.eval_set == "GDXray all crops"].set_index("model").macro_f1.sort_values()
best, worst = gd_rank.index[-1], gd_rank.index[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, name in zip(ax, [best, worst]):
    p4 = scores[name]["gd"][0]; n = len(p4)
    ct = pd.crosstab(pd.Series(np.where(y_gd[:n] == 1, "DEFECT", "ND"), name="true"),
                     pd.Series([CLASSES[i] for i in p4], name="predicted")).reindex(columns=CLASSES, fill_value=0)
    a.imshow(ct.div(ct.sum(1), axis=0).values, cmap="Blues", vmin=0, vmax=1)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            a.text(j, i, f"{ct.iloc[i, j] / ct.iloc[i].sum():.2f}\n({ct.iloc[i, j]})", ha="center", va="center", fontsize=8)
    a.set_xticks(range(4), CLASSES); a.set_yticks(range(ct.shape[0]), ct.index)
    a.set_title(f"{name} on GDXray ({'best' if name == best else 'worst'})", fontsize=9)
plt.tight_layout(); plt.savefig(P.figures / "07_gdxray_confusion_best_worst.png", dpi=130); plt.show()

# %%
p4 = scores[best]["gd"][0]; n = len(p4)
wrong = np.where(to_binary(p4) != y_gd[:n])[0]
pick = np.random.default_rng(SEED).choice(wrong, min(12, len(wrong)), replace=False) if len(wrong) else []
fig, ax = plt.subplots(2, 6, figsize=(13, 5))
for a in ax.flat: a.axis("off")
for a, i in zip(ax.flat, pick):
    a.imshow(plt.imread(GDIR / gd.path[i]), cmap="gray")
    a.set_title(f"true {gd.label[i]} → {CLASSES[p4[i]]}\n{gd.source[i]}", fontsize=7)
plt.suptitle(f"12 GDXray crops {best} got wrong"); plt.tight_layout()
plt.savefig(P.figures / "07_gdxray_misclassified.png", dpi=130); plt.show()

# %% [markdown]
# **What differs from RIAWELC** (write after looking): contrast, noise, scale of the defect in the crop, background,
# defect size relative to the window, annotation quality (is the "defect" visible at all?).
#
# **Against WeldVGG (2025), macro-F1 ≈ 0.64 zero-shot on GDXray:** _state whether you reproduce the drop, and how
# your "unseen radiographs" number compares with "all crops"._

# %% [markdown]
# ## Task 3 — does augmentation close the gap?
# Retrain the better deep model **chosen on RIAWELC validation** (not on GDXray, which would be tuning on the test)
# with Gaussian noise, blur, brightness/contrast jitter and flips (`build_transforms(augment=True)`), same seed.

# %%
res = pd.read_csv(P.results / "all_results.csv")
cand = {"ResNet50": ("resnet50", 1e-4), "ViT fine-tuned": (VIT, 3e-5)}
val_f1 = {}
for name, (arch, _) in cand.items():
    h = pd.read_csv(EX.run_dir(arch, "clean", SEED) / "history.csv")
    val_f1[name] = h.val_macro_f1.max()
pick_name = max(val_f1, key=val_f1.get)
arch, lr = cand[pick_name]
print("validation macro-F1:", val_f1, "-> augmenting", pick_name)
hist_a, chosen_a, RUN_A = EX.train(arch, "clean", seed=SEED, lr=lr, augment=True, tag="_aug")
plot_history(hist_a, chosen_a, title=f"{pick_name} + augmentation", save=P.figures / "07_aug_curves.png"); plt.show()

# %%
scores[pick_name + " + aug"] = deep_scores(RUN_A, arch)
cmp = []
for name in [pick_name, pick_name + " + aug"]:
    for k, label, y, mask in [("ria", "RIAWELC clean test (binary)", y_ria, None), ("gd", "GDXray all crops", y_gd, None),
                              ("gd", "GDXray unseen radiographs", y_gd, unseen)]:
        if mask is not None and mask.sum() == 0:
            continue
        cmp.append({"model": name, "eval_set": label, **row(name, k, y, mask)})
y4 = test.label.map(CLASS_TO_IDX).values
p_aug = scores[pick_name + " + aug"]["ria"][0]
s4 = summary(y4[:len(p_aug)], p_aug)
log_result(P.results, {"lab": 7, "model": pick_name + " + aug", "seed": SEED, "train_split": "clean",
                       "eval_set": "RIAWELC test (clean)", **s4})
for r in cmp[-3:]:
    log_result(P.results, {"lab": 7, "train_split": "clean", **r})
cmp = pd.DataFrame(cmp)
cmp.to_csv(P.results / "augmentation_effect.csv", index=False)
delta = cmp.pivot(index="eval_set", columns="model", values="macro_f1")
delta["change"] = delta[pick_name + " + aug"] - delta[pick_name]
print("4-class RIAWELC macro-F1 with augmentation:", s4["macro_f1"])
delta.round(4)

# %% [markdown]
# **Did augmentation help?** _Three sentences: the change on RIAWELC, the change on GDXray, and what that says about the
# cause of the gap. If noise/contrast augmentation does not move GDXray, the gap is probably not about contrast or
# noise alone (think scale, labelling rule, defect size, or a few plates dominating training)._ A result of "nothing
# closed the gap" is a valid finding if measured honestly.
#
# ## Notes — fill in from your outputs
# - **Did:** ran all final RIAWELC models zero-shot on GDXray crops, scored as defect vs no defect on both datasets; retrained the best deep model with augmentation.
# - **Found:** _RIAWELC-binary vs GDXray macro-F1 per model; seen vs unseen radiographs; which class the models collapse to._
# - **Surprised me:** _…_
