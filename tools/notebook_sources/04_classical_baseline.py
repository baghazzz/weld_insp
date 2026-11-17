# %% [markdown]
# # Lab 4 — Classical ML baseline
# **Concepts:** 7 classical features and SVM / Random Forest · 8 accuracy, precision, recall, F1, macro-F1, confusion matrix. Runtime: CPU.
#
# Before deep learning, describe each tile with numbers a person designed, then fit simple classifiers. This is
# the cheap baseline every later model must beat.
#
# | Family | What it measures | Dims |
# |---|---|---|
# | HOG | direction of edges in 16×16-pixel cells (cracks and LP are oriented edges) | 1,764 |
# | LBP | micro-texture: each pixel vs its 8 or 16 neighbours, as a histogram | 28 |
# | GLCM | how often grey levels co-occur at distance 1, 3, 5: contrast, homogeneity, energy, correlation | 12 |
# | Intensity | 32-bin brightness histogram + mean, std, 5/50/95th percentile | 37 |
#
# **Metrics.** *Precision* of a class = of the tiles we called CR, how many were CR. *Recall* = of the real CR tiles,
# how many we found (for a defect: 1 − missed-defect rate). *F1* = harmonic mean of the two. *Macro-F1* = plain
# average of the per-class F1s, so the rare LP class counts as much as CR. The *confusion matrix* shows which
# class gets mistaken for which.

# %% SETUP

# %%
import time, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from weld_insp.data import get_riawelc
from weld_insp.features import extract_all, FEATURE_FAMILIES
from weld_insp.metrics import summary, classification_report_df, plot_confusion, log_result
from weld_insp.config import CLASS_TO_IDX

RIA = get_riawelc(P)
table = pd.read_csv(P.results / "image_table.csv")         # all 24,407 rows (original split)
clean = pd.read_csv(P.results / "clean_split.csv")         # 21,964 unique rows (leakage-free split)
print(len(table), len(clean))

# %% [markdown]
# ## Task 1 — features
# Extracted once for all 24,407 tiles (≈ 10 min on 2 CPU cores) and saved; the row order matches `image_table.csv`.

# %%
feat_f = P.results / "features.npy"
if feat_f.exists():
    X_all = np.load(feat_f)
else:
    t0 = time.time()
    X_all = extract_all([RIA / p for p in table.path], workers=os.cpu_count())
    np.save(feat_f, X_all)
    print(f"extracted in {time.time() - t0:.0f}s")
print("feature matrix", X_all.shape, "| families:", {k: v.stop - v.start for k, v in FEATURE_FAMILIES.items()})
row_of = pd.Series(np.arange(len(table)), index=table.path)
N_PCA = 32 if QUICK else 256

def xy(df, split):
    d = df[df.split == split]
    if QUICK:
        d = d.sample(frac=1, random_state=SEED).groupby("label").head(60)
    return X_all[row_of[d.path].values], d.label.map(CLASS_TO_IDX).values

Xtr, ytr = xy(clean, "train"); Xva, yva = xy(clean, "val"); Xte, yte = xy(clean, "test")
print("clean split sizes:", len(ytr), len(yva), len(yte))

# %% [markdown]
# **Model selection on validation only.** Each candidate setting is trained on *train* and scored on *val*
# (macro-F1); the test split is used once, at the end. The scaler (and PCA) are fitted on training data only,
# inside a pipeline, so no information from val/test leaks into the preprocessing.
#
# For the SVM, the 1,841 standardised features are first reduced to 256 principal components. An RBF-SVM on
# 15k × 1,841 is slow (its cost grows with the square of the sample count times the dimension); PCA keeps
# about 80% of the variance (printed below) and makes each fit take seconds. The Random Forest uses all features, because we want
# its importances per feature family.

# %%
def fit_eval(model, name):
    t0 = time.time(); model.fit(Xtr, ytr); fit_s = time.time() - t0
    s = summary(yva, model.predict(Xva))
    print(f"{name:28s} val macro-F1 {s['macro_f1']:.4f}  acc {s['accuracy']:.4f}  ({fit_s:.0f}s)")
    return model, s["macro_f1"]

svm_grid = {}
for C in [1, 10, 100]:
    svm_grid[C] = fit_eval(make_pipeline(StandardScaler(), PCA(N_PCA, random_state=SEED),
                                         SVC(kernel="rbf", C=C, gamma="scale")), f"SVM RBF C={C}")
best_C = max(svm_grid, key=lambda c: svm_grid[c][1]); svm = svm_grid[best_C][0]

rf_grid = {}
for n in [100, 300, 500]:
    rf_grid[n] = fit_eval(make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=n, max_features="sqrt", n_jobs=-1, random_state=SEED)), f"Random Forest {n} trees")
best_n = max(rf_grid, key=lambda n: rf_grid[n][1]); rf = rf_grid[best_n][0]
print(f"\nChosen on validation: SVM C={best_C}, Random Forest {best_n} trees")
pca = svm.named_steps["pca"]
print(f"PCA({N_PCA}) keeps {pca.explained_variance_ratio_.sum():.1%} of the variance")

# %% [markdown]
# ## Task 2 — results on the clean test split (touched once)

# %%
models = {
    f"SVM RBF (C={best_C})": svm,
    f"Random Forest ({best_n} trees)": rf,
    "Gaussian NB (equal priors)": make_pipeline(StandardScaler(), GaussianNB(priors=[0.25] * 4)).fit(Xtr, ytr),
    "Gaussian NB (priors from train)": make_pipeline(StandardScaler(), GaussianNB()).fit(Xtr, ytr),
}
rows, preds = [], {}
for name, m in models.items():
    p_tr, p_te = m.predict(Xtr), m.predict(Xte)
    preds[name] = p_te
    s = summary(yte, p_te)
    rows.append({"model": name, "split": "clean", "train_acc": round(summary(ytr, p_tr)["accuracy"], 4),
                 "test_acc": s["accuracy"], "test_macro_f1": s["macro_f1"],
                 **{k: v for k, v in s.items() if k.startswith(("recall_", "f1_"))}})
    log_result(P.results, {"lab": 4, "model": name, "train_split": "clean", "eval_set": "RIAWELC test (clean)", **s})
baseline = pd.DataFrame(rows)
baseline

# %%
for name in list(models)[:2]:
    print(f"\n{name}"); display(classification_report_df(yte, preds[name]))

# %%
best_name = baseline.loc[baseline.test_macro_f1.idxmax(), "model"]
plot_confusion(yte, preds[best_name], title=f"{best_name}, clean test split",
               save=P.figures / "04_confusion_best_classical.png")
plt.show()

# %% [markdown]
# **Most-missed class: CR (crack), recall 0.43 for the SVM; 34% of cracks are called LP.** Lab 1 showed why: both a
# crack and lack of penetration appear as a dark line, and many cracks are faint. HOG sees "an oriented dark edge" in
# both; what separates them (LP is straight, horizontal and spans the tile; a crack is thin, kinked, any direction) is
# shape over the whole tile, which cell-level descriptors capture poorly. ND is easiest (recall 0.74), consistent
# with its distinct brightness range (Lab 1).
#
# Both SVM and RF reach **100% training accuracy** but ~52–61% on test plates: they fit the training plates, not
# the defect types.

# %% [markdown]
# ### The same models on the original (leaky) split
# Trained on the official training folder and tested on the official test folder, which Lab 2 showed is
# made of exact copies of training tiles.

# %%
Xtr_o, ytr_o = xy(table, "train"); Xte_o, yte_o = xy(table, "test")
for name, m in list(models.items())[:2]:
    m_o = clone(m).fit(Xtr_o, ytr_o)                              # same settings, untrained copy
    s = summary(yte_o, m_o.predict(Xte_o))
    rows.append({"model": name, "split": "original", "train_acc": round(summary(ytr_o, m_o.predict(Xtr_o))["accuracy"], 4),
                 "test_acc": s["accuracy"], "test_macro_f1": s["macro_f1"],
                 **{k: v for k, v in s.items() if k.startswith(("recall_", "f1_"))}})
    log_result(P.results, {"lab": 4, "model": name, "train_split": "original", "eval_set": "RIAWELC test (original)", **s})
baseline = pd.DataFrame(rows)
baseline.to_csv(P.results / "baseline_metrics.csv", index=False)
baseline[["model", "split", "train_acc", "test_acc", "test_macro_f1"]]

# %% [markdown]
# ## Task 3 — One-vs-One vs One-vs-Rest, and feature importance
# scikit-learn's `SVC` is One-vs-One inside: one SVM per pair of classes (6 for 4 classes), then a vote.
# One-vs-Rest trains one SVM per class against all others (4 SVMs).

# %%
ovr = make_pipeline(StandardScaler(), PCA(N_PCA, random_state=SEED),
                    OneVsRestClassifier(SVC(kernel="rbf", C=best_C, gamma="scale"), n_jobs=-1)).fit(Xtr, ytr)
ovo_s, ovr_s = summary(yte, svm.predict(Xte)), summary(yte, ovr.predict(Xte))
ovo_ovr = pd.DataFrame([{"strategy": "One-vs-One (6 SVMs)", **ovo_s}, {"strategy": "One-vs-Rest (4 SVMs)", **ovr_s}])
ovo_ovr.to_csv(P.results / "svm_ovo_vs_ovr.csv", index=False)
ovo_ovr[["strategy", "accuracy", "macro_f1"] + [c for c in ovo_ovr if c.startswith("recall_")]]

# %%
imp = rf.named_steps["randomforestclassifier"].feature_importances_
fam = pd.DataFrame([{"family": k, "n_features": v.stop - v.start, "total_importance": imp[v].sum(),
                     "mean_per_feature": imp[v].mean()} for k, v in FEATURE_FAMILIES.items()])
fam["total_importance"] = fam.total_importance.round(3); fam["mean_per_feature"] = fam.mean_per_feature.round(5)
fam.to_csv(P.results / "rf_feature_family_importance.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].bar(fam.family, fam.total_importance); ax[0].set_title("Total importance (sums to 1)")
ax[1].bar(fam.family, fam.mean_per_feature, color="tab:orange"); ax[1].set_title("Importance per single feature")
plt.tight_layout(); plt.savefig(P.figures / "04_rf_feature_importance.png", dpi=130); plt.show()
fam

# %% [markdown]
# **One-vs-One vs One-vs-Rest:** within one point of each other (One-vs-Rest slightly ahead), so the multi-class
# strategy is not what limits the SVM.
#
# **Feature families:** HOG gets half of the total importance only because it has 1,764 of the 1,841 features. Per
# single feature, **GLCM** (texture co-occurrence) and **LBP** (micro-texture) matter most, roughly 30–40× more than a
# single HOG cell. Texture statistics carry most of the signal per number.

# %%
joblib.dump(svm, P.checkpoints / "classical_svm.joblib")
joblib.dump(rf, P.checkpoints / "classical_rf.joblib")
print("saved models for Lab 7")

# %% [markdown]
# ## Notes
# - **Did:** extracted HOG, LBP, GLCM and intensity features (1,841 per tile), chose SVM C and RF trees on validation, and compared SVM, RF and two Gaussian NB variants on the clean split and on the original split.
# - **Found:** the best classical model (SVM RBF, C=10) scores **61.3% accuracy, 0.612 macro-F1** on unseen plates. The same models score **100.0%** on the official test split, because that split is copies of training tiles.
# - **Found:** cracks are missed most (recall 0.43), mostly confused with lack of penetration; ND is easiest. Per feature, texture (GLCM, LBP) beats HOG.
# - **Surprised me:** a classical SVM "matches" the published deep-learning accuracy on the official split, which is a direct demonstration that the 99% figures measure leakage, not model quality.
