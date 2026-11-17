# %% [markdown]
# # Lab 9 — Grad-CAM, packaging and write-up
# **Concepts:** 16 Grad-CAM · 17 reproducible packaging.
#
# **Grad-CAM.** Take the last convolutional feature maps (7×7 for ResNet50). Weight each map by how much the predicted
# class score would change if that map grew (the gradient, averaged over the map), sum, keep the positive part,
# upsample to 224×224. Bright = regions that pushed the model towards its answer. A correct prediction with a heatmap
# away from the defect means the model used a shortcut (e.g. background brightness).

# %% SETUP

# %%
if IN_COLAB:
    !pip -q install timm grad-cam gradio
import shutil
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from weld_insp.data import get_riawelc, build_transforms
from weld_insp.experiment import Experiment
from weld_insp.metrics import to_binary
from weld_insp.config import CLASSES, CLASS_TO_IDX
from weld_insp import gdxray as G
from weld_insp.report import md_table, fill_block, pct
RIA = get_riawelc(P)
EX = Experiment(P, RIA, batch_size=16 if QUICK else 32, quick=QUICK)
VIT = "vit_small_patch16_224.augreg_in21k_ft_in1k"
tf = build_transforms(train=False)

# %% [markdown]
# ## Task 1 — heatmaps

# %%
resnet = EX.load(EX.run_dir("resnet50", "clean", SEED), "resnet50")
cam_r = GradCAM(model=resnet, target_layers=[resnet.layer4[-1]])

def vit_reshape(t, h=14, w=14):          # ViT tokens -> (B, C, 14, 14) maps, CLS token dropped
    return t[:, 1:, :].reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)

def run_cam(cam, model, path):
    x = tf(Image.open(path).convert("L")).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0].cpu().numpy()
    pred = int(prob.argmax())
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
    base = np.asarray(Image.open(path).convert("L").resize((224, 224))).astype(np.float32) / 255
    return base, heat, pred, prob

test = EX.splits("clean")["test"].reset_index(drop=True)
_, prob_t, pred_t = EX.evaluate(EX.run_dir("resnet50", "clean", SEED), df=test)
y_t = test.label.map(CLASS_TO_IDX).values[:len(pred_t)]
gd = pd.read_csv(P.results / "gdxray_labels.csv")
_, prob_g, pred_g = EX.evaluate(EX.run_dir("resnet50", "clean", SEED), df=gd, image_root=P.gdxray_crops)
y_g = (gd.label == "DEFECT").astype(int).values[:len(pred_g)]

def pick(ok_mask, n=6):
    idx = np.where(ok_mask)[0]
    return np.random.default_rng(SEED).choice(idx, min(n, len(idx)), replace=False) if len(idx) else []

def panel(rows, title, fname):
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)), squeeze=False)
    for k, (path, true) in enumerate(rows):
        base, heat, pred, prob = run_cam(cam_r, resnet, path)
        ax[k, 0].imshow(base, cmap="gray"); ax[k, 0].set_title(f"true {true}", fontsize=8)
        ax[k, 1].imshow(show_cam_on_image(np.stack([base] * 3, -1), heat, use_rgb=True)); ax[k, 1].set_title("Grad-CAM", fontsize=8)
        ax[k, 2].barh(CLASSES, prob); ax[k, 2].set_xlim(0, 1); ax[k, 2].set_title(f"pred {CLASSES[pred]}", fontsize=8)
        ax[k, 0].axis("off"); ax[k, 1].axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(P.figures / fname, dpi=110); plt.show()

ria_ok, ria_bad = pick(pred_t == y_t), pick(pred_t != y_t)
panel([(RIA / test.path[i], test.label[i]) for i in ria_ok], "RIAWELC, correctly classified", "09_gradcam_riawelc_correct.png")
panel([(RIA / test.path[i], test.label[i]) for i in ria_bad], "RIAWELC, misclassified", "09_gradcam_riawelc_wrong.png")

# %%
gd_ok, gd_bad = pick(to_binary(pred_g) == y_g), pick(to_binary(pred_g) != y_g)
panel([(P.gdxray_crops / gd.path[i], gd.label[i]) for i in gd_ok], "GDXray, correct (defect vs ND)", "09_gradcam_gdxray_correct.png")
panel([(P.gdxray_crops / gd.path[i], gd.label[i]) for i in gd_bad], "GDXray, wrong (defect vs ND)", "09_gradcam_gdxray_wrong.png")

# %% [markdown]
# ### Does the heatmap land on the defect? A measured check on GDXray
# GDXray has pixel masks, so for every DEFECT crop the model calls a defect we can test the **pointing game**: is the
# hottest heatmap pixel inside the annotated defect (grown by 8 px for tolerance)?

# %%
GD = G.download_welds(P)                       # re-extracts from data/gdxray_welds.zip if needed
masks = G.load_masks(GD)
hits, n = 0, 0
for i in np.where((y_g == 1) & (to_binary(pred_g) == 1))[0][: (20 if QUICK else 300)]:
    r = gd.iloc[i]
    m = masks[G._index(r.source)]
    w = int(r.window)
    crop_m = np.asarray(Image.fromarray(m[r.y:r.y + w, r.x:r.x + w].astype(np.uint8) * 255).resize((224, 224))) > 0
    import cv2
    crop_m = cv2.dilate(crop_m.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    _, heat, _, _ = run_cam(cam_r, resnet, P.gdxray_crops / r.path)
    yx = np.unravel_index(heat.argmax(), heat.shape)
    hits += bool(crop_m[yx]); n += 1
pointing = hits / max(n, 1)
print(f"Pointing game on GDXray defect crops detected as defects: {hits}/{n} = {pointing:.1%} hit the annotated defect")
pd.DataFrame([{"model": "ResNet50", "crops": n, "hits": hits, "hit_rate": pointing}]).to_csv(P.results / "gradcam_pointing_game.csv", index=False)

# %% [markdown]
# **Per class (write after looking):** CR — does the heat follow the line? PO — is it centred on the pore? LP — along
# the band? ND — diffuse? Give one example that scored well but looked at the wrong region (e.g. tile edge or a
# bright background patch), and relate it to the ND brightness difference found in Lab 1.
#
# Optional ViT heatmaps: `GradCAM(model=vit, target_layers=[vit.blocks[-1].norm1], reshape_transform=vit_reshape)`.

# %% [markdown]
# ## Task 2 — repository, README, demo
# The project folder already has the repo layout: `src/weld_insp/` (dataset class, metrics and everything reused),
# `notebooks/`, `app/`, `docs/`, `results/` and a `.gitignore` that excludes `data/`, `*.pth`, `*.pt`, `*.zip` and
# checkpoints. GDXray images must never be pushed (licence: no redistribution).
#
# The cells below write the real numbers into `README.md` and `docs/summary.md`.

# %%
head = pd.read_csv(P.results / "headline_table.csv")
leak_o = pd.read_csv(P.results / "leakage_table_original.csv")
leak_c = pd.read_csv(P.results / "leakage_table_clean.csv")
ops = pd.read_csv(P.results / "operating_points.csv")
auc = pd.read_csv(P.results / "anomaly_auc.csv")
leak = pd.concat([leak_o, leak_c])
leak = leak[leak.threshold_bits.isin([0, 6])].pivot(index="comparison", columns="threshold_bits", values="pct_within")
leak.columns = [f"% within {c} bits" for c in leak.columns]
leak = leak.reset_index()
op20 = ops[(ops.cost_ratio == 20) & (ops.eval_set != "RIAWELC val")][["model", "eval_set", "threshold_from_val", "missed_defect_rate", "false_reject_rate", "cost_per_1000"]]
for f in [P.root / "README.md", P.root / "docs" / "summary.md"]:
    fill_block(f, "headline", md_table(head))
    fill_block(f, "leakage", md_table(leak, "{:.2f}"))
    fill_block(f, "operating", md_table(op20))
    fill_block(f, "anomaly", md_table(auc.pivot(index="model", columns="eval_set", values="roc_auc").reset_index()))
print((P.root / "README.md").read_text()[:3000])

# %%
# A few RIAWELC example tiles for the demo (RIAWELC is freely released; GDXray is not, so none from it)
ex_dir = P.root / "app" / "examples"; ex_dir.mkdir(parents=True, exist_ok=True)
for c in CLASSES:
    r = test[test.label == c].iloc[0]
    shutil.copy(RIA / r.path, ex_dir / f"{c}_{Path(r.path).name.replace(' ', '_')}")
sys.path.insert(0, str(P.root / "app"))
from app import build_demo
demo = build_demo(EX.run_dir("resnet50", "clean", SEED) / "best.pth")
if not QUICK:
    demo.launch(share=IN_COLAB)          # in Colab: a public *.gradio.live link valid for 72 h

# %% [markdown]
# **Push to GitHub** (from your PC, in `D:\weld_insp`, after downloading the results from Drive):
# ```
# git init -b main
# git add .
# git commit -m "weld_insp: leakage-free weld defect classification, GDXray generalisation, PatchCore operating points"
# gh repo create weld_insp --public --source . --push      # or create it on github.com and `git remote add origin ...`
# ```
# `git status` must show no `data/`, `checkpoints/`, `.pth` or `.zip` files before you push.

# %% [markdown]
# ## Task 3 — summary and resume bullets from real numbers

# %%
def get(model_prefix, col):
    r = head[head.Model.str.startswith(model_prefix)]
    return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan
best_ria = head.loc[head["RIAWELC macro-F1"].idxmax()]
best_gd = head.loc[head["GDXray macro-F1 (Lab 7)"].idxmax()] if head["GDXray macro-F1 (Lab 7)"].notna().any() else best_ria
v20 = ops[(ops.cost_ratio == 20) & (ops.eval_set == "RIAWELC val")]
rec = v20.loc[v20.cost_per_1000.idxmin(), "model"]              # chosen on validation, never on test
op = ops[(ops.cost_ratio == 20) & (ops.eval_set == "RIAWELC test") & (ops.model == rec)].iloc[0]
pc_auc = auc[(auc.model == "PatchCore") & (auc.eval_set == "GDXray")].roc_auc
bullets = f"""- Audited a 24,407-image weld X-ray benchmark (RIAWELC) and found its official test set is 100% byte-identical copies of training images; rebuilt a leakage-free split by source radiograph (29 plates).
- Compared classical ML (HOG/LBP/GLCM + SVM/RF), ResNet50, ViT and DINOv2 on the clean split: best macro-F1 {best_ria['RIAWELC macro-F1']:.2f} ({best_ria.Model}).
- Tested zero-shot generalisation on GDXray radiographs: best defect-vs-no-defect macro-F1 {best_gd['GDXray macro-F1 (Lab 7)']:.2f} ({best_gd.Model}){'' if pc_auc.empty else f'; PatchCore anomaly detection trained on good welds only reached ROC-AUC {pc_auc.iloc[0]:.2f}'}.
- Set a cost-based operating point (miss = 20× rework): {op.model} misses {pct(op.missed_defect_rate)} of defects at a {pct(op.false_reject_rate)} false-reject rate on held-out plates.
"""
(P.root / "docs").mkdir(exist_ok=True)
(P.root / "docs" / "resume_bullets.md").write_text("# Resume bullets (generated from results/*.csv)\n\n" + bullets)
fill_block(P.root / "docs" / "summary.md", "bullets", bullets)
print(bullets)

# %% [markdown]
# Now edit the prose parts of `docs/summary.md` (question, findings, limitations, next step) and keep it to two pages.
# Every number in it should already be in a table filled above.
#
# ## Notes — fill in from your outputs
# - **Did:** Grad-CAM on correct and wrong tiles from both datasets, a pointing-game score on GDXray, README and summary filled from the results CSVs, Gradio demo.
# - **Found:** _how often the heatmap hits the defect; any shortcut seen._
# - **Surprised me:** _…_
