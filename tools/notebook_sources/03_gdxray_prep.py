# %% [markdown]
# # Lab 3 — GDXray preparation
# **Concept:** 6 domain shift and generalisation. GDXray is the "new X-ray data" for Lab 7. Prepare it now, then
# leave it untouched (no model runs on it) until Lab 7.
#
# **Domain shift** = the test images come from a different distribution than the training images (other machine,
# film, scanner, contrast processing, crop size). A model can score 99% in its own domain and much lower outside it.
#
# **What we already know from Labs 1–2:** RIAWELC file names start with `RRT-xxR`, the BAM Round Robin Test plates.
# GDXray's Welds group was donated by the same BAM lab, and its series W0003 *is* the round-robin set. So part of
# GDXray may show the very same radiographs RIAWELC was cut from. This lab measures that overlap, because a test
# on images the model trained on is not a generalisation test.

# %% SETUP

# %%
if IN_COLAB:
    !pip -q install imagehash opencv-python-headless
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from weld_insp import gdxray as G
from weld_insp.data import get_riawelc

# %% [markdown]
# ## Task 1 — read the README, download Welds
# From github.com/computervision-xray-testing/GDXray (README and Mery et al., 2015):
# - **Licence:** free for research and education only; **redistribution and commercial use are prohibited**, and
#   results must cite Mery et al. (2015). So the GitHub repo must never contain GDXray images (`data/` is git-ignored).
# - **Welds group:** ≈ 209 MB, one zip on Dropbox; 3–4 series (W0001–W0004, 88–98 images depending on release).
# - **Storage:** one folder per series, one PNG per radiograph (`W0001_0001.png` …). W0001 holds 10 radiographs taken from
#   the BAM collection; **W0002 holds their ground truth as binary images** (white = defect pixel). W0003 holds the 67
#   round-robin radiographs without masks.
# - **Annotation meaning:** masks mark *where* a defect is, **not which type**. GDXray cannot give CR/PO/LP labels.

# %%
GD = G.download_welds(P)          # needs internet; Colab can reach Dropbox
print("GDXray series root:", GD)
for d in sorted(p for p in GD.iterdir() if p.is_dir()):
    print(f"  {d.name}: {len(list(d.glob('*.png')))} png, other files: {[f.name for f in d.iterdir() if f.suffix != '.png'][:4]}")

# %%
inv = G.inventory(GD)
inv.to_csv(P.results / "gdxray_inventory.csv", index=False)
summary_inv = inv.groupby("series").agg(radiographs=("file", "size"), min_w=("width", "min"), max_w=("width", "max"),
                                        min_h=("height", "min"), max_h=("height", "max"),
                                        modes=("mode", lambda s: ",".join(sorted(set(s)))),
                                        bit_depth=("bit_depth", lambda s: ",".join(map(str, sorted(set(s))))))
summary_inv

# %% [markdown]
# ## Task 2 — compare the two domains and fix the cropping and labelling rules

# %%
RIA = get_riawelc(P)
table = pd.read_csv(P.results / "image_table.csv")
masks = G.load_masks(GD)
first = sorted((GD / "W0001").glob("*.png"))[0]
gimg, _ = G.read_radiograph(first)
gm = masks.get(G._index(first.name))
ria_row = table[(table.label == "PO")].sample(1, random_state=SEED).iloc[0]

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax1.imshow(gimg, cmap="gray", aspect="auto")
if gm is not None and gm.shape == gimg.shape:
    ax1.contour(gm, levels=[0.5], colors="red", linewidths=0.6)
ax1.set_title(f"GDXray {first.name}: full radiograph {gimg.shape[1]}×{gimg.shape[0]} px, annotated defects in red")
ax2 = fig.add_subplot(2, 3, 4); ax2.imshow(plt.imread(RIA / ria_row.path), cmap="gray"); ax2.set_title(f"RIAWELC tile ({ria_row.label})")
ys, xs = np.nonzero(gm) if gm is not None and gm.any() else ([gimg.shape[0] // 2], [gimg.shape[1] // 2])
cy, cx = int(np.median(ys)), int(np.median(xs))
raw = gimg[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40]
ax3 = fig.add_subplot(2, 3, 5); ax3.imshow(raw, cmap="gray"); ax3.set_title("GDXray 80×80 window, raw")
ax4 = fig.add_subplot(2, 3, 6); ax4.imshow(G.clahe(gimg)[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40], cmap="gray")
ax4.set_title("same window after CLAHE (as RIAWELC)")
for a in [ax2, ax3, ax4]: a.axis("off")
plt.tight_layout(); plt.savefig(P.figures / "03_gdxray_vs_riawelc.png", dpi=130); plt.show()

# %% [markdown]
# **Three visible differences** — *write these after looking at the figure above.* Compare:
# 1. **Contrast:** RIAWELC tiles were CLAHE-enhanced; raw GDXray is not (compare panels 2 and 3).
# 2. **Noise / grain:** how grainy is the GDXray window versus the RIAWELC tile at the same magnification?
# 3. **Scale and background:** a full GDXray radiograph includes plate, markers and background; RIAWELC tiles show only
#    weld bead, magnified about 2.8× (80 px → 224 px).
#
# ### Which RIAWELC plates are in GDXray, and at what scale?
# RIAWELC tiles are 80×80 windows of the BAM scans. If GDXray stores a scan at another resolution, the same physical
# area spans a different number of pixels. For every RIAWELC plate we take 6 high-contrast tiles, shrink them to
# 60–160 px, and search each GDXray radiograph for them with **normalised cross-correlation** (template matching).
# Image and tiles are high-pass filtered first, so only grain and edges are compared, not smooth shading.
# A radiograph counts as RIAWELC plate X if X scores NCC ≥ 0.75 and beats the runner-up plate by ≥ 0.10.
# (Validated on synthetic radiographs built from RIAWELC tiles: true plates scored 0.91–0.96.)
# This takes a few minutes per radiograph.

# %%
clean = pd.read_csv(P.results / "clean_split.csv")
plate_split = clean.drop_duplicates("plate").set_index("plate").split
series = tuple(s for s in ("W0001", "W0003") if (GD / s).is_dir())
if QUICK:
    series = ("W0001",)
pm = G.find_riawelc_plates(GD, table, RIA, sizes=(60, 80, 100, 120, 160), tiles_per_plate=6, series=series)
pm.to_csv(P.results / "gdxray_plate_match_all.csv", index=False)
matches = G.summarise_plate_matches(pm, plate_split)
matches.to_csv(P.results / "gdxray_plate_match.csv")
w = matches[matches.is_riawelc_plate].window_px
WINDOW = int(w.median()) if len(w) else 80
print(f"{int(matches.is_riawelc_plate.sum())} of {len(matches)} radiographs match a RIAWELC plate; "
      f"crop window = {WINDOW} px ({'from the matches' if len(w) else 'default, no matches'})")
matches[["series", "plate", "best_ncc", "runner_up_ncc", "margin", "window_px", "is_riawelc_plate", "plate_split", "seen_in_training"]]

# %%
# Check the matches by eye: RIAWELC tile (left) and the GDXray region it was found at (right)
show = matches[matches.is_riawelc_plate].head(6)
if len(show):
    fig, ax = plt.subplots(len(show), 2, figsize=(5, 2.4 * len(show)), squeeze=False)
    for k, (rad, r) in enumerate(show.iterrows()):
        g = table[(table.plate == r.plate) & (table.label != "ND")].sample(1, random_state=SEED).iloc[0]
        tile = np.asarray(Image.open(RIA / g.path).convert("L"))
        img, _ = G.read_radiograph(GD / r.series / rad)
        x, y, ncc = G.locate(img, tile, int(r.window_px))
        s_ = int(r.window_px)
        ax[k, 0].imshow(tile, cmap="gray"); ax[k, 0].set_title(f"RIAWELC {g.file[:22]}", fontsize=7)
        ax[k, 1].imshow(G.clahe(img)[y:y + s_, x:x + s_], cmap="gray"); ax[k, 1].set_title(f"{rad} at ({x},{y}) NCC {ncc:.2f}", fontsize=7)
        ax[k, 0].axis("off"); ax[k, 1].axis("off")
    plt.tight_layout(); plt.savefig(P.figures / "03_plate_match_check.png", dpi=120); plt.show()

# %% [markdown]
# **Crop method (3 lines).**
# 1. Apply CLAHE to each annotated W0001 radiograph, as RIAWELC did.
# 2. Slide a `WINDOW`×`WINDOW` box with 50% overlap; keep only windows inside the weld band (rows where defects
#    are annotated, ± half a window), so no-defect crops show weld bead, not background.
# 3. Resize each kept window to 224×224 grayscale.
#
# **Label rule.**
# - **DEFECT:** at least 16 annotated pixels (scaled with window area) inside the central 75% of the window.
# - **ND:** no annotated pixel inside the window grown by 10 px on every side.
# - Anything in between (a defect touching the edge) is **discarded** as ambiguous.
# - ND crops are capped at 300 per radiograph (random, seed 42) so one long weld does not dominate.

# %% [markdown]
# ## Task 3 — cut, label, save

# %%
out = P.gdxray_crops
crops = G.cut_crops(GD, out, window=WINDOW, use_clahe=True, seed=SEED)
print(len(crops), "crops saved to", out)
counts = crops.groupby(["source", "label"]).size().unstack(fill_value=0)
counts.loc["total"] = counts.sum()
counts

# %% [markdown]
# **Missing classes.** GDXray masks carry no defect type, so CR, PO and LP all become one class, DEFECT.
# There is no per-type ground truth; Lab 7 will score the 4-class models as defect-vs-no-defect.

# %%
# Tag each crop with its radiograph's match, then save the label table (no model is run on it here)
for col in ["plate", "plate_split", "is_riawelc_plate", "seen_in_training"]:
    crops["source_" + col if col == "plate" else col] = crops.source.map(matches[col])
crops["seen_in_training"] = crops.seen_in_training.fillna(False).astype(bool)
crops.to_csv(P.results / "gdxray_labels.csv", index=False)
print(f"Crops from radiographs seen in RIAWELC training/validation: {crops.seen_in_training.mean():.1%}")
crops.groupby(["seen_in_training", "label"]).size().unstack(fill_value=0)

# %% [markdown]
# Lab 7 reports GDXray results on **all** crops and on **unseen** crops only (radiographs that are not a
# RIAWELC train/val plate). If every annotated GDXray radiograph turns out to be a RIAWELC training plate, the
# "unseen" set is empty and the honest conclusion is that GDXray W0001 is not an independent test of these models.

# %% [markdown]
# ## Notes — fill in from your outputs
# - **Did:** downloaded GDXray Welds, inventoried it (number, size, bit depth of radiographs), cut weld-band crops with
#   a window size taken from the plate matches, labelled them DEFECT / ND from the W0002 masks, and measured overlap with RIAWELC.
# - **Found:** _how many crops per class; which window size matched; how many GDXray radiographs are RIAWELC plates, and in which split._
# - **Decisions:** CLAHE applied (to match RIAWELC); binary labels only; ambiguous edge crops discarded; ND capped at 300 per radiograph.
# - **Surprised me:** _…_
