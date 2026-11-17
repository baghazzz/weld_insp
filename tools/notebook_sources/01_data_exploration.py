# %% [markdown]
# # Lab 1 — Data audit
# **Concepts:** 1 weld defects and radiographs · 2 train / validation / test split · 3 class imbalance. No model is trained here.
#
# **Why start here.** A radiograph is an X-ray "shadow" of the weld: denser metal stops more X-rays, so voids
# (pores), gaps (lack of penetration) and cracks show up as darker or brighter marks. RIAWELC cuts small
# tiles out of such radiographs and labels each tile CR (crack), PO (porosity), LP (lack of penetration) or ND (no defect).
# Before training anything we need to know: how many tiles per class and split, what each class looks like,
# and where the tiles come from. Every later result depends on these facts.

# %% SETUP

# %% [markdown]
# ## Task 1 — get the data and count it
# `get_riawelc` uses `data/riawelc.zip` if you uploaded it. If not, it clones the official repository,
# unpacks the 19 RAR parts, and saves `data/riawelc.zip` to Drive so the next session is fast.
# In Colab the images are unzipped to `/content` (local disk), which is much faster to read than Drive.

# %%
if IN_COLAB:
    !apt-get -qq install -y unrar > /dev/null
from weld_insp.data import get_riawelc, build_image_table
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RIA = get_riawelc(P)
print("Images extracted to", RIA)

# %%
# Folder tree with the number of images in each folder
for split_dir in sorted(p for p in RIA.iterdir() if p.is_dir()):
    print(split_dir.name + "/")
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        print(f"    {cls_dir.name:10s} {len(list(cls_dir.glob('*.png'))):6,d} images")

# %% [markdown]
# One row per image. Besides path, split, label and file name, the table parses the file name
# (plate, section, window, grid position) and stores an MD5 checksum of the file bytes, which Lab 2 uses.

# %%
table_f = P.results / "image_table.csv"
if table_f.exists():
    table = pd.read_csv(table_f)
else:
    table = build_image_table(RIA)
    table.to_csv(table_f, index=False)
print(len(table), "rows saved to", table_f)
table.head()

# %%
counts = pd.crosstab(table.label, table.split, margins=True, margins_name="total")
counts = counts.loc[["CR", "PO", "LP", "ND", "total"], ["train", "val", "test", "total"]]
counts.to_csv(P.results / "class_counts_original.csv")
print("Total images:", len(table), "(expected 24,407)")
counts

# %%
pct = (pd.crosstab(table.label, table.split, normalize="columns") * 100).round(1)
pct.loc[["CR", "PO", "LP", "ND"], ["train", "val", "test"]]

# %%
# Majority-class baseline: a "lazy model" that always answers the most common class
majority = table[table.split == "train"].label.value_counts().idxmax()
for split in ["train", "val", "test"]:
    acc = (table[table.split == split].label == majority).mean()
    print(f"Always predict {majority}: accuracy on {split:5s} = {acc:.1%}")

# %% [markdown]
# **Reading this.** Any real model must clearly beat this number. Accuracy alone can hide a model that
# ignores the rarest class (LP), which is why later labs report per-class recall and macro-F1.

# %%
# Image size check: the README says 224 x 224
table.groupby(["width", "height"]).size().rename("images").to_frame()

# %% [markdown]
# ## Task 2 — look at the images

# %%
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(4, 6, figsize=(12, 8.4))
for r, cls in enumerate(["CR", "PO", "LP", "ND"]):
    sub = table[table.label == cls]
    for c, i in enumerate(rng.choice(len(sub), 6, replace=False)):
        row = sub.iloc[i]
        axes[r, c].imshow(plt.imread(RIA / row.path), cmap="gray")
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, c].set_title(row.file[:22], fontsize=6)
    axes[r, 0].set_ylabel(cls, fontsize=14)
plt.suptitle("RIAWELC: 6 random tiles per class (seed 42)")
plt.tight_layout()
plt.savefig(P.figures / "01_sample_grid.png", dpi=130)
plt.show()

# %% [markdown]
# **What each defect looks like** (one sentence per class, written after looking at the grid):
#
# - **CR (crack):** a thin dark line, sometimes faint, running straight or slightly kinked across the tile in any direction.
# - **PO (porosity):** a round or oval dark blob with soft edges, from a few pixels to half the tile wide.
# - **LP (lack of penetration):** a horizontal dark band running the full width of the tile, along the weld root.
# - **ND (no defect):** grainy weld texture with no isolated mark; brightness changes only gradually across the tile.
#
# The tiles also differ a lot in grain and contrast even within one class, a sign of the CLAHE contrast
# enhancement RIAWELC applied and of different source plates.

# %%
# Do file names carry a source-radiograph ID?
print(table.file.sample(4, random_state=SEED).tolist())
print("\nDistinct plates (RRT-xxR = BAM Round Robin Test radiograph):", table.plate.nunique())
print("Distinct radiograph sections (plate + Img#):", table.radiograph.nunique())
print("Files marked ' - Copia' (Italian for 'copy'):", int(table.is_copy.sum()))
table.groupby("plate").size().sort_values(ascending=False).head(10).rename("tiles")

# %% [markdown]
# **Yes — file names carry a source-radiograph ID.** Example: `RRT-12R_Img1_A80_S2_[6][104].png` =
# plate **RRT-12R** (a BAM Round Robin Test radiograph), section **Img1**, an **80×80 window**, step/scale **S2**,
# grid position row 6, column 104. All tiles come from only ~30 plates, so many tiles share a source radiograph.
# That matters for Lab 2: tiles from the same radiograph can land in both train and test.
#
# The `RRT` plates are the BAM weld radiographs that GDXray also distributes (series W0003), which matters for Lab 3.

# %%
# Pixel-intensity histogram, 200 images per class
fig, ax = plt.subplots(figsize=(8, 4))
stats = []
for cls in ["CR", "PO", "LP", "ND"]:
    sub = table[table.label == cls].sample(200, random_state=SEED)
    px = np.concatenate([plt.imread(RIA / p).ravel() for p in sub.path]) * 255
    ax.hist(px, bins=64, range=(0, 255), density=True, histtype="step", lw=1.8, label=cls)
    stats.append({"class": cls, "mean": px.mean(), "std": px.std(), "median": np.median(px)})
ax.set_xlabel("pixel intensity (0 = black, 255 = white)"); ax.set_ylabel("density"); ax.legend()
ax.set_title("Intensity distribution, 200 random tiles per class")
plt.tight_layout(); plt.savefig(P.figures / "01_intensity_hist.png", dpi=130); plt.show()
pd.DataFrame(stats).round(1)

# %% [markdown]
# **Do the classes differ in brightness? Yes, in spread more than in mean.** Mean intensity is similar (147–162),
# but ND tiles have a much narrower range (std ≈ 33 vs ≈ 52 for the defect classes) and almost no very dark or
# very bright pixels. A model could therefore separate ND partly from brightness range alone, without "seeing"
# a defect. Keep that in mind when Lab 9's Grad-CAM shows where models look.

# %% [markdown]
# ## Notes (what I did, what I found, what surprised me)
# - **Did:** extracted all 24,407 RIAWELC tiles, built `image_table.csv` (one row per tile, with source IDs parsed from file names and an MD5 per file), counted classes per split, viewed samples and intensity histograms.
# - **Found:** classes are moderately imbalanced (CR 31%, PO 26%, ND 25%, LP 18%); the lazy "always CR" model scores 31.3%, so that is the floor. The official split is a stratified 65/25/10 with identical class shares in every split.
# - **Found:** every tile comes from one of only 29 source plates (109 radiograph sections), named RRT-xxR, i.e. BAM Round Robin Test radiographs, the same source GDXray uses.
# - **Surprised me:** tiles are 227×227, not the documented 224×224; 30 files are marked " - Copia" (copies); ND tiles have a visibly narrower brightness range than defect tiles.
