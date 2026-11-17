# %%
# ---- Setup: identical in every notebook ------------------------------------
# Finds the project folder (Google Drive in Colab, or D:\weld_insp locally),
# makes src/ importable, fixes the random seed.
import os, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/weld_insp")
else:
    ROOT = Path(os.environ.get("WELD_ROOT", Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
os.environ["WELD_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src"))
QUICK = os.environ.get("WELD_QUICK") == "1"   # tiny smoke-test mode, leave off for real runs

from weld_insp.config import get_paths, set_seed, CLASSES, SEED
P = get_paths()
P.show()
set_seed(SEED)
