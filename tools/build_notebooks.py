"""Build notebooks/*.ipynb from tools/notebook_sources/*.py (jupytext percent format).

    python tools/build_notebooks.py            # build all
    python tools/build_notebooks.py 01 02      # build some
    python tools/build_notebooks.py 01 --run   # build and execute (writes outputs)

The shared setup cell lives in _setup.py and replaces each '# %% SETUP' marker.
"""
import os
import sys
from pathlib import Path

import jupytext
import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "notebook_sources"
OUT = ROOT / "notebooks"


def build(src: Path) -> Path:
    setup = (SRC / "_setup.py").read_text(encoding="utf-8")
    setup = setup.split("# %%", 1)[1]                       # drop leading marker
    text = src.read_text(encoding="utf-8").replace("# %% SETUP", "# %%" + setup.rstrip())
    nb = jupytext.reads(text, fmt="py:percent")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata.pop("jupytext", None)
    out = OUT / (src.stem + ".ipynb")
    nbformat.write(nb, out)
    return out


def run(nb_path: Path, timeout: int = 7200) -> None:
    from nbclient import NotebookClient
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT)}}).execute()
    nbformat.write(nb, nb_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sources = sorted(p for p in SRC.glob("[0-9]*.py") if not args or any(p.name.startswith(a) for a in args))
    OUT.mkdir(exist_ok=True)
    for s in sources:
        out = build(s)
        print("built", out.name)
        if "--run" in sys.argv:
            os.environ.setdefault("WELD_ROOT", str(ROOT))
            run(out)
            print("  executed", out.name)
