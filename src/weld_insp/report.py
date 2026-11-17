"""Write result tables into README.md / docs/summary.md between marker comments, so every
number in the write-up comes from a results CSV rather than being typed by hand.

A block in a markdown file looks like:
    <!-- BEGIN headline -->
    ...replaced automatically...
    <!-- END headline -->
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fill_block(path: Path, name: str, text: str) -> bool:
    """Replace the content of block `name` in file `path`. Returns True if found."""
    path = Path(path)
    s = path.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- BEGIN {re.escape(name)} -->\n)(.*?)(<!-- END {re.escape(name)} -->)", re.S)
    if not pat.search(s):
        return False
    s = pat.sub(lambda m: m.group(1) + text.rstrip() + "\n" + m.group(3), s)
    path.write_text(s, encoding="utf-8")
    return True


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"
