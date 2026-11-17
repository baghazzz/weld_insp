"""Threshold sweeps and the cost of misjudgment (Lab 8).

Two ways to be wrong at an inspection station:
  missed defect  (escape)       a defective weld is called good  -> field failure risk
  false reject   (overkill)     a good weld is called defective  -> rework / re-inspection
Moving the decision threshold trades one for the other. With a cost for each,
the best threshold is the one with the lowest expected cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sweep(y_true_bin, score, n: int = 401) -> pd.DataFrame:
    """Missed-defect rate and false-reject rate at every threshold.
    y_true_bin: 1 = defect, 0 = good. score: higher = more likely defect."""
    y = np.asarray(y_true_bin).astype(int)
    s = np.asarray(score, dtype=float)
    lo, hi = s.min(), s.max()
    thr = np.unique(np.concatenate([np.linspace(lo, hi, n), [hi + 1e-9]]))
    pos, neg = y == 1, y == 0
    rows = []
    for t in thr:
        pred = s >= t
        tp, fn = (pred & pos).sum(), (~pred & pos).sum()
        fp, tn = (pred & neg).sum(), (~pred & neg).sum()
        rows.append(dict(threshold=t,
                         missed_defect_rate=fn / max(pos.sum(), 1),
                         false_reject_rate=fp / max(neg.sum(), 1),
                         precision=tp / max(tp + fp, 1) if (tp + fp) else 1.0,
                         recall=tp / max(pos.sum(), 1)))
    return pd.DataFrame(rows)


def cost_per_1000(sw: pd.DataFrame, cost_ratio: float, defect_rate: float,
                  cost_false_reject: float = 1.0) -> pd.Series:
    """Expected cost per 1,000 welds, in units of one false-reject (rework) cost.
    cost_ratio = cost of one missed defect / cost of one false reject.
    defect_rate = share of welds that are truly defective (an ASSUMPTION for production)."""
    miss = 1000 * defect_rate * sw.missed_defect_rate * cost_ratio * cost_false_reject
    fr = 1000 * (1 - defect_rate) * sw.false_reject_rate * cost_false_reject
    return miss + fr


def operating_point(sw: pd.DataFrame, cost_ratio: float, defect_rate: float) -> dict:
    c = cost_per_1000(sw, cost_ratio, defect_rate)
    i = int(c.idxmin())
    r = sw.loc[i]
    return dict(cost_ratio=cost_ratio, defect_rate=defect_rate, threshold=float(r.threshold),
                missed_defect_rate=float(r.missed_defect_rate),
                false_reject_rate=float(r.false_reject_rate),
                cost_per_1000=float(c[i]))
