"""Trivial-ensemble control: average the released pipeline's score with its weak learner's score.

No model is run. For every table under results/qini/ that has both `cpfn_default` and `weak`,
two fixed 50/50 blends are built from the saved test-fold predictions:

    blend_rank : mean of the two within-fold rank percentiles (ties keep the same rank)
    blend_z    : mean of the two within-fold z-scores

The question is whether score-as-feature (`cpfn_*_taufeat`) does more than such a blend.
Pooled Qini with paired bootstrap intervals, on the folds that every listed method has.

    python src/blend.py                 # all tables
    python src/blend.py x5_full         # one table

Writes results/summary/blend_<tag>.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import bootstrap_qini, ci, qini, save_json  # noqa: E402

RES = ROOT / "results"
N_BOOT = 300
COMPARE = ["cpfn_K1_B10k_taufeat", "cpfn_A_B10k_taufeat", "cpfn_default_taufeat"]


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def pct(x):
    return rankdata(x, method="average") / len(x)


def one(tag):
    d = RES / "qini" / tag
    need = ["cpfn_default", "weak"]
    methods = need + [m for m in COMPARE if (d / m).is_dir()]
    folds = [f for f in range(5) if all((d / m / f"fold{f}.npz").exists() for m in need)]
    if not folds:
        return None
    # optional methods are kept only if they cover the same folds; otherwise restrict the folds
    have = {m: [f for f in folds if (d / m / f"fold{f}.npz").exists()] for m in methods}
    methods = [m for m in methods if have[m]]
    folds = [f for f in folds if all(f in have[m] for m in methods)]
    parts = {m: [] for m in methods + ["blend_rank", "blend_z"]}
    t_all, y_all = [], []
    for f in folds:
        z = {m: np.load(d / m / f"fold{f}.npz") for m in methods}
        t, y = z["cpfn_default"]["t"], z["cpfn_default"]["y"]
        for m in methods:
            assert np.array_equal(z[m]["t"], t) and np.array_equal(z[m]["y"], y), (tag, m, f)
            parts[m].append(np.asarray(z[m]["tau"], dtype=np.float64))
        a, b = parts["cpfn_default"][-1], parts["weak"][-1]
        parts["blend_rank"].append(0.5 * (pct(a) + pct(b)))
        parts["blend_z"].append(0.5 * (zscore(a) + zscore(b)))
        t_all.append(t)
        y_all.append(y)
    t = np.concatenate(t_all).astype(np.float64)
    y = np.concatenate(y_all).astype(np.float64)
    scores = {m: np.concatenate(v) for m, v in parts.items()}
    boot = bootstrap_qini(t, y, scores, n_boot=N_BOOT, seed=0, normalize=True)
    out = dict(tag=tag, folds=folds, n_units=int(len(t)), n_boot=N_BOOT, methods={})
    for m in scores:
        e = dict(pooled=qini(t, y, scores[m], True), pooled_ci95=ci(boot[m]))
        for other in scores:
            if other != m:
                e[f"diff_vs_{other}"] = dict(pooled=e["pooled"] - qini(t, y, scores[other], True), ci95=ci(boot[m] - boot[other]))
        out["methods"][m] = e
    save_json(RES / "summary" / f"blend_{tag}.json", out)
    return out


def main():
    tags = sys.argv[1:] or sorted(p.name for p in (RES / "qini").iterdir() if p.is_dir())
    for tag in tags:
        out = one(tag)
        if out is None:
            continue
        print(f"\n{tag}  folds {out['folds']}  n={out['n_units']}")
        for m, e in out["methods"].items():
            lo, hi = e["pooled_ci95"]
            line = f"  {m:24s} {e['pooled']:.3f} [{lo:.3f}, {hi:.3f}]"
            for other in ("blend_rank", "blend_z"):
                if other != m and f"diff_vs_{other}" in e and m not in ("blend_rank", "blend_z"):
                    dd = e[f"diff_vs_{other}"]
                    line += f"   vs {other}: {dd['pooled']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]"
            print(line)


if __name__ == "__main__":
    main()
