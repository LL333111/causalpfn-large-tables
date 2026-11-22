"""One table over all tables: pooled Qini of the main methods on the folds they share.

For every tag under results/qini/ the folds used are those where `weak`, `cpfn_default` and
`cpfn_A_B10k_taufeat` all have predictions (so every difference is paired on the same test units).
Other methods are listed only if they cover the same folds. The two 50/50 blends of `weak` and
`cpfn_default` are built as in blend.py.

    python src/overview.py            # writes results/summary/overview.json and prints a table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blend import pct, zscore  # noqa: E402
from common import bootstrap_qini, ci, qini, save_json  # noqa: E402

RES = ROOT / "results"
N_BOOT = 300
NEED = ["weak", "cpfn_default", "cpfn_A_B10k_taufeat"]
OPTIONAL = ["cpfn_placebo", "cpfn_K1_B10k", "cpfn_A_B10k", "cpfn_K1_B10k_taufeat", "cpfn_default_taufeat", "cpfn_xfit"]
NEW = "cpfn_A_B10k_taufeat"
VERSUS = ["cpfn_default", "weak", "blend_z", "blend_rank", "cpfn_K1_B10k_taufeat", "cpfn_default_taufeat", "cpfn_xfit"]


def one(tag):
    d = RES / "qini" / tag
    folds = [f for f in range(5) if all((d / m / f"fold{f}.npz").exists() for m in NEED)]
    if not folds:
        return None
    methods = NEED + [m for m in OPTIONAL if all((d / m / f"fold{f}.npz").exists() for f in folds)]
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
    out = dict(tag=tag, folds=folds, n_units=int(len(t)), n_boot=N_BOOT, methods={}, new_minus={})
    for m in scores:
        out["methods"][m] = dict(pooled=qini(t, y, scores[m], True), pooled_ci95=ci(boot[m]))
    for other in VERSUS:
        if other in scores:
            out["new_minus"][other] = dict(
                pooled=out["methods"][NEW]["pooled"] - out["methods"][other]["pooled"], ci95=ci(boot[NEW] - boot[other])
            )
    return out


def main():
    tags = sys.argv[1:] or sorted(p.name for p in (RES / "qini").iterdir() if p.is_dir())
    path = RES / "summary" / "overview.json"
    # when only some tables are named, keep the entries of the others
    res = json.loads(path.read_text()) if (sys.argv[1:] and path.exists()) else {}
    for tag in tags:
        out = one(tag)
        if out is None:
            continue
        res[tag] = out
        print(f"\n{tag}  folds {out['folds']}  n={out['n_units']}")
        for m, e in sorted(out["methods"].items(), key=lambda kv: -kv[1]["pooled"]):
            print(f"  {m:24s} {e['pooled']:.3f} [{e['pooled_ci95'][0]:.3f}, {e['pooled_ci95'][1]:.3f}]")
        for other, e in out["new_minus"].items():
            print(f"  new - {other:22s} {e['pooled']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")
    save_json(RES / "summary" / "overview.json", res)


if __name__ == "__main__":
    main()
