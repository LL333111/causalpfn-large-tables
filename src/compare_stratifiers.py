"""Same comparison with two implementations of the stratifier.

`<tag>`      : the repo's GradientBoostingRegressor S-learner (exact splits)
`<tag>_hgb`  : the same S-learner with HistGradientBoostingRegressor (`run_qini.py --stratifier hist`)

Folds and test units are identical, so every difference is paired. For each table: pooled Qini
of the stratifier alone, the released rule and score-as-feature (one chunk, all chunks) under
both stratifiers, the differences inside each setting, and the change from one to the other.

    python src/compare_stratifiers.py [tag ...]      # writes results/summary/stratifier_<tag>.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import bootstrap_qini, ci, qini, save_json  # noqa: E402

RES = ROOT / "results"
N_BOOT = 300
METHODS = ["weak", "cpfn_default", "cpfn_K1_B10k_taufeat", "cpfn_A_B10k_taufeat"]
NEW = "cpfn_A_B10k_taufeat"


def one(tag):
    a, b = RES / "qini" / tag, RES / "qini" / f"{tag}_hgb"
    if not b.is_dir():
        return None
    methods = [m for m in METHODS if (b / m).is_dir()]
    folds = [f for f in range(5) if all((b / m / f"fold{f}.npz").exists() for m in methods)]
    methods_a = [m for m in methods if all((a / m / f"fold{f}.npz").exists() for f in folds)]
    if not folds:
        return None
    parts, t_all, y_all, secs = {}, [], [], {}
    for f in folds:
        ref = np.load(b / methods[0] / f"fold{f}.npz")
        t, y = ref["t"], ref["y"]
        for src, d, ms in (("repo", a, methods_a), ("hist", b, methods)):
            for m in ms:
                z = np.load(d / m / f"fold{f}.npz")
                assert np.array_equal(z["t"], t) and np.array_equal(z["y"], y), (tag, src, m, f)
                parts.setdefault(f"{src}:{m}", []).append(np.asarray(z["tau"], dtype=np.float64))
                secs.setdefault(f"{src}:{m}", []).append(float(z["seconds"]))
        t_all.append(t)
        y_all.append(y)
    t = np.concatenate(t_all).astype(np.float64)
    y = np.concatenate(y_all).astype(np.float64)
    scores = {k: np.concatenate(v) for k, v in parts.items()}
    boot = bootstrap_qini(t, y, scores, n_boot=N_BOOT, seed=0, normalize=True)
    q = {k: qini(t, y, s, True) for k, s in scores.items()}

    def diff(k1, k2):
        return dict(pooled=q[k1] - q[k2], ci95=ci(boot[k1] - boot[k2]))

    out = dict(tag=tag, folds=folds, n_units=int(len(t)), n_boot=N_BOOT, pooled={}, within={}, hist_minus_repo={})
    for k in scores:
        out["pooled"][k] = dict(pooled=q[k], pooled_ci95=ci(boot[k]), model_seconds=float(np.mean(secs[k])))
    for src in ("repo", "hist"):
        for other in ("cpfn_default", "weak", "cpfn_K1_B10k_taufeat"):
            if f"{src}:{NEW}" in q and f"{src}:{other}" in q:
                out["within"][f"{src}: new - {other}"] = diff(f"{src}:{NEW}", f"{src}:{other}")
        if f"{src}:cpfn_default" in q and f"{src}:weak" in q:
            out["within"][f"{src}: cpfn_default - weak"] = diff(f"{src}:cpfn_default", f"{src}:weak")
    for m in methods_a:
        out["hist_minus_repo"][m] = diff(f"hist:{m}", f"repo:{m}")
    save_json(RES / "summary" / f"stratifier_{tag}.json", out)
    return out


def main():
    tags = sys.argv[1:] or sorted(p.name[:-4] for p in (RES / "qini").iterdir() if p.name.endswith("_hgb"))
    for tag in tags:
        out = one(tag)
        if out is None:
            continue
        print(f"\n{tag}  folds {out['folds']}  n={out['n_units']}")
        for k, e in out["pooled"].items():
            print(f"  {k:30s} {e['pooled']:.3f} [{e['pooled_ci95'][0]:.3f}, {e['pooled_ci95'][1]:.3f}]  {e['model_seconds']:.0f}s")
        for k, e in {**out["within"], **{'hist - repo, ' + m: v for m, v in out["hist_minus_repo"].items()}}.items():
            print(f"  {k:44s} {e['pooled']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
