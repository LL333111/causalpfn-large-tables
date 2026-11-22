"""Pooled Qini with paired bootstrap intervals on a subset of folds.

Used when a method has only some folds finished (analyze.py needs all five).

    python src/compare_folds.py megafon_full --folds 0 --methods cpfn_default,weak,cpfn_A_B10k_taufeat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import bootstrap_qini, ci, qini, save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--folds", default="0")
    p.add_argument("--methods", required=True)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    folds = [int(v) for v in args.folds.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    tag_dir = ROOT / "results" / "qini" / args.tag

    data = {}
    for m in methods:
        files = [tag_dir / m / f"fold{f}.npz" for f in folds]
        if all(f.exists() for f in files):
            data[m] = [dict(np.load(f)) for f in files]
        else:
            print(f"skipped {m}: not all of folds {folds} are finished")
    ref = next(iter(data.values()))
    t = np.concatenate([f["t"] for f in ref]).astype(np.float64)
    y = np.concatenate([f["y"] for f in ref]).astype(np.float64)
    for m, v in data.items():  # same test units in the same order for every method
        assert np.array_equal(np.concatenate([f["t"] for f in v]), t) and np.array_equal(np.concatenate([f["y"] for f in v]), y), m
    scores = {m: np.concatenate([f["tau"] for f in v]) for m, v in data.items()}
    boot = bootstrap_qini(t, y, scores, n_boot=args.n_boot, seed=0, normalize=True)

    out = dict(tag=args.tag, folds=folds, n_units=int(len(t)), n_boot=args.n_boot, methods={})
    for m, v in data.items():
        d = dict(
            pooled=qini(t, y, scores[m], True),
            pooled_ci95=ci(boot[m]),
            per_fold=[qini(f["t"], f["y"], f["tau"], True) for f in v],
            seconds_per_fold=float(np.mean([f["seconds"] for f in v])),
        )
        for base in data:
            if base != m:
                d[f"diff_vs_{base}"] = dict(pooled=d["pooled"] - qini(t, y, scores[base], True), ci95=ci(boot[m] - boot[base]))
        out["methods"][m] = d
    path = Path(args.out) if args.out else ROOT / "results" / "summary" / f"partial_{args.tag}_folds{''.join(map(str, folds))}.json"
    save_json(path, out)
    print(f"{args.tag}, folds {folds}, {len(t)} test units")
    for m, d in sorted(out["methods"].items(), key=lambda kv: -kv[1]["pooled"]):
        lo, hi = d["pooled_ci95"]
        print(f"  {m:24s} {d['pooled']:.4f} [{lo:.4f}, {hi:.4f}]  folds {np.round(d['per_fold'], 4).tolist()}")
    for m, d in out["methods"].items():
        for k, v in d.items():
            if k.startswith("diff_vs_"):
                print(f"  {m} - {k[8:]}: {v['pooled']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(f"written {path}")


if __name__ == "__main__":
    main()
