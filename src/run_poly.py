"""Ground-truth experiment on the paper's synthetic polynomial data (Table 4 setting).

10 covariates, 100 test units, N training rows. Metric: PEHE against the true CATE.
For every (N, table) all methods see the same data.

    python src/run_poly.py --sizes 2000,5000,10000,20000,50000 --tables 30 --stage weak --n-jobs 7
    python src/run_poly.py --sizes ... --tables 30 --stage run
    python src/run_poly.py --sizes ... --tables 30 --stage baselines --n-jobs 4

Output: results/poly/N<size>/<method>.json   {"pehe": [...per table...], "seconds": [...]}
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "CausalPFN"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO))

from common import METHODS, log, save_json, save_npz  # noqa: E402

TEST_SIZE = 100
X_DIM = 10
OOF_FOLDS = 5

# Methods on this benchmark. "repo_default" is the untouched CATEEstimator(device), as in
# notebooks/ablation_dataset_size.ipynb. Strategy kwargs use the repo's default context
# arguments (4096 / 4096 / 1024) because that is what the notebook uses.
POLY_METHODS = {
    "repo_default": dict(kind="repo"),  # optional; cpfn_default is bit-identical (results/smoke_test.json)
    "cpfn_default": dict(kind="cpfn", tau="insample", kwargs=dict(strategy="default")),
    "cpfn_xfit": dict(kind="cpfn", tau="oof", kwargs=dict(strategy="default")),
    "cpfn_full": METHODS["cpfn_full"],
    "cpfn_A_B2k": METHODS["cpfn_A_B2k"],
    "cpfn_A_B4k": METHODS["cpfn_A_B4k"],
    "cpfn_A_B10k": METHODS["cpfn_A_B10k"],
    "cpfn_K1_B4k": METHODS["cpfn_K1_B4k"],
    "cpfn_A_B4k_resid": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=4_000, residualize=True)),
    "cpfn_A_B4k_perm": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=4_000, permute_features=True)),
    "cpfn_Aplus_B4k": METHODS["cpfn_Aplus_B4k"],
    "S": METHODS["S"],
    "T": METHODS["T"],
    "X": METHODS["X"],
    "DA": METHODS["DA"],
}


def get_table(n, idx):
    """Same construction as notebooks/ablation_dataset_size.ipynb::get_datasets."""
    from benchmarks import PolynomialDataset

    total = n + TEST_SIZE
    ds = PolynomialDataset(n_tables=idx + 1, n_samples=total, test_ratio=TEST_SIZE / total, x_dim_dist=lambda: X_DIM, seed=42)
    cate, _ = ds[idx]
    return cate


def pehe(true, pred):
    return float(np.sqrt(np.mean((np.asarray(true) - np.asarray(pred)) ** 2)))


# ----------------------------------------------------------------------------------------------
def _weak_task(cache, n, idx):
    from strategies import crossfit_weak_tau, fit_weak, weak_tau

    out = cache / f"N{n}" / f"weak_{idx}.npz"
    if out.exists():
        return
    d = get_table(n, idx)
    tic = time.time()
    obj = fit_weak(d.X_train, d.t_train, d.y_train)
    tau_in, tau_q = weak_tau(obj, d.X_train), weak_tau(obj, d.X_test)
    oof = crossfit_weak_tau(d.X_train, d.t_train, d.y_train, folds=OOF_FOLDS, seed=idx)
    save_npz(out, tau_ctx_insample=tau_in, tau_query=tau_q, tau_ctx_oof=oof, seconds=np.array(time.time() - tic))


def stage_weak(args, cache):
    from joblib import Parallel, delayed

    tasks = [(n, i) for n in args.sizes for i in range(args.tables)]
    log(f"weak learner on {len(tasks)} tables")
    Parallel(n_jobs=args.n_jobs)(delayed(_weak_task)(cache, n, i) for n, i in tasks)


# ----------------------------------------------------------------------------------------------
def applicable(name, n):
    """Skip settings that would duplicate another method or cannot be formed."""
    kw = POLY_METHODS[name].get("kwargs", {})
    if kw.get("strategy") == "chunks" and kw.get("chunk_size", 0) < 10**8:
        if kw.get("max_chunks") == 1:
            return n > kw["chunk_size"]
        return n >= 2 * kw["chunk_size"]  # at least two chunks, otherwise it equals cpfn_full
    return True


def stage_run(args, cache, out_root):
    import torch

    from causalpfn import CATEEstimator
    from strategies import StrategyCATE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n in args.sizes:
        for name in args.methods:
            spec = POLY_METHODS[name]
            if spec["kind"] == "baseline" or not applicable(name, n):
                continue
            out = out_root / f"N{n}" / f"{name}.json"
            if out.exists():
                continue
            res = dict(method=name, n=n, pehe=[], seconds=[], peak_gpu_gb=[], status="ok")
            for idx in range(args.tables):
                d = get_table(n, idx)
                tic = time.time()
                try:
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    if spec["kind"] == "repo":
                        est = CATEEstimator(device=device)
                    else:
                        kw = dict(spec["kwargs"])
                        if kw["strategy"] == "default":
                            weak = np.load(cache / f"N{n}" / f"weak_{idx}.npz")
                            kw["tau_context"] = weak["tau_ctx_oof" if spec.get("tau") == "oof" else "tau_ctx_insample"]
                            kw["tau_query"] = weak["tau_query"]
                        est = StrategyCATE(device=device, seed=idx, **kw)
                    est.fit(d.X_train, d.t_train, d.y_train)
                    tau = est.estimate_cate(d.X_test)
                    del est
                except torch.cuda.OutOfMemoryError as e:
                    res["status"] = f"CUDA OOM at table {idx}: {str(e)[:200]}"
                    torch.cuda.empty_cache()
                    break
                res["pehe"].append(pehe(d.true_cate, tau))
                res["seconds"].append(time.time() - tic)
                res["peak_gpu_gb"].append(torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0)
            save_json(out, res)
            if res["pehe"]:
                log(f"N={n} {name}: PEHE {np.mean(res['pehe']):.3f} +- {np.std(res['pehe']) / np.sqrt(len(res['pehe'])):.3f} "
                    f"({len(res['pehe'])} tables, {np.mean(res['seconds']):.1f}s each, peak {max(res['peak_gpu_gb']):.1f} GB) {res['status']}")
            else:
                log(f"N={n} {name}: {res['status']}")


# ----------------------------------------------------------------------------------------------
def _baseline_task(out_root, name, n, tables):
    out = out_root / f"N{n}" / f"{name}.json"
    if out.exists():
        return
    if "benchmarks.baselines" not in sys.modules:
        pkg = types.ModuleType("benchmarks.baselines")
        pkg.__path__ = [str(REPO / "benchmarks" / "baselines")]
        sys.modules["benchmarks.baselines"] = pkg
    mod = importlib.import_module("benchmarks.baselines.econml")
    res = dict(method=name, n=n, pehe=[], seconds=[], status="ok")
    for idx in range(tables):
        d = get_table(n, idx)
        tic = time.time()
        model = getattr(mod, POLY_METHODS[name]["cls"])(hpo=False)
        tau = model.estimate_cate(X_train=d.X_train, t_train=d.t_train, y_train=d.y_train, X_test=d.X_test)
        res["pehe"].append(pehe(d.true_cate, np.asarray(tau).ravel()))
        res["seconds"].append(time.time() - tic)
    save_json(out, res)
    log(f"N={n} {name}: PEHE {np.mean(res['pehe']):.3f}")


def stage_baselines(args, out_root):
    from joblib import Parallel, delayed

    tasks = [(m, n) for n in args.sizes for m in args.methods if POLY_METHODS[m]["kind"] == "baseline"]
    Parallel(n_jobs=args.n_jobs)(delayed(_baseline_task)(out_root, m, n, args.tables) for m, n in tasks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="2000,5000,10000,20000,50000")
    p.add_argument("--tables", type=int, default=30)
    p.add_argument("--stage", required=True, choices=["weak", "run", "baselines"])
    p.add_argument("--methods", default=",".join(m for m in POLY_METHODS if m != "repo_default"))
    p.add_argument("--n-jobs", type=int, default=7)
    args = p.parse_args()
    args.sizes = [int(v) for v in args.sizes.split(",")]
    args.methods = [m for m in args.methods.split(",") if m]

    cache = ROOT / "cache" / "poly"
    out_root = ROOT / "results" / "poly"
    if args.stage == "weak":
        stage_weak(args, cache)
    elif args.stage == "run":
        stage_run(args, cache, out_root)
    else:
        stage_baselines(args, out_root)


if __name__ == "__main__":
    main()
