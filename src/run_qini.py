"""Run CATE methods on an RCT uplift table with the repo's folds and save per-fold predictions.

    python src/run_qini.py --dataset x5 --stage data
    python src/run_qini.py --dataset x5 --stage weak --n-jobs 9
    python src/run_qini.py --dataset x5 --stage run --methods cpfn_default,cpfn_A_B10k
    python src/run_qini.py --dataset x5 --stage baselines --methods S,T,X,DA --n-jobs 4
    (add --subsample 50000 for the Table-2 style subsample)

Outputs (never overwritten; delete a file to re-run it):
    cache/<tag>/data.npz                       table + fold indices from the repo's dataset class
    cache/<tag>/weak_fold<i>.npz               weak-learner CATE: in-sample, query, out-of-fold
    results/qini/<tag>/<method>/fold<i>.npz    predictions + bookkeeping
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
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

SEED = 82718  # notebooks/qini.ipynb
N_FOLDS = 5
OOF_FOLDS = 5


def tag_of(args):
    return f"{args.dataset}_{'sub' + str(args.subsample // 1000) + 'k' if args.subsample else 'full'}"


# ----------------------------------------------------------------------------------------------
def stage_data(args, cache):
    path = cache / "data.npz"
    if path.exists():
        return
    import benchmarks

    if args.dataset in ("hill1", "hill2"):  # arms as in notebooks/qini.ipynb
        arm = {"hill1": "Womens E-Mail", "hill2": "Mens E-Mail"}[args.dataset]
        ds = benchmarks.HillstromDataset(seed=SEED, n_folds=N_FOLDS, outcome_col="visit", control_arm="No E-Mail", treatment_arm=arm, subsample_max_rows=args.subsample)
    else:
        cls = {"x5": benchmarks.X5Dataset, "lenta": benchmarks.LentaDataset, "megafon": benchmarks.MegafonDataset}[args.dataset]
        ds = cls(seed=SEED, n_folds=N_FOLDS, subsample_max_rows=args.subsample)
    arrays = dict(X=np.asarray(ds.all_X), t=np.asarray(ds.all_t), y=np.asarray(ds.all_y))
    for i in range(N_FOLDS):
        arrays[f"train_idx_{i}"] = np.asarray(ds.all_fold_train_idx[i])
        arrays[f"test_idx_{i}"] = np.asarray(ds.all_fold_test_idx[i])
    save_npz(path, **arrays)
    X, t, y = arrays["X"], arrays["t"], arrays["y"]
    info = dict(
        n=int(len(t)), n_features=int(X.shape[1]), X_dtype=str(X.dtype), t_dtype=str(t.dtype), y_dtype=str(y.dtype),
        treated_frac=float(np.mean(t)), outcome_mean=float(np.mean(y)), n_distinct_y=int(len(np.unique(y))),
        n_nan=int(np.isnan(X.astype(np.float64)).sum()),
        raw_ate=float(y[t == 1].mean() - y[t == 0].mean()),
    )
    save_json(cache / "data_info.json", info)
    log(f"data cached: {info}")


def load_fold(cache, i):
    z = np.load(cache / "data.npz", allow_pickle=False)
    tr, te = z[f"train_idx_{i}"], z[f"test_idx_{i}"]
    X, t, y = z["X"], z["t"], z["y"]
    return X[tr], t[tr], y[tr], X[te], t[te], y[te]


# ----------------------------------------------------------------------------------------------
def _weak_task(cache, fold, part):
    """part = -1: fit on the whole training fold; part = k >= 0: k-th out-of-fold split."""
    from sklearn.model_selection import KFold

    from strategies import fit_weak, weak_tau

    out = cache / f"weakpart_fold{fold}_{part}.npz"
    if out.exists():
        return
    X_tr, t_tr, y_tr, X_te, _, _ = load_fold(cache, fold)
    tic = time.time()
    if part < 0:
        obj = fit_weak(X_tr, t_tr, y_tr)
        save_npz(out, tau_ctx_insample=weak_tau(obj, X_tr), tau_query=weak_tau(obj, X_te), seconds=np.array(time.time() - tic))
    else:
        splits = list(KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=fold).split(X_tr))
        a, b = splits[part]
        obj = fit_weak(X_tr[a], t_tr[a], y_tr[a])
        save_npz(out, idx=b, tau=weak_tau(obj, X_tr[b]), seconds=np.array(time.time() - tic))


def stage_weak(args, cache):
    from joblib import Parallel, delayed

    tasks = [(f, p) for f in args.folds for p in range(-1, OOF_FOLDS) if not (cache / f"weak_fold{f}.npz").exists()]
    log(f"weak learner: {len(tasks)} fits, n_jobs={args.n_jobs}")
    Parallel(n_jobs=args.n_jobs)(delayed(_weak_task)(cache, f, p) for f, p in tasks)
    for f in args.folds:
        final = cache / f"weak_fold{f}.npz"
        if final.exists():
            continue
        main = np.load(cache / f"weakpart_fold{f}_-1.npz")
        oof = np.empty_like(main["tau_ctx_insample"], dtype=np.float64)
        secs = [float(main["seconds"])]
        for p in range(OOF_FOLDS):
            z = np.load(cache / f"weakpart_fold{f}_{p}.npz")
            oof[z["idx"]] = z["tau"]
            secs.append(float(z["seconds"]))
        save_npz(final, tau_ctx_insample=main["tau_ctx_insample"], tau_query=main["tau_query"], tau_ctx_oof=oof, seconds=np.array(secs))
        log(f"weak fold {f}: fit seconds {np.round(secs, 1).tolist()}")


# ----------------------------------------------------------------------------------------------
def stage_run(args, cache, out_root):
    import torch

    from strategies import StrategyCATE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.methods:
        spec = METHODS[name]
        for f in args.folds:
            out = out_root / name / f"fold{f}.npz"
            if out.exists() or (out_root / name / f"fold{f}.failed").exists():
                continue
            X_tr, t_tr, y_tr, X_te, t_te, y_te = load_fold(cache, f)
            weak_path = cache / f"weak_fold{f}.npz"
            weak = np.load(weak_path) if (weak_path.exists() and not args.inline_weak) else None
            if args.inline_weak and spec["kind"] == "weak":
                continue  # written from the cpfn_default run below
            is_random = spec.get("tau") == "random"  # placebo grouping needs no weak learner
            if weak is None and not args.inline_weak and not is_random and (spec["kind"] == "weak" or spec.get("kwargs", {}).get("strategy") == "default"):
                log(f"{name} fold {f}: weak-learner cache missing, skipped for now")
                continue
            tic = time.time()
            extra = {}
            try:
                if spec["kind"] == "weak":
                    tau = weak["tau_query"]
                else:
                    kw = dict(spec["kwargs"])
                    if kw["strategy"] == "default" and is_random:
                        rng = np.random.default_rng(1000 + f)
                        kw["tau_context"] = rng.normal(size=len(t_tr))
                        kw["tau_query"] = rng.normal(size=len(t_te))
                    elif kw["strategy"] == "default" and args.inline_weak:
                        # the estimator trains its own weak learner exactly as the repo does
                        # (including the SVD feature reduction for wide tables)
                        if spec.get("tau") == "oof":
                            kw["crossfit_folds"] = OOF_FOLDS
                    elif kw["strategy"] == "default":
                        kw["tau_context"] = weak["tau_ctx_oof" if spec.get("tau") == "oof" else "tau_ctx_insample"]
                        kw["tau_query"] = weak["tau_query"]
                    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
                    X_fit, X_pred = X_tr, X_te
                    if spec.get("aug") == "tau":
                        if weak is None:
                            log(f"{name} fold {f}: needs the weak-learner cache, skipped")
                            continue
                        X_fit = np.column_stack([X_tr, weak["tau_ctx_oof"]]).astype(np.float32)
                        X_pred = np.column_stack([X_te, weak["tau_query"]]).astype(np.float32)
                        kw["keep_last_cols"] = 1
                    if spec.get("aug") in ("stack", "stackonly"):
                        if weak is None or not (cache / f"stackpart_DA_fold{f}_{OOF_FOLDS - 1}.npz").exists():
                            log(f"{name} fold {f}: stack scores missing, skipped")
                            continue
                        s_ctx, s_qry = load_stack(cache, f, weak, len(t_tr))
                        if spec["aug"] == "stack":
                            X_fit = np.column_stack([X_tr, s_ctx]).astype(np.float32)
                            X_pred = np.column_stack([X_te, s_qry]).astype(np.float32)
                            kw["keep_last_cols"] = s_ctx.shape[1]
                        else:
                            X_fit, X_pred = s_ctx.astype(np.float32), s_qry.astype(np.float32)
                    est = StrategyCATE(device=device, seed=f, **kw)
                    est.fit(X_fit, t_tr, y_tr)
                    tau = est.estimate_cate(X_pred)
                    n = len(t_te)
                    for key, val in est.record.items():
                        if key == "batch_id":
                            extra["batch_id_t0"], extra["batch_id_t1"] = val[:n], val[n:]
                        elif key == "query_tau":
                            extra["tau_weak_inline"] = np.asarray(val)[:n]
                        elif val is not None:
                            extra["rec_" + key] = np.asarray(val)
                    if device == "cuda":
                        extra["peak_gpu_gb"] = np.array(torch.cuda.max_memory_allocated() / 2**30)
                    del est
            except torch.cuda.OutOfMemoryError as e:  # keep going, record the failure
                (out_root / name).mkdir(parents=True, exist_ok=True)
                (out_root / name / f"fold{f}.failed").write_text(f"CUDA OOM: {str(e)[:300]}")
                log(f"{name} fold {f}: CUDA OOM, skipped")
                torch.cuda.empty_cache()
                continue
            secs = time.time() - tic
            tau_weak = extra.pop("tau_weak_inline", None) if weak is None else weak["tau_query"]
            if weak is None and is_random:
                tau_weak = None  # the recorded scores are the placebo noise, not a weak learner
            extra.pop("tau_weak_inline", None)
            save_npz(out, tau=np.asarray(tau, dtype=np.float64), t=t_te, y=y_te, tau_weak=tau_weak, seconds=np.array(secs), **extra)
            if args.inline_weak and name == "cpfn_default" and tau_weak is not None:
                save_npz(out_root / "weak" / f"fold{f}.npz", tau=np.asarray(tau_weak, dtype=np.float64), t=t_te, y=y_te, seconds=np.array(0.0))
            log(f"{name} fold {f}: {secs:.1f}s  tau mean={np.mean(tau):.4f} sd={np.std(tau):.4f}")


# ----------------------------------------------------------------------------------------------
def _baseline_task(cache, out_root, name, f):
    out = out_root / name / f"fold{f}.npz"
    if out.exists():
        return
    # import the repo's EconML baselines without benchmarks/baselines/__init__ (which pulls in
    # R / catenets baselines that are not needed here)
    if "benchmarks.baselines" not in sys.modules:
        pkg = types.ModuleType("benchmarks.baselines")
        pkg.__path__ = [str(REPO / "benchmarks" / "baselines")]
        sys.modules["benchmarks.baselines"] = pkg
    mod = importlib.import_module("benchmarks.baselines.econml")
    X_tr, t_tr, y_tr, X_te, t_te, y_te = load_fold(cache, f)
    tic = time.time()
    model = getattr(mod, METHODS[name]["cls"])(hpo=False)
    tau = model.estimate_cate(X_train=X_tr, t_train=t_tr, y_train=y_tr, X_test=X_te)
    save_npz(out, tau=np.asarray(tau, dtype=np.float64).ravel(), t=t_te, y=y_te, seconds=np.array(time.time() - tic))
    log(f"{name} fold {f}: {time.time() - tic:.1f}s")


STACK_METHODS = ["S", "T", "X", "DA"]


def _stack_task(cache, name, f, part):
    """Scores of one classical estimator: part -1 -> fit on the training fold, predict test units;
    part k >= 0 -> fit without the k-th split of the training fold, predict that split."""
    from sklearn.model_selection import KFold

    out = cache / f"stackpart_{name}_fold{f}_{part}.npz"
    if out.exists():
        return
    if "benchmarks.baselines" not in sys.modules:
        pkg = types.ModuleType("benchmarks.baselines")
        pkg.__path__ = [str(REPO / "benchmarks" / "baselines")]
        sys.modules["benchmarks.baselines"] = pkg
    mod = importlib.import_module("benchmarks.baselines.econml")
    X_tr, t_tr, y_tr, X_te, _, _ = load_fold(cache, f)
    model = getattr(mod, METHODS[name]["cls"])(hpo=False)
    if part < 0:
        tau = model.estimate_cate(X_train=X_tr, t_train=t_tr, y_train=y_tr, X_test=X_te)
        save_npz(out, tau=np.asarray(tau, dtype=np.float64).ravel())
    else:
        a, b = list(KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=f).split(X_tr))[part]
        tau = model.estimate_cate(X_train=X_tr[a], t_train=t_tr[a], y_train=y_tr[a], X_test=X_tr[b])
        save_npz(out, idx=b, tau=np.asarray(tau, dtype=np.float64).ravel())


def stage_stack(args, cache):
    from joblib import Parallel, delayed

    tasks = [(m, f, p) for f in args.folds for m in STACK_METHODS for p in range(-1, OOF_FOLDS)]
    log(f"stack scores: {len(tasks)} fits, n_jobs={args.n_jobs}")
    Parallel(n_jobs=args.n_jobs)(delayed(_stack_task)(cache, m, f, p) for m, f, p in tasks)


def load_stack(cache, f, weak, n_tr):
    """(n_tr, 5) out-of-fold scores for the context rows and (n_te, 5) scores for the test units."""
    ctx, qry = [weak["tau_ctx_oof"]], [weak["tau_query"]]
    for m in STACK_METHODS:
        oof = np.empty(n_tr, dtype=np.float64)
        for p in range(OOF_FOLDS):
            z = np.load(cache / f"stackpart_{m}_fold{f}_{p}.npz")
            oof[z["idx"]] = z["tau"]
        ctx.append(oof)
        qry.append(np.load(cache / f"stackpart_{m}_fold{f}_-1.npz")["tau"])
    return np.column_stack(ctx), np.column_stack(qry)


def stage_baselines(args, cache, out_root):
    from joblib import Parallel, delayed

    tasks = [(m, f) for m in args.methods for f in args.folds]
    Parallel(n_jobs=args.n_jobs)(delayed(_baseline_task)(cache, out_root, m, f) for m, f in tasks)


# ----------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="x5", choices=["x5", "lenta", "megafon", "hill1", "hill2"])
    p.add_argument("--inline-weak", action="store_true", help="let the estimator train its own weak learner (needed for tables wider than the model)")
    p.add_argument("--stratifier", default="repo", choices=["repo", "hist"], help="hist: same S-learner with HistGradientBoostingRegressor; tag gets the suffix _hgb")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--stage", required=True, choices=["data", "weak", "run", "baselines", "stack"])
    p.add_argument("--methods", default="")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    args.methods = [m for m in args.methods.split(",") if m]
    args.folds = [int(v) for v in args.folds.split(",")]

    tag = tag_of(args)
    base_cache = ROOT / "cache" / tag
    if args.stratifier == "hist":
        # same table and folds, stratifier fitted with histogram-based boosting; own cache and
        # result folders so that nothing made with the repo's stratifier is overwritten
        assert not args.inline_weak, "--stratifier hist is for tables that use the weak-learner cache"
        os.environ["CPFN_STRATIFIER"] = "hist"
        tag += "_hgb"
    cache = ROOT / "cache" / tag
    out_root = ROOT / "results" / "qini" / tag
    cache.mkdir(parents=True, exist_ok=True)
    if args.stratifier == "hist":
        for name in ("data.npz", "data_info.json"):
            if (base_cache / name).exists() and not (cache / name).exists():
                shutil.copy(base_cache / name, cache / name)

    stage_data(args, cache)
    if args.stage == "weak":
        stage_weak(args, cache)
    elif args.stage == "run":
        stage_run(args, cache, out_root)
    elif args.stage == "baselines":
        stage_baselines(args, cache, out_root)
    elif args.stage == "stack":
        stage_stack(args, cache)


if __name__ == "__main__":
    main()
