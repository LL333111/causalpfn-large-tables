"""Shared helpers: method registry, weak-learner cache, Qini metrics, bootstrap."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------
# Method registry. Chunk sizes and the number of chunks were fixed before any test fold was looked at.
#   kind "cpfn"      -> StrategyCATE(**kwargs)
#   kind "weak"      -> the repo's stratifier alone
#   kind "baseline"  -> repo baseline class, hpo=False
# `tau` says which weak-learner CATE the default strategy uses for the context rows:
#   "insample" (released behaviour) or "oof" (cross-fitted).
# ----------------------------------------------------------------------------------------------
QINI_CTX = dict(max_context_length=50_000)  # as in notebooks/qini.ipynb

METHODS = {
    # released behaviour, re-implemented with bookkeeping
    "cpfn_default": dict(kind="cpfn", tau="insample", kwargs=dict(strategy="default", **QINI_CTX)),
    # same rule, cross-fitted weak-learner CATE for the context rows
    "cpfn_xfit": dict(kind="cpfn", tau="oof", kwargs=dict(strategy="default", **QINI_CTX)),
    # random chunks shared by all queries, predictions averaged over chunks
    "cpfn_A_B10k": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=10_000)),
    "cpfn_A_B4k": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=4_000)),
    "cpfn_A_B2k": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=2_000)),
    "cpfn_A_B40k": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=40_000)),
    # single random context (no batch effects, uses only B rows)
    "cpfn_K1_B10k": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=10_000, max_chunks=1)),
    "cpfn_K1_B4k": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=4_000, max_chunks=1)),
    # random chunks plus one add-on at a time, then all three
    "cpfn_A_B10k_strat": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=10_000, stratify="arm_outcome")),
    "cpfn_A_B10k_perm": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=10_000, permute_features=True)),
    "cpfn_A_B10k_resid": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=10_000, residualize=True)),
    "cpfn_Aplus_B10k": dict(
        kind="cpfn",
        kwargs=dict(strategy="chunks", chunk_size=10_000, residualize=True, permute_features=True, stratify="arm_outcome"),
    ),
    "cpfn_Aplus_B4k": dict(
        kind="cpfn",
        kwargs=dict(strategy="chunks", chunk_size=4_000, residualize=True, permute_features=True, stratify="arm_outcome"),
    ),
    # whole table in one context (only when it fits in GPU memory)
    "cpfn_full": dict(kind="cpfn", kwargs=dict(strategy="chunks", chunk_size=10**9, query_batch=100_000)),
    # parameter-only control: wide slices, large real context
    "cpfn_wide": dict(kind="cpfn", tau="insample", kwargs=dict(strategy="default", num_neighbours=12_000, **QINI_CTX)),
    # placebo: released rule, but queries and context rows are grouped by pure noise instead of
    # the weak learner's CATE. Tests whether the stratifier's ranking is what carries the result.
    "cpfn_placebo": dict(kind="cpfn", tau="random", kwargs=dict(strategy="default", **QINI_CTX)),
    # weak learner's estimate as a feature: no grouping by the weak learner at all; instead its CATE estimate
    # (out-of-fold for context rows) is appended as one extra covariate, with shared random chunks.
    "cpfn_K1_B10k_taufeat": dict(kind="cpfn", aug="tau", kwargs=dict(strategy="chunks", chunk_size=10_000, max_chunks=1)),
    "cpfn_A_B10k_taufeat": dict(kind="cpfn", aug="tau", kwargs=dict(strategy="chunks", chunk_size=10_000)),
    # released grouping (context rows picked by the weak learner's in-sample CATE, exactly as
    # cpfn_default) AND the cross-fitted score appended as a covariate. Completes the 2 x 2 of
    # {grouped by score, random rows} x {score hidden, score visible}. Narrow tables only: the
    # grouping scores come from the weak-learner cache, which equals the estimator's own fit there.
    "cpfn_default_taufeat": dict(kind="cpfn", tau="insample", aug="tau", kwargs=dict(strategy="default", **QINI_CTX)),
    # in-context stacking: out-of-fold CATE scores of several classical estimators (weak learner,
    # S, T, X, DA) are appended as covariates; the model combines them in context.
    "cpfn_K1_B10k_stack": dict(kind="cpfn", aug="stack", kwargs=dict(strategy="chunks", chunk_size=10_000, max_chunks=1)),
    "cpfn_A_B10k_stack": dict(kind="cpfn", aug="stack", kwargs=dict(strategy="chunks", chunk_size=10_000)),
    # same, but the model sees ONLY the scores (no raw covariates)
    "cpfn_A_B10k_stackonly": dict(kind="cpfn", aug="stackonly", kwargs=dict(strategy="chunks", chunk_size=10_000)),
    "weak": dict(kind="weak"),
    "S": dict(kind="baseline", cls="SLearnerBaseline"),
    "T": dict(kind="baseline", cls="TLearnerBaseline"),
    "X": dict(kind="baseline", cls="XLearnerBaseline"),
    "DA": dict(kind="baseline", cls="DALearnerBaseline"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=default)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **{k: v for k, v in arrays.items() if v is not None})
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# weak learner cache: in-sample tau for context, tau for queries, out-of-fold tau for context
# ----------------------------------------------------------------------------------------------
def weak_cache(path, X_tr, t_tr, y_tr, X_te, oof_folds=5, seed=0, need_oof=True):
    from strategies import crossfit_weak_tau, fit_weak, weak_tau

    path = Path(path)
    if path.exists():
        z = np.load(path)
        if (not need_oof) or "tau_ctx_oof" in z.files:
            return {k: z[k] for k in z.files}
    tic = time.time()
    obj = fit_weak(X_tr, t_tr, y_tr)
    out = dict(tau_ctx_insample=weak_tau(obj, X_tr), tau_query=weak_tau(obj, X_te))
    out["seconds_insample"] = np.array(time.time() - tic)
    if need_oof:
        tic = time.time()
        out["tau_ctx_oof"] = crossfit_weak_tau(X_tr, t_tr, y_tr, folds=oof_folds, seed=seed)
        out["seconds_oof"] = np.array(time.time() - tic)
    save_npz(path, **out)
    return out


# ----------------------------------------------------------------------------------------------
# Qini (same formula as causalpfn.evaluation.get_qini_curve; kept here so that the analysis
# runs without torch)
# ----------------------------------------------------------------------------------------------
def qini(t, y, score, normalize=True):
    order = np.argsort(score)[::-1]
    t = np.asarray(t, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]
    n_t = t.cumsum()
    n_c = (1 - t).cumsum()
    y_t = (y * t).cumsum()
    y_c = (y * (1 - t)).cumsum()
    curve = np.where(n_c != 0, y_t - y_c * (n_t / np.maximum(n_c, 1e-12)), 0.0)
    if normalize:
        curve = curve / (np.abs(curve[-1]) + 1e-8)
    return float(curve.sum() / len(curve))


def bootstrap_qini(t, y, scores: dict, n_boot=300, seed=0, normalize=True):
    """Paired bootstrap over test units. Returns {method: array(n_boot)}."""
    rng = np.random.default_rng(seed)
    n = len(t)
    out = {m: np.empty(n_boot) for m in scores}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tb, yb = t[idx], y[idx]
        for m, s in scores.items():
            out[m][b] = qini(tb, yb, s[idx], normalize=normalize)
    return out


def ci(a, lo=2.5, hi=97.5):
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]
