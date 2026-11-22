"""GPU smoke test. Run this before anything else on a new machine.

1. Equivalence: StrategyCATE(strategy="default") must reproduce the untouched repo
   CATEEstimator on the same data (same selection rule, same forward passes).
2. Chunk strategy runs and one chunk holding the whole table equals a plain full-context pass.
3. Peak GPU memory and time of one forward pass at several context lengths.

    python src/smoke_test.py --out results/smoke_test.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "CausalPFN"))

from causalpfn import CATEEstimator  # noqa: E402

from common import log, save_json  # noqa: E402
from strategies import StrategyCATE  # noqa: E402


def synthetic_rct(n, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    base = 0.6 + 0.1 * np.tanh(X[:, 0]) - 0.08 * X[:, 1] * (X[:, 2] > 0)
    tau = 0.04 + 0.03 * np.tanh(X[:, 3]) + 0.01 * (X[:, 4] > 1)
    t = (rng.random(n) < 0.5).astype(np.float32)
    y = (rng.random(n) < np.clip(base + t * tau, 0.01, 0.99)).astype(np.float32)
    return X, t, y, tau.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results" / "smoke_test.json"))
    p.add_argument("--lengths", default="4096,10000,16000,25000,32000,40000,50000")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = dict(device=device, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)

    # 1. equivalence with the repo -----------------------------------------------------------
    X, t, y, _ = synthetic_rct(12_000, seed=1)
    X_tr, t_tr, y_tr, X_te = X[:10_000], t[:10_000], y[:10_000], X[10_000:]
    checks = {}
    for label, kw in [("defaults", {}), ("max_ctx_50k", dict(max_context_length=50_000))]:
        ref = CATEEstimator(device=device, **kw)
        ref.fit(X_tr, t_tr, y_tr)
        a = ref.estimate_cate(X_te)
        mine = StrategyCATE(device=device, strategy="default", **kw)
        mine.fit(X_tr, t_tr, y_tr)
        b = mine.estimate_cate(X_te)
        checks[label] = dict(
            max_abs_diff=float(np.max(np.abs(a - b))),
            corr=float(np.corrcoef(a, b)[0, 1]),
            n_batches=int(mine.record["n_batches"]),
            ctx_rows=(mine.record["n_ctx_treated"] + mine.record["n_ctx_control"]).tolist(),
            tau_mean_repo=float(a.mean()),
            tau_sd_repo=float(a.std()),
            output_shape=list(np.shape(a)),
        )
        log(f"equivalence [{label}]: max|diff|={checks[label]['max_abs_diff']:.2e}, batches={checks[label]['n_batches']}")
    report["equivalence_with_repo"] = checks

    # 2. chunks ------------------------------------------------------------------------------
    full = StrategyCATE(device=device, strategy="chunks", chunk_size=10**9)
    full.fit(X_tr, t_tr, y_tr)
    tau_full = full.estimate_cate(X_te)
    ref = CATEEstimator(device=device, max_context_length=20_000, num_neighbours=10_000)  # argument-only full context
    ref.fit(X_tr, t_tr, y_tr)
    tau_ref = ref.estimate_cate(X_te)
    a4 = StrategyCATE(device=device, strategy="chunks", chunk_size=2_500)
    a4.fit(X_tr, t_tr, y_tr)
    tau_a4 = a4.estimate_cate(X_te)
    report["chunks"] = dict(
        full_vs_argument_only_full_max_abs_diff=float(np.max(np.abs(tau_full - tau_ref))),
        full_vs_argument_only_full_corr=float(np.corrcoef(tau_full, tau_ref)[0, 1]),
        A_4chunks_vs_full_corr=float(np.corrcoef(tau_a4, tau_full)[0, 1]),
        n_chunks=int(a4.record["n_chunks"]),
        chunk_sizes=a4.record["chunk_sizes"].tolist(),
    )
    log(f"chunks: {report['chunks']}")

    # 3. memory / time per forward pass ----------------------------------------------------------
    mem = []
    if device == "cuda":
        lengths = [int(v) for v in args.lengths.split(",")]
        Xb, tb, yb, _ = synthetic_rct(max(lengths) + 16384, seed=2)
        model = full.icl_model
        temp = torch.tensor([1.0], device=device)
        for c in lengths:
            for q in (4096, 16384):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    tic = time.time()
                    with torch.no_grad():
                        model.predict_cepo(
                            X_context=torch.from_numpy(Xb[:c]).to(device).unsqueeze(0),
                            t_context=torch.from_numpy(tb[:c]).to(device).unsqueeze(0),
                            y_context=torch.from_numpy(yb[:c]).to(device).unsqueeze(0),
                            X_query=torch.from_numpy(Xb[-q:]).to(device).unsqueeze(0),
                            t_query=torch.from_numpy(tb[-q:]).to(device).unsqueeze(0),
                            temperature=temp,
                            n_samples=None,
                        )
                    torch.cuda.synchronize()
                    row = dict(context=c, queries=q, seconds=time.time() - tic, peak_gb=torch.cuda.max_memory_allocated() / 2**30, ok=True)
                except torch.cuda.OutOfMemoryError:
                    row = dict(context=c, queries=q, ok=False)
                    torch.cuda.empty_cache()
                mem.append(row)
                log(f"forward: {row}")
    report["forward_pass"] = mem
    save_json(args.out, report)
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
