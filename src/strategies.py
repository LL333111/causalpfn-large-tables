"""Context-construction strategies for CausalPFN.

The cloned CausalPFN repo is not modified. `StrategyCATE` subclasses `CATEEstimator` and
overrides `fit` and `_predict_cepo` only.

strategy="default"
    Re-implementation of the released selection rule (vdblm/CausalPFN @ b7ed2f1,
    `CausalEstimator._predict_cepo`, Apache-2.0) with bookkeeping added: batch id of every
    query, context size per arm, the per-batch difference in arm means of the context
    ("level"), and which context rows were ever used. `src/smoke_test.py` checks that its
    output equals the original `CATEEstimator` on the same data.
    Options: `tau_context` / `tau_query` (precomputed weak-learner CATE, e.g. cross-fitted
    for the context rows), `crossfit_folds`, `jitter` (tie breaking).

strategy="chunks"
    The context rows are split into K disjoint random chunks (stratified by arm, optionally
    by outcome), every query is predicted with every chunk, the K predictions are averaged.
    `max_chunks=1` gives the single-random-context control. `chunk_size >= N` gives one
    chunk holding the whole table (full context).
    Options: `permute_features` (a different feature order per chunk), `residualize`
    (outcome minus a cross-fitted E[y|x] model before anything else).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from causalpfn import CATEEstimator
from causalpfn._nearest_neighbors import nearest_indices_1d


# ----------------------------------------------------------------------------------------------
# weak learner helpers (same code path as the repo, without loading the transformer)
# ----------------------------------------------------------------------------------------------
def fit_weak(X, t, y):
    """Train the repo's stratifier on (X, t, y) and return the fitted helper object.

    With CPFN_STRATIFIER=hist in the environment (set by `run_qini.py --stratifier hist`) the same
    S-learner is fitted with scikit-learn's histogram-based gradient boosting instead of the
    repo's exact-split GradientBoostingRegressor: same design matrix, same number of trees, depth,
    minimum leaf size and seed. It is a different implementation, so the scores are not identical.
    """
    obj = CATEEstimator.__new__(CATEEstimator)
    if os.environ.get("CPFN_STRATIFIER", "repo") == "hist":
        obj.t_transformer = OneHotEncoder(sparse_output=False, categories="auto", drop="first")
        T = obj.t_transformer.fit_transform(t.reshape(-1, 1))
        obj._d_t = (T.shape[1],)
        feat = np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
        obj.stratifier = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            max_leaf_nodes=None,
            min_samples_leaf=max(2, int(X.shape[0] / 100)),
            learning_rate=0.1,
            early_stopping=False,
            random_state=111,
        )
        obj.stratifier.fit(feat, y)
        return obj
    obj._train_weak_learner(X, t, y)
    return obj


def weak_tau(obj, X):
    return obj._estimate_cate_weak_learner(X_test=X)


def crossfit_weak_tau(X, t, y, folds=5, seed=0):
    """Out-of-fold weak-learner CATE for every row of (X, t, y)."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for tr, te in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        out[te] = weak_tau(fit_weak(X[tr], t[tr], y[tr]), X[te])
    return out


def crossfit_outcome_model(X, y, folds=5, seed=0):
    """Out-of-fold prediction of E[y | x] (treatment not used)."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for tr, te in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
        model.fit(X[tr], y[tr])
        out[te] = model.predict(X[te])
    return out


# ----------------------------------------------------------------------------------------------
def make_chunks(t, y, chunk_size, rng, stratify="arm", max_chunks=None):
    """Disjoint random chunks with the same strata proportions in every chunk."""
    n = len(t)
    t = np.asarray(t)
    strata = (t > 0.5).astype(np.int64)
    if stratify == "arm_outcome":
        y = np.asarray(y, dtype=np.float64)
        if len(np.unique(y)) <= 2:
            ybin = (y > y.min()).astype(np.int64)
        else:
            ybin = np.searchsorted(np.quantile(y, [0.25, 0.5, 0.75]), y)
        strata = strata * 4 + ybin
    elif stratify != "arm":
        raise ValueError(stratify)

    if max_chunks is not None and max_chunks * chunk_size < n:
        k = max_chunks
        keep_frac = max_chunks * chunk_size / n
    else:
        k = max(1, n // chunk_size)
        keep_frac = 1.0

    chunk_of = np.full(n, -1, dtype=np.int64)
    for s in np.unique(strata):
        idx = rng.permutation(np.where(strata == s)[0])
        if keep_frac < 1.0:
            idx = idx[: int(round(len(idx) * keep_frac))]
        chunk_of[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    return [np.where(chunk_of == c)[0] for c in range(k)]


def arm_level(t, y):
    t = np.asarray(t)
    y = np.asarray(y, dtype=np.float64)
    m1 = y[t > 0.5].mean() if (t > 0.5).any() else np.nan
    m0 = y[t <= 0.5].mean() if (t <= 0.5).any() else np.nan
    return float(m1 - m0)


# ----------------------------------------------------------------------------------------------
class StrategyCATE(CATEEstimator):
    def __init__(
        self,
        device,
        strategy: str = "default",
        chunk_size: int = 10_000,
        max_chunks: int | None = None,
        stratify: str = "arm",
        permute_features: bool = False,
        residualize: bool = False,
        crossfit_folds: int = 0,
        jitter: float = 0.0,
        query_batch: int = 16384,
        seed: int = 0,
        tau_context: np.ndarray | None = None,
        tau_query: np.ndarray | None = None,
        keep_last_cols: int = 0,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        assert strategy in ("default", "chunks")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.stratify = stratify
        self.permute_features = permute_features
        self.residualize = residualize
        self.crossfit_folds = crossfit_folds
        self.jitter = jitter
        self.query_batch = query_batch
        self.seed = seed
        self.tau_context = tau_context
        self.tau_query = tau_query
        self.keep_last_cols = keep_last_cols
        self._wide = None
        self.stratifier = None
        self.t_transformer = None
        self.record: dict = {}

    # -- fit ------------------------------------------------------------------------------
    def _needs_weak_learner(self) -> bool:
        return self.strategy == "default" and (self.tau_context is None or self.tau_query is None)

    def _train_weak_learner(self, X, t, y):
        if self._needs_weak_learner():
            super()._train_weak_learner(X, t, y)

    def _reduce(self, X, fit=False):
        """Wide tables with appended score columns (keep_last_cols > 0): the repo would run its
        SVD over all columns and mix the scores into the components. Here the SVD is fitted on the
        original covariates only and the last `keep_last_cols` columns are passed through.
        Tables that fit into the model are returned unchanged."""
        k = self.keep_last_cols
        if not k:
            return X
        if fit:
            self.load_model()
            self._wide = None
            if self.max_feature_size is not None and X.shape[1] > self.max_feature_size:
                from sklearn.decomposition import TruncatedSVD

                self._wide = TruncatedSVD(n_components=self.max_feature_size - k, algorithm="arpack").fit(X[:, :-k])
        if self._wide is None:
            return X
        return np.column_stack([self._wide.transform(X[:, :-k]), X[:, -k:]]).astype(np.float32)

    def estimate_cate(self, X):
        return super().estimate_cate(self._reduce(np.asarray(X)))

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.record = {}
        X = self._reduce(X, fit=True)
        if self.residualize:
            tic = time.time()
            m_hat = crossfit_outcome_model(X, y, folds=5, seed=self.seed)
            y = (y - m_hat).astype(np.float32)
            self.record["residualize_seconds"] = time.time() - tic
            self.record["outcome_model_r2"] = float(1.0 - y.var() / (y + m_hat).var())
        super().fit(X, t, y)
        # Note for wide tables: a precomputed tau_hat (tau_context / tau_query) is used as given.
        # The repo fits its weak learner on the SVD components, so to reproduce the released
        # method on a wide table leave tau_context=None (run_qini.py --inline-weak).
        self._tau_ctx_oof = None
        if self.strategy == "default" and self.crossfit_folds and self.tau_context is None:
            self._tau_ctx_oof = crossfit_weak_tau(self.X_train, self.t_train, self.y_train, self.crossfit_folds, self.seed)
        return self

    # -- weak-learner CATE used for selection ------------------------------------------------
    def _context_tau(self, X_context):
        if self.tau_context is not None:
            assert len(self.tau_context) == X_context.shape[0]
            return np.asarray(self.tau_context, dtype=np.float64)
        if self._tau_ctx_oof is not None:
            return self._tau_ctx_oof
        return self._estimate_cate_weak_learner(X_test=X_context)

    def _query_tau(self, X_query):
        if self.tau_query is not None:
            tq = np.asarray(self.tau_query, dtype=np.float64)
            reps, rem = divmod(X_query.shape[0], len(tq))
            assert rem == 0 and reps in (1, 2), "tau_query must have one value per test unit"
            return np.tile(tq, reps)
        return self._estimate_cate_weak_learner(X_test=X_query)

    # -- prediction -------------------------------------------------------------------------
    @torch.no_grad()
    def _predict_cepo(self, X_context, t_context, y_context, X_query, t_query, temperature, n_samples=None):
        if self.icl_model is None:
            raise ValueError("fit() first")
        if n_samples is not None:
            raise NotImplementedError("StrategyCATE implements point estimates only")
        self.icl_model.eval()
        temp = torch.tensor([temperature], device=self.device)
        if self.strategy == "chunks":
            return self._predict_chunks(X_context, t_context, y_context, X_query, t_query, temp)
        return self._predict_default(X_context, t_context, y_context, X_query, t_query, temp)

    def _forward(self, x_c, t_c, y_c, x_q, t_q, temp):
        res = self.icl_model.predict_cepo(
            X_context=x_c,
            t_context=t_c,
            y_context=y_c,
            X_query=torch.from_numpy(np.ascontiguousarray(x_q)).to(self.device).unsqueeze(0).float(),
            t_query=torch.from_numpy(np.ascontiguousarray(t_q)).to(self.device).unsqueeze(0).float(),
            n_samples=None,
            temperature=temp,
        )
        return res.squeeze(0).squeeze(0).cpu().numpy()

    def _to_ctx(self, x_c, t_c, y_c):
        return (
            torch.from_numpy(np.ascontiguousarray(x_c)).to(self.device).unsqueeze(0).float(),
            torch.from_numpy(np.ascontiguousarray(t_c)).to(self.device).unsqueeze(0).float(),
            torch.from_numpy(np.ascontiguousarray(y_c)).to(self.device).unsqueeze(0).float(),
        )

    # .. random chunks, averaged ............................................................
    def _predict_chunks(self, X_context, t_context, y_context, X_query, t_query, temp):
        rng = np.random.default_rng(self.seed)
        chunks = make_chunks(t_context, y_context, self.chunk_size, rng, self.stratify, self.max_chunks)
        acc = np.zeros(X_query.shape[0], dtype=np.float64)
        levels, sizes = [], []
        n_feat = X_context.shape[1]
        for idx in chunks:
            perm = rng.permutation(n_feat) if self.permute_features else np.arange(n_feat)
            ctx = self._to_ctx(X_context[idx][:, perm], t_context[idx], y_context[idx])
            for s in range(0, X_query.shape[0], self.query_batch):
                e = min(s + self.query_batch, X_query.shape[0])
                acc[s:e] += self._forward(*ctx, X_query[s:e][:, perm], t_query[s:e], temp)
            levels.append(arm_level(t_context[idx], y_context[idx]))
            sizes.append(len(idx))
            del ctx
        self.record.update(
            n_chunks=len(chunks), chunk_sizes=np.array(sizes), chunk_levels=np.array(levels), n_forward=None
        )
        return (acc / len(chunks)).astype(X_query.dtype if X_query.dtype.kind == "f" else np.float32)

    # .. released rule, with bookkeeping .........................................................
    def _predict_default(self, X_context, t_context, y_context, X_query, t_query, temp):
        n_q = X_query.shape[0]
        all_cepo = np.zeros((n_q,), dtype=X_query.dtype if X_query.dtype.kind == "f" else np.float32)

        context_effects = np.asarray(self._context_tau(X_context), dtype=np.float64)
        query_effects = np.asarray(self._query_tau(X_query), dtype=np.float64)
        if self.jitter > 0:
            rng = np.random.default_rng(self.seed)
            scale = self.jitter * (np.std(context_effects) + 1e-12)
            context_effects = context_effects + rng.normal(0, scale, context_effects.shape)
            query_effects = query_effects + rng.normal(0, scale, query_effects.shape)

        treat_idx = np.where(np.isclose(t_context, 1))[0]
        ctrl_idx = np.where(np.isclose(t_context, 0))[0]
        query_sorted = np.argsort(query_effects)

        k = min(self.num_neighbours, self.max_context_length // 2)
        nb_t = nearest_indices_1d(context_effects[treat_idx], query_effects, k=k)
        nb_c = nearest_indices_1d(context_effects[ctrl_idx], query_effects, k=k)

        batch_id = np.full(n_q, -1, dtype=np.int32)
        used = np.zeros(X_context.shape[0], dtype=bool)
        rec = dict(n_queries=[], n_ctx_treated=[], n_ctx_control=[], level=[], ctx_tau_mean=[], query_tau_mean=[])
        start, b = 0, 0
        limit = self.max_context_length // 2
        while start < n_q:

            def get_stratum(neighbours):
                L = start + 1
                R = min(start + self.max_query_length, n_q)
                while R > L + 1:
                    mid = (L + R) // 2
                    if len(np.unique(neighbours[query_sorted[start:mid]].flatten())) > limit:
                        R = mid - 1
                    else:
                        L = mid
                return min(L, n_q)

            c_end = get_stratum(nb_c)
            real_c = ctrl_idx[np.unique(nb_c[query_sorted[start:c_end]].flatten())]
            t_end = get_stratum(nb_t)
            real_t = treat_idx[np.unique(nb_t[query_sorted[start:t_end]].flatten())]

            end = min(c_end, t_end)
            idx_q = query_sorted[start:end]
            ctx_idx = np.concatenate([real_c, real_t])
            assert len(ctx_idx) <= self.max_context_length

            ctx = self._to_ctx(X_context[ctx_idx], t_context[ctx_idx], y_context[ctx_idx])
            all_cepo[idx_q] = self._forward(*ctx, X_query[idx_q], t_query[idx_q], temp)
            del ctx

            batch_id[idx_q] = b
            used[ctx_idx] = True
            rec["n_queries"].append(end - start)
            rec["n_ctx_treated"].append(len(real_t))
            rec["n_ctx_control"].append(len(real_c))
            rec["level"].append(arm_level(t_context[ctx_idx], y_context[ctx_idx]))
            rec["ctx_tau_mean"].append(float(context_effects[ctx_idx].mean()))
            rec["query_tau_mean"].append(float(query_effects[idx_q].mean()))
            start, b = end, b + 1

        self.record.update({key: np.array(val) for key, val in rec.items()})
        self.record.update(
            batch_id=batch_id,
            n_batches=b,
            frac_rows_ever_used=float(used.mean()),
            n_distinct_tau_context=int(len(np.unique(context_effects))),
            n_distinct_tau_query=int(len(np.unique(query_effects))),
            query_tau=query_effects,
        )
        return all_cepo
