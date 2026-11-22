"""Turn saved predictions into tables and figures. Needs numpy + matplotlib only (no torch).

    python src/analyze.py            # everything that is available under results/

Writes results/summary/{qini_<tag>.json, decomposition_<tag>.json, poly.json, summary.md}
and figures/*.png. Every number in summary.md comes from the files under results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import bootstrap_qini, ci, qini, save_json  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
N_FOLDS = 5
N_BOOT = 300


def load_method(tag_dir, name):
    folds = []
    for f in range(N_FOLDS):
        p = tag_dir / name / f"fold{f}.npz"
        if not p.exists():
            return None
        folds.append(dict(np.load(p)))
    return folds


def rank01(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x), 1)


# ----------------------------------------------------------------------------------------------
def qini_table(tag_dir):
    methods = sorted(d.name for d in tag_dir.iterdir() if d.is_dir())
    data = {m: load_method(tag_dir, m) for m in methods}
    data = {m: v for m, v in data.items() if v is not None}
    if not data:
        return None
    ref = next(iter(data.values()))
    t = np.concatenate([f["t"] for f in ref]).astype(np.float64)
    y = np.concatenate([f["y"] for f in ref]).astype(np.float64)
    scores = {m: np.concatenate([f["tau"] for f in v]) for m, v in data.items()}

    out = dict(n_units=int(len(t)), methods={})
    for normalize, key in ((True, "qini_norm"), (False, "qini_raw")):
        boot = bootstrap_qini(t, y, scores, n_boot=N_BOOT, seed=0, normalize=normalize)
        for m, v in data.items():
            per_fold = [qini(f["t"], f["y"], f["tau"], normalize) for f in v]
            d = out["methods"].setdefault(m, {})
            d[key] = dict(
                pooled=qini(t, y, scores[m], normalize),
                pooled_ci95=ci(boot[m]),
                fold_mean=float(np.mean(per_fold)),
                fold_std=float(np.std(per_fold, ddof=1)),
                per_fold=per_fold,
            )
            for base in ("cpfn_default", "S", "weak"):
                if base in boot and base != m:
                    d[key][f"diff_vs_{base}"] = dict(
                        pooled=d[key]["pooled"] - qini(t, y, scores[base], normalize), ci95=ci(boot[m] - boot[base])
                    )
        best = max(out["methods"][m][key]["pooled"] for m in data)
        for m in data:
            out["methods"][m][key]["column_normalised"] = out["methods"][m][key]["pooled"] / best
    for m, v in data.items():
        d = out["methods"][m]
        d["seconds_per_fold"] = float(np.mean([f["seconds"] for f in v]))
        d["tau_sd"] = float(np.std(scores[m]))
        d["tau_mean"] = float(np.mean(scores[m]))
        if "peak_gpu_gb" in v[0]:
            d["peak_gpu_gb"] = float(max(f["peak_gpu_gb"] for f in v))
    out["raw_ate"] = float(y[t == 1].mean() - y[t == 0].mean())
    return out


# ----------------------------------------------------------------------------------------------
def decomposition(tag_dir, name):
    """What part of a retrieval-based prediction is the batch level and what part is the model?"""
    folds = load_method(tag_dir, name)
    if folds is None or "batch_id_t0" not in folds[0]:
        return None
    per_fold, pooled = [], {k: [] for k in ("t", "y", "pred", "weak", "level_only", "between_weak_within_pred", "between_pred_within_weak", "pred_recentred", "pred_within_only")}
    batches = []
    for fi, f in enumerate(folds):
        t, y, pred, weak = f["t"].astype(float), f["y"].astype(float), f["tau"], f["tau_weak"]
        b0, b1 = f["batch_id_t0"], f["batch_id_t1"]
        level = f["rec_level"]
        nb = len(level)
        mean_pred = np.array([pred[b0 == b].mean() if (b0 == b).any() else np.nan for b in range(nb)])
        mean_weak = np.array([weak[b0 == b].mean() if (b0 == b).any() else np.nan for b in range(nb)])
        honest, n_units = [], []
        for b in range(nb):
            m = b0 == b
            n_units.append(int(m.sum()))
            tt, yy = t[m], y[m]
            honest.append(yy[tt == 1].mean() - yy[tt == 0].mean() if (tt == 1).any() and (tt == 0).any() else np.nan)
        honest = np.array(honest)
        ok = np.isfinite(honest) & np.isfinite(mean_pred)
        within = pred - mean_pred[b0]
        variants = dict(
            pred=pred,
            weak=weak,
            level_only=level[b0] + 1e-9 * rank01(weak),
            between_weak_within_pred=b0 + 0.999 * rank01(within),
            between_pred_within_weak=rank01(mean_pred)[b0] * nb + 0.999 * rank01(weak),
            pred_recentred=within + mean_weak[b0],
            pred_within_only=within,
        )
        for k, v in variants.items():
            pooled[k].append(v)
        pooled["t"].append(t)
        pooled["y"].append(y)
        per_fold.append(
            dict(
                fold=fi,
                n_batches=int(nb),
                ctx_rows_median=float(np.median(f["rec_n_ctx_treated"] + f["rec_n_ctx_control"])),
                frac_rows_ever_used=float(f["rec_frac_rows_ever_used"]),
                n_distinct_tau_context=int(f["rec_n_distinct_tau_context"]),
                n_distinct_tau_query=int(f["rec_n_distinct_tau_query"]),
                frac_units_split_across_batches=float(np.mean(b0 != b1)),
                sd_context_level=float(np.nanstd(level)),
                sd_honest_test_diff=float(np.nanstd(honest[ok])),
                sd_batch_mean_pred=float(np.nanstd(mean_pred[ok])),
                slope_level_on_honest=float(np.polyfit(honest[ok], level[ok], 1)[0]) if ok.sum() > 2 else None,
                slope_meanpred_on_level=float(np.polyfit(level[ok], mean_pred[ok], 1)[0]) if ok.sum() > 2 else None,
                corr_meanpred_level=float(np.corrcoef(level[ok], mean_pred[ok])[0, 1]) if ok.sum() > 2 else None,
                share_pred_variance_between_batches=float(np.var(mean_pred[b0]) / (np.var(pred) + 1e-30)),
                sd_pred=float(np.std(pred)),
                sd_pred_within_batch=float(np.std(within)),
            )
        )
        batches.append(dict(fold=fi, level=level, honest=honest, mean_pred=mean_pred, mean_weak=mean_weak, n_units=n_units))

    t = np.concatenate(pooled.pop("t"))
    y = np.concatenate(pooled.pop("y"))
    scores = {k: np.concatenate(v) for k, v in pooled.items()}
    out = dict(method=name, per_fold=per_fold, batches=batches, qini_variants={})
    for normalize, key in ((True, "qini_norm"), (False, "qini_raw")):
        boot = bootstrap_qini(t, y, scores, n_boot=N_BOOT, seed=1, normalize=normalize)
        for k in scores:
            d = out["qini_variants"].setdefault(k, {})
            d[key] = dict(pooled=qini(t, y, scores[k], normalize), ci95=ci(boot[k]))
            if k != "pred":
                d[key]["diff_vs_pred_ci95"] = ci(boot[k] - boot["pred"])
    keys = [k for k in per_fold[0] if k != "fold"]
    out["mean_over_folds"] = {k: float(np.mean([p[k] for p in per_fold if p[k] is not None])) for k in keys}
    return out


# ----------------------------------------------------------------------------------------------
def poly_table():
    root = RES / "poly"
    if not root.exists():
        return None
    out = {}
    for d in sorted(root.iterdir(), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        if not d.is_dir():
            continue
        res = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
        base = res.get("cpfn_default", {}).get("pehe", [])
        row = {}
        for m, r in res.items():
            p = np.array(r["pehe"], dtype=float)
            if len(p) == 0:
                row[m] = dict(status=r.get("status"), n_tables=0)
                continue
            e = dict(mean=float(p.mean()), se=float(p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 else 0.0, n_tables=int(len(p)), status=r.get("status", "ok"))
            if len(base) >= len(p) and m != "cpfn_default":
                diff = p - np.array(base[: len(p)])
                e["diff_vs_cpfn_default"] = dict(mean=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0, frac_tables_better=float(np.mean(diff < 0)))
            if "peak_gpu_gb" in r and r["peak_gpu_gb"]:
                e["peak_gpu_gb"] = float(max(r["peak_gpu_gb"]))
            e["seconds"] = float(np.mean(r["seconds"]))
            row[m] = e
        out[d.name] = row
    return out


# ----------------------------------------------------------------------------------------------
def figures(qinis, decomps, poly):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    if poly:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sizes = [int(k[1:]) for k in poly]
        for m in sorted({m for row in poly.values() for m in row}):
            xs = [n for n, k in zip(sizes, poly) if poly[k].get(m, {}).get("n_tables", 0) > 0]
            ys = [poly[f"N{n}"][m]["mean"] for n in xs]
            es = [poly[f"N{n}"][m]["se"] for n in xs]
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1.2, capsize=2, label=m, ls="--" if m in ("S", "T", "X", "DA") else "-")
        ax.set_xscale("log")
        ax.set_xlabel("training rows N")
        ax.set_ylabel("PEHE (lower is better)")
        ax.set_title("Synthetic polynomial data (paper Table 4 setting), mean ± SE")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "poly_pehe_vs_n.png", dpi=160)
        plt.close(fig)
    for tag, q in qinis.items():
        ms = sorted(q["methods"], key=lambda m: q["methods"][m]["qini_norm"]["pooled"])
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(ms) + 1.5))
        vals = [q["methods"][m]["qini_norm"]["pooled"] for m in ms]
        lo = [v - q["methods"][m]["qini_norm"]["pooled_ci95"][0] for v, m in zip(vals, ms)]
        hi = [q["methods"][m]["qini_norm"]["pooled_ci95"][1] - v for v, m in zip(vals, ms)]
        ax.barh(ms, vals, xerr=[lo, hi], color=["#c0504d" if m.startswith("cpfn") else "#7f7f7f" for m in ms], capsize=2)
        ax.set_xlim(min(vals) - 0.05, max(vals) + 0.05)
        ax.set_xlabel("pooled Qini (repo formula), 95% bootstrap CI")
        ax.set_title(tag)
        fig.tight_layout()
        fig.savefig(FIG / f"qini_{tag}.png", dpi=160)
        plt.close(fig)
    for (tag, name), d in decomps.items():
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for b in d["batches"]:
            axes[0].scatter(100 * np.array(b["honest"]), 100 * np.array(b["level"]), s=10, alpha=0.6)
            axes[1].scatter(100 * np.array(b["level"]), 100 * np.array(b["mean_pred"]), s=10, alpha=0.6)
        for ax in axes:
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lim, lim, "k:", lw=0.8)
        axes[0].set_xlabel("honest effect of the batch's test units (pp)")
        axes[0].set_ylabel("difference in arm means inside the context (pp)")
        axes[1].set_xlabel("difference in arm means inside the context (pp)")
        axes[1].set_ylabel("mean predicted CATE of the batch (pp)")
        fig.suptitle(f"{tag} / {name}: one dot per batch, colours = folds")
        fig.tight_layout()
        fig.savefig(FIG / f"batch_levels_{tag}_{name}.png", dpi=160)
        plt.close(fig)


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_markdown(qinis, decomps, poly):
    L = ["# Results summary (generated by src/analyze.py)", ""]
    for tag, q in qinis.items():
        L += [f"## Qini on {tag} ({q['n_units']} test units pooled over 5 folds, raw ATE {100 * q['raw_ate']:.2f} pp)", "",
              "| method | pooled Qini (norm.) | 95% CI | column-normalised | fold mean ± std | raw Qini | diff vs cpfn_default (95% CI) | sd of predicted CATE (pp) | s/fold |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(q["methods"], key=lambda m: -q["methods"][m]["qini_norm"]["pooled"]):
            d = q["methods"][m]
            a = d["qini_norm"]
            dv = a.get("diff_vs_cpfn_default")
            L.append(f"| {m} | {a['pooled']:.4f} | [{a['pooled_ci95'][0]:.4f}, {a['pooled_ci95'][1]:.4f}] | {a['column_normalised']:.3f} | {a['fold_mean']:.4f} ± {a['fold_std']:.4f} | {d['qini_raw']['pooled']:.2f} | "
                     + (f"{dv['pooled']:+.4f} [{dv['ci95'][0]:+.4f}, {dv['ci95'][1]:+.4f}]" if dv else "") + f" | {100 * d['tau_sd']:.2f} | {d['seconds_per_fold']:.0f} |")
        L.append("")
    for (tag, name), d in decomps.items():
        mf = d["mean_over_folds"]
        L += [f"## Decomposition of {name} on {tag}", "", "Mean over folds:", ""]
        L += [f"- {k}: {fmt(v, 4)}" for k, v in mf.items()]
        L += ["", "| ranking | pooled Qini (norm.) | 95% CI | diff vs model prediction (95% CI) |", "|---|---|---|---|"]
        for k, v in d["qini_variants"].items():
            a = v["qini_norm"]
            dd = a.get("diff_vs_pred_ci95")
            L.append(f"| {k} | {a['pooled']:.4f} | [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] | " + (f"[{dd[0]:+.4f}, {dd[1]:+.4f}]" if dd else "") + " |")
        L.append("")
    if poly:
        ms = sorted({m for row in poly.values() for m in row})
        L += ["## Synthetic polynomial data: PEHE, mean ± SE over tables (lower is better)", "", "| method | " + " | ".join(poly) + " |", "|---|" + "---|" * len(poly)]
        for m in ms:
            cells = []
            for k in poly:
                e = poly[k].get(m)
                cells.append("" if e is None else (f"{e['mean']:.3f} ± {e['se']:.3f}" if e.get("n_tables") else str(e.get("status"))[:12]))
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L += ["", "Paired difference vs cpfn_default (the released behaviour) (negative = better), mean ± SE, share of tables improved:", "", "| method | " + " | ".join(poly) + " |", "|---|" + "---|" * len(poly)]
        for m in ms:
            cells = []
            for k in poly:
                e = (poly[k].get(m) or {}).get("diff_vs_cpfn_default")
                cells.append("" if e is None else f"{e['mean']:+.3f} ± {e['se']:.3f} ({100 * e['frac_tables_better']:.0f}%)")
            if any(cells):
                L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    (RES / "summary").mkdir(parents=True, exist_ok=True)
    (RES / "summary" / "summary.md").write_text("\n".join(L), encoding="utf8")


def _one_table(tag_dir):
    q = qini_table(tag_dir)
    ds = {}
    for name in ("cpfn_default", "cpfn_xfit", "cpfn_wide"):
        d = decomposition(tag_dir, name)
        if d:
            ds[name] = d
    return tag_dir.name, q, ds


def main():
    from concurrent.futures import ProcessPoolExecutor

    qinis, decomps = {}, {}
    qroot = RES / "qini"
    if qroot.exists():
        dirs = sorted(p for p in qroot.iterdir() if p.is_dir())
        with ProcessPoolExecutor(max_workers=min(len(dirs), 6)) as ex:  # one table per process
            for tag, q, ds in ex.map(_one_table, dirs):
                if q:
                    qinis[tag] = q
                    save_json(RES / "summary" / f"qini_{tag}.json", q)
                for name, d in ds.items():
                    decomps[(tag, name)] = d
                    save_json(RES / "summary" / f"decomposition_{tag}_{name}.json", d)
    poly = poly_table()
    if poly:
        save_json(RES / "summary" / "poly.json", poly)
    write_markdown(qinis, decomps, poly)
    try:
        figures(qinis, decomps, poly)
    except Exception as e:  # figures are optional
        print("figure error:", repr(e))
    print((RES / "summary" / "summary.md").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
