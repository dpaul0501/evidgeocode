#!/usr/bin/env python
"""Turn run_eval.py outputs into every table and number used in the paper.

    python scripts/analyze.py --run-dir <out of run_eval.py> [--grid 7]

Writes <run-dir>/results/:
  *.csv           one file per analysis
  numbers.json    every scalar quoted in the text
  numbers.tex     the same as \\EV{key} macros, so the paper never hand-copies a number

Design choices (all to avoid the problems raised in review):
  * guidance effects are paired by prompt (same gid across modes) with
    Holm correction over every test reported;
  * classifiers use GroupKFold by real-image row, so a caption never sits in
    both train and test folds; chance = majority-class rate;
  * nothing is filtered or subsampled silently: every n is reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidgeo import metrics as M  # noqa: E402

SIGNALS = ["pec", "rollout", "ss", "loo", "sal"]
PAIRS = [("text", "weak"), ("i2i", "weak"), ("text", "i2i")]
NUM: dict = {}


# ---------------------------------------------------------------- helpers
def holm(p: pd.Series) -> pd.Series:
    p = p.to_numpy(dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if ok.any():
        idx = np.argsort(p[ok])
        m = ok.sum()
        adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * p[ok][idx]))
        tmp = np.empty(m)
        tmp[idx] = adj
        out[ok] = tmp
    return pd.Series(out)


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    n = len(d)
    if n < 3:
        return dict(n=n)
    t = stats.ttest_1samp(d, 0.0)
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    return dict(n=n, mean_a=float(a[d.index].mean()), mean_b=float(b[d.index].mean()),
                diff=float(d.mean()), dz=float(d.mean() / (d.std(ddof=1) + 1e-12)),
                t=float(t.statistic), p_t=float(t.pvalue),
                p_wilcoxon=float(w.pvalue) if w else 1.0, frac_pos=float((d > 0).mean()))


def put(key: str, val):
    NUM[key] = val


def fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", "{,}")
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        if 0 < abs(v) < 1e-3:
            return f"{v:.1e}"
        return f"{v:.3f}"
    return str(v)


# ------------------------------------------------------------------ load
def load(run: str, g: int):
    df = pd.concat([pd.read_parquet(p) for p in glob(f"{run}/features/**/*.parquet", recursive=True)],
                   ignore_index=True)
    df["key"] = (df.corpus + "/" + df.dataset + "/" + df.model + "/" + df["mode"] + "/" +
                 df.gid.astype(str) + "/" + df.seed.astype(str))
    store = {}
    for p in glob(f"{run}/maps/**/*.npz", recursive=True):
        z = np.load(p)
        for i, k in enumerate(z["keys"]):
            store[str(k)] = {f: z[f][i] for f in z.files if f != "keys"}
    rename = {c: c[len(f"g{g}_"):] for c in df.columns if c.startswith(f"g{g}_")}
    return df.rename(columns=rename), store


# -------------------------------------------------------------- analyses
def corpus_summary(df, out):
    t = df.groupby(["corpus", "dataset", "model", "mode"]).size().rename("n").reset_index()
    t.to_csv(f"{out}/corpus_summary.csv", index=False)
    gen = df[df.corpus == "guidance"]
    put("n_total", int(len(df)))
    put("n_guidance", int(len(gen)))
    put("n_consistency", int((df.corpus == "consistency").sum()))
    put("n_real", int((df.corpus == "real").sum()))
    put("n_prompts_guidance", int(gen.hf_row.nunique()))
    for (ds, m, mo), n in gen.groupby(["dataset", "model", "mode"]).size().items():
        put(f"n_{ds}_{m}_{mo}", int(n))
    return t


def bucket_table(df, out):
    rows = []
    sub = df[df.corpus.isin(["guidance", "real"])]
    for (ds, m, mo), grp in sub.groupby(["dataset", "model", "mode"]):
        r = dict(dataset=ds, model=m, mode=mo, n=len(grp))
        for s in SIGNALS + ["clipscore"]:
            col = f"{s}_gini" if s != "clipscore" else "clipscore"
            if col in grp:
                x = grp[col].to_numpy(float)
                lo, hi = boot_ci(x)
                r.update({f"{s}_mean": np.nanmean(x), f"{s}_sd": np.nanstd(x, ddof=1),
                          f"{s}_lo": lo, f"{s}_hi": hi})
        if "rollout_entropy" in grp:
            r["pec_entropy_mean"] = grp["pec_entropy"].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/bucket_table.csv", index=False)
    return t


def guidance_tests(df, out):
    gen = df[df.corpus == "guidance"]
    rows = []
    for (ds, m), grp in gen.groupby(["dataset", "model"]):
        for s in SIGNALS:
            col = f"{s}_gini"
            if col not in grp:
                continue
            wide = grp.pivot_table(index="gid", columns="mode", values=col)
            for a, b in PAIRS:
                if a in wide and b in wide:
                    rows.append(dict(dataset=ds, model=m, signal=s, a=a, b=b,
                                     **paired(wide[a], wide[b])))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t["p_t"])
        t["p_wilcoxon_holm"] = holm(t["p_wilcoxon"])
        for _, r in t.iterrows():
            k = f"{r.dataset}_{r.model}_{r.signal}_{r.a}_vs_{r.b}"
            put(f"{k}_diff", r["diff"])
            put(f"{k}_dz", r.dz)
            put(f"{k}_p", r.p_holm)
            put(f"{k}_n", int(r.n))
    t.to_csv(f"{out}/guidance_tests.csv", index=False)
    return t


def signal_agreement(df, gtests, out):
    gen = df[df.corpus == "guidance"]
    rows = []
    for a, b in combinations([s for s in SIGNALS if f"{s}_gini" in gen], 2):
        x, y = gen[f"{a}_gini"], gen[f"{b}_gini"]
        ok = x.notna() & y.notna()
        rho = stats.spearmanr(x[ok], y[ok])
        r = stats.pearsonr(x[ok], y[ok])
        rows.append(dict(a=a, b=b, n=int(ok.sum()), spearman=rho.statistic, p_s=rho.pvalue,
                         pearson=r.statistic, p_r=r.pvalue))
        put(f"agree_{a}_{b}_rho", float(rho.statistic))
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/signal_agreement.csv", index=False)
    if len(gtests):  # do the signals agree on the direction of each guidance contrast?
        sign = gtests.pivot_table(index=["dataset", "model", "a", "b"], columns="signal",
                                  values="diff").apply(np.sign)
        sign.to_csv(f"{out}/signal_direction.csv")
        main = [s for s in ("pec", "rollout", "ss") if s in sign]
        if main:
            put("direction_agree_frac", float((sign[main].nunique(axis=1) == 1).mean()))
    return t


def prompt_dependence(df, out):
    g = df[df.corpus == "guidance"]
    rows = []
    for col in ("pec_swap_rho", "pec_sal_rho"):
        if col in g:
            x = g[col].dropna()
            rows.append(dict(measure=col, n=len(x), mean=x.mean(), median=x.median(),
                             q25=x.quantile(.25), q75=x.quantile(.75)))
            put(f"{col}_mean", float(x.mean()))
            put(f"{col}_n", int(len(x)))
    pd.DataFrame(rows).to_csv(f"{out}/prompt_dependence.csv", index=False)


def clipscore_relation(df, out):
    g = df[(df.corpus == "guidance") & df.clipscore.notna()]
    rows = []
    for s in ("pec", "ss", "rollout"):
        if f"{s}_gini" not in g:
            continue
        r = stats.pearsonr(g.clipscore, g[f"{s}_gini"])
        within = [stats.pearsonr(x.clipscore, x[f"{s}_gini"]).statistic
                  for _, x in g.groupby(["dataset", "model", "mode"]) if len(x) > 10]
        rows.append(dict(signal=s, n=len(g), pearson=r.statistic, p=r.pvalue,
                         r2=r.statistic ** 2, within_bucket_median_r=np.median(within)))
        put(f"clipscore_{s}_r", float(r.statistic))
        put(f"clipscore_{s}_r2", float(r.statistic ** 2))
    pd.DataFrame(rows).to_csv(f"{out}/clipscore_relation.csv", index=False)


def smoothing(df, out):
    g = df[df.corpus == "guidance"]
    if "ss_gini" not in g:
        return
    rows = []
    for name in ["loo", "ss"] + [c[:-5] for c in g.columns if c.startswith("gauss") and c.endswith("_gini")]:
        r = dict(map=name)
        if f"{name}_peak_shift" in g:
            r["peak_shift"] = g[f"{name}_peak_shift"].mean()
        if f"{name}_leak" in g:
            r["mass_leak_unnormalised"] = g[f"{name}_leak"].mean()
        dzs = []
        for (_, _), grp in g.groupby(["dataset", "model"]):
            w = grp.pivot_table(index="gid", columns="mode", values=f"{name}_gini")
            if "text" in w and "weak" in w:
                dzs.append(paired(w["text"], w["weak"]).get("dz", np.nan))
        r["text_vs_weak_dz_mean"] = np.nanmean(dzs) if dzs else np.nan
        r["text_vs_weak_dz_min"] = np.nanmin(dzs) if dzs else np.nan
        for kind in ("del", "ins"):
            col = f"{name}_{kind}_auc"
            if col in g:
                r[f"{kind}_auc"] = g[col].mean()
                r[f"{kind}_auc_n"] = int(g[col].notna().sum())
        rows.append(r)
        for k, v in r.items():
            if k != "map":
                put(f"smooth_{name}_{k}", float(v) if not isinstance(v, int) else v)
    t = pd.DataFrame(rows)
    # faithfulness: paired comparisons on the same images
    fa = []
    for a, b in [("ss", "loo"), ("ss", "gauss1"), ("ss", "gauss2"), ("pec", "random"),
                 ("rollout", "random"), ("ss", "random"), ("loo", "random")]:
        for kind in ("del", "ins"):
            ca, cb = f"{a}_{kind}_auc", f"{b}_{kind}_auc"
            if ca in g and cb in g:
                res = paired(g[ca], g[cb])
                fa.append(dict(a=a, b=b, kind=kind, **res))
                put(f"faith_{kind}_{a}_vs_{b}_diff", res.get("diff", np.nan))
                put(f"faith_{kind}_{a}_vs_{b}_p", res.get("p_wilcoxon", np.nan))
    t.to_csv(f"{out}/smoothing.csv", index=False)
    pd.DataFrame(fa).to_csv(f"{out}/faithfulness.csv", index=False)
    sweep = [c for c in g.columns if c.startswith("ss") and c.endswith("_gini") and c[2:-5].isdigit()]
    rob = [dict(alpha=int(c[2:-5]) / 100, mean_abs_dgini=(g[c] - g.ss_gini).abs().mean(),
                max_abs_dgini=(g[c] - g.ss_gini).abs().max()) for c in sweep]
    pd.DataFrame(rob).to_csv(f"{out}/alpha_sweep.csv", index=False)


def grid_ablation(run, out):
    df = pd.concat([pd.read_parquet(p) for p in glob(f"{run}/features/guidance/**/*.parquet",
                                                     recursive=True)], ignore_index=True)
    grids = sorted({int(c.split("_")[0][1:]) for c in df.columns if c[0] == "g" and c[1].isdigit()})
    rows = []
    for gr in grids:
        for s in ("pec", "ss", "rollout"):
            col = f"g{gr}_{s}_gini"
            if col not in df:
                continue
            for (ds, m), grp in df.groupby(["dataset", "model"]):
                w = grp.pivot_table(index="gid", columns="mode", values=col)
                if "text" in w and "weak" in w:
                    rows.append(dict(grid=gr, signal=s, dataset=ds, model=m,
                                     **paired(w["text"], w["weak"])))
    pd.DataFrame(rows).to_csv(f"{out}/grid_ablation.csv", index=False)


def stability(df, store, out):
    c = df[df.corpus == "consistency"]
    if c.empty:
        return
    rows = []
    for (mode, pid), grp in c.groupby(["mode", "gid"]):
        if len(grp) < 2:
            continue
        emb = np.stack([store[k]["clip_emb"] for k in grp.key])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        ss = [store[k]["g7_ss"] if "g7_ss" in store[k] else None for k in grp.key]
        jsds = [M.jsd(a, b) for a, b in combinations(ss, 2)] if ss[0] is not None else [np.nan]
        rows.append(dict(mode=mode, gid=pid, n_seeds=len(grp),
                         sem_var=float(((emb - emb.mean(0)) ** 2).sum(1).mean()),
                         pec_gini_mean=grp.pec_gini.mean(), pec_gini_var=grp.pec_gini.var(ddof=1),
                         ss_gini_mean=grp.ss_gini.mean() if "ss_gini" in grp else np.nan,
                         flow_jsd=float(np.mean(jsds))))
    per = pd.DataFrame(rows)
    per.to_csv(f"{out}/stability_perprompt.csv", index=False)
    summ = per.groupby("mode")[["sem_var", "pec_gini_var", "flow_jsd", "pec_gini_mean"]].agg(
        ["mean", "std", "count"])
    summ.to_csv(f"{out}/stability_by_mode.csv")
    for mode in per["mode"].unique():
        for m in ("sem_var", "flow_jsd", "pec_gini_mean"):
            put(f"stab_{mode}_{m}", float(per.loc[per["mode"] == mode, m].mean()))
    tests = []
    for a, b in PAIRS:
        for m in ("sem_var", "flow_jsd", "pec_gini_var"):
            w = per.pivot_table(index="gid", columns="mode", values=m)
            if a in w and b in w:
                res = paired(w[a], w[b])
                tests.append(dict(measure=m, a=a, b=b, **res))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t["p_wilcoxon"])
        for _, r in t.iterrows():
            put(f"stab_{r.measure}_{r.a}_vs_{r.b}_p", float(r.p_holm))
    t.to_csv(f"{out}/stability_tests.csv", index=False)
    # concentration-stability relation: within mode, and pooled with mode fixed effects
    corr = []
    for mode, grp in per.groupby("mode"):
        r = stats.spearmanr(grp.pec_gini_mean, grp.sem_var)
        corr.append(dict(scope=mode, n=len(grp), spearman=r.statistic, p=r.pvalue))
        put(f"duality_{mode}_rho", float(r.statistic))
        put(f"duality_{mode}_p", float(r.pvalue))
    try:
        import statsmodels.formula.api as smf

        fit = smf.ols("sem_var ~ pec_gini_mean + C(mode)", data=per).fit()
        corr.append(dict(scope="pooled_ols_mode_fe", n=int(fit.nobs),
                         beta=fit.params["pec_gini_mean"], p=fit.pvalues["pec_gini_mean"]))
        put("duality_pooled_beta", float(fit.params["pec_gini_mean"]))
        put("duality_pooled_p", float(fit.pvalues["pec_gini_mean"]))
    except ImportError:
        pass
    pd.DataFrame(corr).to_csv(f"{out}/duality.csv", index=False)


def real_vs_generated(df, out):
    real = df[df.corpus == "real"].set_index("hf_row")
    gen = df[df.corpus == "guidance"]
    if real.empty:
        return
    rows = []
    for (ds, m, mo), grp in gen.groupby(["dataset", "model", "mode"]):
        grp = grp.set_index("hf_row")
        idx = grp.index.intersection(real.index)
        for s in ("pec", "ss", "rollout"):
            res = paired(grp.loc[idx, f"{s}_gini"], real.loc[idx, f"{s}_gini"])
            rows.append(dict(dataset=ds, model=m, mode=mo, signal=s, **res))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p_wilcoxon"])
    t.to_csv(f"{out}/real_vs_generated.csv", index=False)
    pec = t[t.signal == "pec"]
    put("realgen_n_conditions", int(len(pec)))
    put("realgen_n_gen_lower", int((pec["diff"] < 0).sum()))
    put("realgen_n_sig", int((pec.p_holm < 0.05).sum()))
    put("realgen_real_pec_gini_mean", float(real.pec_gini.mean()))


def classify(X, y, groups, seed=1337):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    acc, f1 = [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0,
                                                                 random_state=seed))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        acc.append(accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)


def detection(df, store, out):
    ev_cols = [f"{s}_{k}" for s in ("pec", "rollout", "ss") for k in ("gini", "entropy", "top10")]
    ev_cols = [c for c in ev_cols if c in df] + ["clipscore"]
    tasks = []
    gen = df[df.corpus == "guidance"].dropna(subset=ev_cols)
    sd = gen[gen.model.isin(["sd14", "sd15"])]
    tasks.append(("mode_sd", sd, "mode"))
    tasks.append(("model_sd14_vs_sd15", sd, "model"))
    tasks.append(("model_incl_biggan_RESOLUTION_CONFOUNDED", gen[gen["mode"].isin(["text", "weak"])], "model"))
    real = df[df.corpus == "real"].dropna(subset=ev_cols)
    if len(real):
        rg = pd.concat([gen.assign(label="generated"), real.assign(label="real")])
        tasks.append(("real_vs_generated", rg, "label"))
    rows = []
    for name, d, target in tasks:
        if d[target].nunique() < 2:
            continue
        y = d[target].to_numpy()
        groups = d.hf_row.to_numpy()
        emb = np.stack([store[k]["clip_emb"] for k in d.key])
        ev = d[ev_cols].to_numpy(float)
        chance = d[target].value_counts(normalize=True).max()
        for feats, X in [("clip", emb), ("evidence", ev), ("clip+evidence", np.hstack([emb, ev]))]:
            a, asd, f, fsd = classify(X, y, groups)
            rows.append(dict(task=name, features=feats, n=len(d), classes=d[target].nunique(),
                             chance=chance, acc=a, acc_sd=asd, macro_f1=f, macro_f1_sd=fsd))
            put(f"det_{name}_{feats.replace('+', '_')}_acc", float(a))
            put(f"det_{name}_{feats.replace('+', '_')}_f1", float(f))
        put(f"det_{name}_chance", float(chance))
    pd.DataFrame(rows).to_csv(f"{out}/detection.csv", index=False)


def write_numbers(out):
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in NUM.items()}
    with open(f"{out}/numbers.json", "w") as f:
        json.dump(clean, f, indent=1, sort_keys=True, default=float)
    with open(f"{out}/numbers.tex", "w") as f:
        f.write("% generated by scripts/analyze.py -- do not edit\n")
        f.write("\\providecommand{\\EV}[1]{\\csname ev@#1\\endcsname}\n")
        for k in sorted(NUM):
            f.write(f"\\expandafter\\def\\csname ev@{k}\\endcsname{{{fmt(NUM[k])}}}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--grid", type=int, default=7)
    args = ap.parse_args()
    out = os.path.join(args.run_dir, "results")
    os.makedirs(out, exist_ok=True)
    df, store = load(args.run_dir, args.grid)
    print(f"{len(df)} rows loaded")
    corpus_summary(df, out)
    bucket_table(df, out)
    gt = guidance_tests(df, out)
    signal_agreement(df, gt, out)
    prompt_dependence(df, out)
    clipscore_relation(df, out)
    smoothing(df, out)
    grid_ablation(args.run_dir, out)
    stability(df, store, out)
    real_vs_generated(df, out)
    detection(df, store, out)
    write_numbers(out)
    print(f"wrote {len(NUM)} numbers and {len(glob(out + '/*.csv'))} tables to {out}")


if __name__ == "__main__":
    main()
