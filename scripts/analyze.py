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
from evidgeo import uncertainty as U  # noqa: E402

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
    if n < 3:  # too few pairs to test; keep the columns so downstream code is uniform
        nan = float("nan")
        return dict(n=n, mean_a=nan, mean_b=nan, diff=float(d.mean()) if n else nan, dz=nan,
                    t=nan, p_t=nan, p_wilcoxon=nan, frac_pos=nan)
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
    if "step" not in df:
        df["step"] = -1
    df["step"] = df["step"].fillna(-1).astype(int)
    df["key"] = (df.corpus + "/" + df.dataset + "/" + df.model + "/" + df["mode"] + "/" +
                 df.gid.astype(str) + "/" + df.seed.astype(str))
    has_step = df.step >= 0  # must match Record.key()
    df.loc[has_step, "key"] = df.loc[has_step, "key"] + "/" + df.loc[has_step, "step"].astype(str)
    store = {}
    for p in glob(f"{run}/maps/**/*.npz", recursive=True):
        with np.load(p) as z:  # z[f] decompresses on every access: read each array once
            arrs = {f: z[f] for f in z.files}
        for i, k in enumerate(arrs.pop("keys")):
            store[str(k)] = {f: a[i] for f, a in arrs.items()}
    ims = glob(f"{run}/imstats/*.parquet")
    if ims:  # prompt-free image statistics (scripts/image_stats.py)
        st = pd.concat([pd.read_parquet(p) for p in ims], ignore_index=True)
        st = st.drop(columns=[c for c in ("error",) if c in st]).drop_duplicates("key")
        df = df.merge(st, on="key", how="left")
    rename = {c: c[len(f"g{g}_"):] for c in df.columns if c.startswith(f"g{g}_")}
    if "label" not in df:
        df["label"] = ""
    df["label"] = df["label"].fillna("")
    return add_units(df.rename(columns=rename)), store


COCO_IDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "coco_master_cocoid.json")


def fe_fit(d: pd.DataFrame, y: str, cols: list[str], term: str, fe: str = "gid", cluster: str = "unit"):
    """OLS of y on cols with `fe` fixed effects (within transformation) and errors clustered by
    `cluster`; returns (coef, lo, hi, p) for `term`.

    Small-sample factor: when every fixed-effect level lies inside one cluster (captions inside
    their photograph, or caption = cluster), the absorbed fixed effects are not counted, as in
    Stata's xtreg, fe vce(cluster) (Cameron & Miller 2015, Sec. IV.C); otherwise they are counted,
    which equals y ~ cols + C(fe) with clustered errors.
    """
    import statsmodels.api as sm

    d = d.dropna(subset=[y] + cols)
    dm = d[[y] + cols] - d.groupby(fe)[[y] + cols].transform("mean")
    keep = [c for c in cols if dm[c].std() > 1e-12]
    nan = (float("nan"),) * 4
    if term not in keep or dm[y].std() < 1e-12:  # no within-caption variation (e.g. a constant map)
        return nan
    m = sm.OLS(dm[y], dm[keep]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    nested = bool((d.groupby(fe)[cluster].nunique() == 1).all())
    n, k, g = len(d), len(keep), (0 if nested else d[fe].nunique())
    se = float(np.sqrt(m.cov_params().loc[term, term] * (n - k) / (n - k - g)))
    if not se > 0:
        return nan
    b = float(m.params[term])
    p = float(2 * stats.norm.sf(abs(b / se)))
    return b, b - 1.959964 * se, b + 1.959964 * se, p


def add_units(df: pd.DataFrame) -> pd.DataFrame:
    """unit = the independent sampling unit for tests.

    The COCO caption source has five captions per photograph (1,000 images for 5,000
    rows), so captions of one photo are not independent: COCO-derived rows get the
    COCO image id as unit, every other corpus its own caption id. Paired tests average
    captions within a unit; regressions keep caption fixed effects and cluster by unit.
    """
    ids = json.load(open(COCO_IDS))["cocoid"] if os.path.exists(COCO_IDS) else []
    hr = df["hf_row"].fillna(-1).astype(int) if "hf_row" in df else pd.Series(-1, index=df.index)
    coco = (hr >= 0) & (hr < len(ids))
    df["image_id"] = -1
    if coco.any():
        df.loc[coco, "image_id"] = [ids[h] for h in hr[coco]]
    df["unit"] = np.where(coco, "coco" + df["image_id"].astype(str),
                          df["dataset"].astype(str) + "#" + df["gid"].astype(str))
    return df


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
    put("n_images_guidance", int(gen.loc[gen.image_id >= 0, "image_id"].nunique()))
    put("n_real_images", int(df.loc[(df.corpus == "real") & (df.image_id >= 0), "image_id"].nunique()))
    put("n_captions_guidance", int(gen.groupby(["dataset", "gid"]).ngroups))  # hf_row is COCO-only
    put("n_units_guidance", int(gen["unit"].nunique()) if "unit" in gen else 0)
    put("n_external", int((df.corpus == "external").sum()))
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
            wide = grp.pivot_table(index="unit", columns="mode", values=col)
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
    if len(g) < 3:
        return
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
            w = grp.pivot_table(index="unit", columns="mode", values=f"{name}_gini")
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
    # Tests use raw AUC (both maps share the image's own full/empty similarity). For a level on a
    # common 0-1 scale, nauc is pooled: (mean AUC - mean u_empty) / (mean u_full - mean u_empty);
    # per-image ratios blow up when the fill barely changes the similarity.
    pairs = [("ss", "loo"), ("ss", "gauss1"), ("ss", "gauss2"), ("pec", "sal")]
    pairs += [(m, base) for base in ("random", "centre") for m in ("pec", "sal", "rollout", "ss", "loo")]

    def pooled(col, idx):
        if "u_full" not in g or "u_empty" not in g:
            return np.nan
        ue, uf = g.loc[idx, "u_empty"].mean(), g.loc[idx, "u_full"].mean()
        return float((g.loc[idx, col].mean() - ue) / (uf - ue))

    for a, b in pairs:
        for kind in ("del", "ins"):
            ca, cb = f"{a}_{kind}_auc", f"{b}_{kind}_auc"
            if ca in g and cb in g:
                res = paired(g[ca], g[cb])
                idx = (g[ca] - g[cb]).dropna().index
                res.update(nauc_a=pooled(ca, idx), nauc_b=pooled(cb, idx))
                fa.append(dict(a=a, b=b, kind=kind, **res))
                for k in ("diff", "nauc_a", "nauc_b"):
                    put(f"faith_{kind}_{a}_vs_{b}_{k}", res.get(k, np.nan))
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
    df = add_units(df)
    rows = []
    for gr in grids:
        for s in ("pec", "ss", "rollout"):
            col = f"g{gr}_{s}_gini"
            if col not in df:
                continue
            for (ds, m), grp in df.groupby(["dataset", "model"]):
                w = grp.pivot_table(index="unit", columns="mode", values=col)
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
        rows.append(dict(mode=mode, gid=pid, unit=grp.unit.iloc[0], n_seeds=len(grp),
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
            w = per.pivot_table(index="unit", columns="mode", values=m)
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
    real = df[df.corpus == "real"].groupby("unit").mean(numeric_only=True)
    gen = df[df.corpus == "guidance"]
    if real.empty:
        return
    rows = []
    for (ds, m, mo), grp in gen.groupby(["dataset", "model", "mode"]):
        # multi-seed corpora: one value per prompt before pairing with its real image
        grp = grp[grp.hf_row >= 0].groupby("unit").mean(numeric_only=True)
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
    if (gen.model == "biggan").any():
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
        groups = d.unit.to_numpy()  # all captions and the real photo of one image stay in one fold
        if len(set(groups)) < 5:  # GroupKFold(5) needs at least 5 prompts
            continue
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


UNC_SCORES = ["seed_jsd", "seed_mi", "seed_patch_var", "seed_peak_spread",
              "seed_centroid_spread", "seed_gini_sd", "seed_sem_var"]


def seed_uncertainty(df, store, g, out):
    """Per prompt: spread of evidence maps across seeds (any corpus with >= 2 seeds)."""
    multi = df[df.corpus.isin(["guidance", "consistency"]) & (df.seed >= 0)]
    rows = []
    for (c, ds, m, mo, gid), grp in multi.groupby(["corpus", "dataset", "model", "mode", "gid"]):
        if grp.seed.nunique() < 2:
            continue
        r = dict(corpus=c, dataset=ds, model=m, mode=mo, gid=gid, unit=grp.unit.iloc[0], n_seeds=grp.seed.nunique(),
                 clipscore_mean=grp.clipscore.mean())
        embs = np.stack([store[k]["clip_emb"] for k in grp.key])
        for sig in ("pec", "ss"):
            f = f"g{g}_{sig}"
            if f in store[grp.key.iloc[0]]:
                u = U.seed_uncertainty(np.stack([store[k][f] for k in grp.key]), embs)
                r.update({f"{sig}_{k}": v for k, v in u.items() if np.isscalar(v)})
        rows.append(r)
    per = pd.DataFrame(rows)
    if per.empty:
        return per
    per.to_csv(f"{out}/uncertainty_seed_perprompt.csv", index=False)
    cols = [c for c in per.columns if any(c.endswith(u) for u in UNC_SCORES)]
    per.groupby(["corpus", "dataset", "model", "mode"])[cols].mean().to_csv(
        f"{out}/uncertainty_seed_by_mode.csv")
    tests = []
    for (c, ds, m), grp in per.groupby(["corpus", "dataset", "model"]):
        for col in cols:
            w = grp.pivot_table(index="unit", columns="mode", values=col)
            for a, b in PAIRS:
                if a in w and b in w:
                    tests.append(dict(corpus=c, dataset=ds, model=m, score=col, a=a, b=b,
                                      **paired(w[a], w[b])))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t["p_wilcoxon"])
        for _, r in t[t.score.isin(["pec_seed_jsd", "ss_seed_jsd", "pec_seed_sem_var"])].iterrows():
            put(f"unc_{r.dataset}_{r.model}_{r.score}_{r.a}_vs_{r.b}_dz", r.dz)
            put(f"unc_{r.dataset}_{r.model}_{r.score}_{r.a}_vs_{r.b}_p", r.p_holm)
    t.to_csv(f"{out}/uncertainty_seed_tests.csv", index=False)
    return per


def trajectory(df, store, g, out):
    tr = df[df.corpus == "trajectory"]
    rows = []
    for (ds, m, mo, gid, seed), grp in tr.groupby(["dataset", "model", "mode", "gid", "seed"]):
        grp = grp.sort_values("step")
        if len(grp) < 3:
            continue
        r = dict(dataset=ds, model=m, mode=mo, gid=gid, unit=grp.unit.iloc[0], seed=seed, n_steps=len(grp))
        for sig in ("pec", "ss"):
            f = f"g{g}_{sig}"
            if f in store[grp.key.iloc[0]]:
                u = U.trajectory_uncertainty(np.stack([store[k][f] for k in grp.key]),
                                             grp.step.to_numpy())
                r.update({f"{sig}_{k}": v for k, v in u.items() if np.isscalar(v)})
        rows.append(r)
    per = pd.DataFrame(rows)
    if per.empty:
        return per
    per.to_csv(f"{out}/uncertainty_trajectory.csv", index=False)
    per.groupby(["dataset", "model", "mode"]).mean(numeric_only=True).to_csv(
        f"{out}/uncertainty_trajectory_by_mode.csv")
    return per


def failure_prediction(seed_per, traj_per, correctness, out):
    """Does uncertainty predict failed generations? Needs a correctness CSV with
    dataset, model, mode, gid, seed, correct (0/1), e.g. from GenEval 2 / VQAScore."""
    if not correctness or not os.path.exists(correctness):
        return
    cor = pd.read_csv(correctness)
    keys = ["dataset", "model", "mode", "gid"]
    per_prompt = cor.groupby(keys).correct.mean().rename("acc").reset_index()
    rows = []
    if seed_per is not None and len(seed_per):
        d = seed_per.merge(per_prompt, on=keys)
        d["failed"] = d.acc < 0.5
        cands = [c for c in d.columns if any(c.endswith(u) for u in UNC_SCORES)]
        d["neg_clipscore"] = -d.clipscore_mean
        for c in cands + ["neg_clipscore"]:
            _, _, aurc = U.coverage_curve(d[c].to_numpy(), 1 - d.failed.to_numpy())
            rows.append(dict(level="prompt", score=c, n=len(d), fail_rate=d.failed.mean(),
                             auroc=U.auroc(d[c].to_numpy(), d.failed.to_numpy()), aurc=aurc))
    if traj_per is not None and len(traj_per):
        d = traj_per.merge(cor, on=keys + ["seed"])
        d["failed"] = d.correct < 0.5
        for c in [c for c in d.columns if "traj_" in c]:
            _, _, aurc = U.coverage_curve(d[c].to_numpy(), 1 - d.failed.to_numpy())
            rows.append(dict(level="image", score=c, n=len(d), fail_rate=d.failed.mean(),
                             auroc=U.auroc(d[c].to_numpy(), d.failed.to_numpy()), aurc=aurc))
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/failure_prediction.csv", index=False)
    for _, r in t.iterrows():
        put(f"fail_{r.level}_{r.score}_auroc", float(r.auroc))


def external(df, store, out):
    """Faces / memes / other sets: does evidence geometry differ by label?"""
    ex = df[(df.corpus == "external") & (df.label != "")] if "label" in df else df.iloc[:0]
    rows, det = [], []
    for ds, grp in ex.groupby("dataset"):
        labels = sorted(grp.label.unique())
        for s in ("pec", "rollout", "ss"):
            col = f"{s}_gini"
            if col not in grp:
                continue
            for a, b in combinations(labels, 2):
                x, y = grp.loc[grp.label == a, col].dropna(), grp.loc[grp.label == b, col].dropna()
                if len(x) > 2 and len(y) > 2:
                    u = stats.mannwhitneyu(x, y)
                    rows.append(dict(dataset=ds, signal=s, a=a, b=b, n_a=len(x), n_b=len(y),
                                     mean_a=x.mean(), mean_b=y.mean(),
                                     auc=u.statistic / (len(x) * len(y)), p=u.pvalue))
        ev_cols = [c for c in (f"{s}_{k}" for s in ("pec", "rollout", "ss")
                               for k in ("gini", "entropy", "top10")) if c in grp] + ["clipscore"]
        d = grp.dropna(subset=ev_cols)
        if d.label.nunique() > 1 and d.gid.nunique() >= 5:
            y, groups = d.label.to_numpy(), d.gid.to_numpy()
            emb = np.stack([store[k]["clip_emb"] for k in d.key])
            ev = d[ev_cols].to_numpy(float)
            for feats, X in [("clip", emb), ("evidence", ev), ("clip+evidence", np.hstack([emb, ev]))]:
                a, asd, f, fsd = classify(X, y, groups)
                det.append(dict(dataset=ds, features=feats, n=len(d),
                                chance=d.label.value_counts(normalize=True).max(),
                                acc=a, acc_sd=asd, macro_f1=f, macro_f1_sd=fsd))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t["p"])
    t.to_csv(f"{out}/external_label_tests.csv", index=False)
    pd.DataFrame(det).to_csv(f"{out}/external_detection.csv", index=False)


def guidance_sweep(df, seed_per, out, traj_per=None):
    """Modern-model sweep (modes cfg<s>): is concentration / uncertainty monotone in s?

    Per prompt we average over seeds, then fit  y ~ log(s) + C(prompt)  so the
    slope uses within-prompt variation only, and report Spearman of the
    per-scale means as a model-free check.
    """
    g = df[(df.corpus == "guidance") & df["mode"].str.match(r"^(edit-)?cfg[0-9.]+$")].copy()
    if g.empty:
        return
    g["cfg"] = g["mode"].str.extract(r"cfg([0-9.]+)$")[0].astype(float)
    rows = []
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        smf = None
    targets = [("pec_gini", g), ("ss_gini", g), ("rollout_gini", g), ("clipscore", g)]
    if seed_per is not None and len(seed_per):
        sp = seed_per[seed_per["mode"].str.match(r"^(edit-)?cfg[0-9.]+$")].copy()
        sp["cfg"] = sp["mode"].str.extract(r"cfg([0-9.]+)$")[0].astype(float)
        targets += [(c, sp) for c in ("pec_seed_jsd", "ss_seed_jsd", "pec_seed_sem_var") if c in sp]
    if traj_per is not None and len(traj_per):
        tp = traj_per[traj_per["mode"].str.match(r"^(edit-)?cfg[0-9.]+$")].copy()
        tp["cfg"] = tp["mode"].str.extract(r"cfg([0-9.]+)$")[0].astype(float)
        targets += [(c, tp) for c in ("pec_traj_commit_frac", "ss_traj_commit_frac") if c in tp]
    for col, src in targets:
        if col not in src:
            continue
        for (ds, m), grp in src.groupby(["dataset", "model"]):
            unit = src["unit"] if "unit" in src else src["gid"].astype(str)
            per = grp.assign(unit=unit).groupby(["gid", "unit", "cfg"])[col].mean().reset_index()
            means = per.groupby("cfg")[col].mean()
            if len(means) < 2:
                continue
            # rank correlation of the per-scale means needs >= 3 scales; the slope needs 2
            rho = stats.spearmanr(means.index, means.values).statistic if len(means) >= 3 else np.nan
            r = dict(dataset=ds, model=m, measure=col, n_prompts=per.gid.nunique(), n_units=per.unit.nunique(),
                     n_scales=len(means), spearman_means=rho,
                     **{f"mean_cfg{k:g}": v for k, v in means.items()})
            if smf is not None and per.gid.nunique() > 2:
                per["logcfg"] = np.log(per.cfg)
                b, _, _, pv = fe_fit(per, col, ["logcfg"], "logcfg")  # caption FE, unit-clustered
                r.update(slope_logcfg=b, p=pv)
                put(f"sweep_{ds}_{m}_{col}_slope", b)
                put(f"sweep_{ds}_{m}_{col}_p", pv)
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/guidance_sweep.csv", index=False)


def mask_grid(path: str, g: int) -> np.ndarray:
    """Fraction of each of the g x g patches covered by the (white) mask."""
    from PIL import Image

    m = np.asarray(Image.open(path).convert("L").resize((g * 32, g * 32)), dtype=float) / 255.0
    return m.reshape(g, 32, g, 32).mean(axis=(1, 3))


def mask_evidence(df, store, g, gen_roots, out):
    """Inpainting / editing: share of evidence inside the changed region vs its area.

    ratio > 1 means the map concentrates on the region the task changed. For
    MagicBrush the human target and the unedited source are scored against the
    same mask as references.
    """
    roots = {os.path.basename(os.path.normpath(r)): r for r in gen_roots}
    rows = []
    for _, r in df[df.corpus == "guidance"].iterrows():
        root = roots.get(r.dataset)
        if root is None:
            continue
        if r["mode"].startswith("inpaint"):
            mp = os.path.join(root, "masks", f"{int(r.gid):05d}_{r['mode']}.png")
        elif r["mode"].startswith("edit") or r.model == "magicbrush":
            mp = os.path.join(root, "masks", f"{int(r.gid):05d}_edit.png")
        else:
            continue
        if not os.path.exists(mp) or r.key not in store:
            continue
        cov = mask_grid(mp, g)
        area = float(cov.mean())
        row = dict(dataset=r.dataset, model=r.model, mode=r["mode"], gid=r.gid, seed=r.seed, area=area)
        for sig in ("pec", "rollout", "loo", "ss"):
            f = f"g{g}_{sig}"
            if f in store[r.key]:
                p = M.normalize(store[r.key][f]).reshape(g, g)
                row[f"{sig}_in"] = float((p * cov).sum())
                row[f"{sig}_ratio"] = row[f"{sig}_in"] / max(area, 1e-9)
        rows.append(row)
    per = pd.DataFrame(rows)
    if per.empty:
        return
    per.to_csv(f"{out}/mask_evidence_perimage.csv", index=False)
    summ = []
    for (ds, m, mo), grp in per.groupby(["dataset", "model", "mode"]):
        r = dict(dataset=ds, model=m, mode=mo, n=len(grp), area=grp.area.mean())
        for sig in ("pec", "rollout", "loo", "ss"):
            if f"{sig}_in" in grp:
                d = grp[f"{sig}_in"] - grp.area
                w = stats.wilcoxon(d) if len(d) > 2 and np.any(d != 0) else None
                r.update({f"{sig}_in": grp[f"{sig}_in"].mean(), f"{sig}_ratio": grp[f"{sig}_ratio"].mean(),
                          f"{sig}_p": float(w.pvalue) if w else np.nan})
                put(f"mask_{ds}_{m}_{mo}_{sig}_ratio", float(grp[f"{sig}_ratio"].mean()))
        summ.append(r)
    t = pd.DataFrame(summ)
    for sig in ("pec", "rollout", "loo", "ss"):
        if f"{sig}_p" in t:
            t[f"{sig}_p_holm"] = holm(t[f"{sig}_p"])
    t.to_csv(f"{out}/mask_evidence.csv", index=False)


def mask_control(df, store, g, gen_roots, out):
    """Inpainting control: is evidence inside the mask higher for the inpainted image
    than for the paired real photo and the i2i image under the *same* mask?
    (A centred mask collects evidence anyway because photos are centre-biased.)"""
    roots = {os.path.basename(os.path.normpath(r)): r for r in gen_roots}
    real = df[df.corpus == "real"].set_index("hf_row")
    rows = []
    for ds, root in roots.items():
        gen = df[(df.corpus == "guidance") & (df.dataset == ds)]
        for kind in ("inpaint-center", "inpaint-box"):
            for gid, grp in gen[gen["mode"] == kind].groupby("gid"):
                mp = os.path.join(root, "masks", f"{int(gid):05d}_{kind}.png")
                if not os.path.exists(mp):
                    continue
                cov = mask_grid(mp, g)
                def inside(keys, sig):
                    vals = [float((M.normalize(store[k][f"g{g}_{sig}"]).reshape(g, g) * cov).sum())
                            for k in keys if k in store and f"g{g}_{sig}" in store[k]]
                    return float(np.mean(vals)) if vals else np.nan
                i2i = gen[(gen["mode"] == "i2i") & (gen.gid == gid)]
                hr = grp.hf_row.iloc[0]
                rk = [real.loc[hr, "key"]] if hr in real.index else []
                for sig in ("pec", "rollout", "loo", "ss"):
                    rows.append(dict(dataset=ds, mask=kind, gid=gid, unit=grp.unit.iloc[0], signal=sig, area=float(cov.mean()),
                                     inpaint=inside(grp.key, sig), i2i=inside(i2i.key, sig), real=inside(rk, sig)))
    per = pd.DataFrame(rows)
    if per.empty:
        return
    per.to_csv(f"{out}/mask_control_perimage.csv", index=False)
    summ = []
    for (ds, kind, sig), grp in per.groupby(["dataset", "mask", "signal"]):
        grp = grp.groupby("unit")[["area", "inpaint", "i2i", "real"]].mean()  # one value per image
        r = dict(dataset=ds, mask=kind, signal=sig, n=len(grp), area=grp.area.mean(),
                 inpaint=grp.inpaint.mean(), i2i=grp.i2i.mean(), real=grp.real.mean())
        for ref in ("real", "i2i"):
            res = paired(grp.inpaint, grp[ref])
            r[f"vs_{ref}_dz"], r[f"vs_{ref}_p"] = res.get("dz", np.nan), res.get("p_wilcoxon", np.nan)
        summ.append(r)
    t = pd.DataFrame(summ)
    for ref in ("real", "i2i"):
        t[f"vs_{ref}_p_holm"] = holm(t[f"vs_{ref}_p"])
    for _, r in t.iterrows():
        k = f"maskctl_{r['mask']}_{r.signal}"
        for col in ("area", "inpaint", "i2i", "real", "vs_real_dz", "vs_real_p_holm", "vs_i2i_dz", "vs_i2i_p_holm"):
            put(f"{k}_{col.replace('_holm', '')}", float(r[col]))
    t.to_csv(f"{out}/mask_control.csv", index=False)


IMSTATS = ["rms_contrast", "colourfulness", "hf_energy", "saturation", "edge_gini", "edge_centre"]


def confound(df, out):
    """Is the guidance effect on Grad-PEC concentration about the prompt, or just
    about guided images being simpler? For each contrast/sweep report:
      * the effect on Grad-PEC, prompt-free saliency (sal) and Sobel-edge Gini;
      * the effect on the residual pec - sal (the prompt-specific part);
      * the Grad-PEC effect before and after adding image statistics as
        covariates, with caption fixed effects and caption-clustered errors.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return
    g = df[df.corpus == "guidance"].copy()
    if "sal_gini" not in g or "pec_gini" not in g:
        return
    g["resid"] = g.pec_gini - g.sal_gini
    covs = [c for c in IMSTATS if c in g and g[c].notna().any()]
    rows = []

    def fit(d, term, y, extra):
        """Caption fixed effects, SEs clustered by unit (the image, for COCO captions)."""
        return fe_fit(d, y, [term] + list(extra), term)

    for (ds, model), grp in g.groupby(["dataset", "model"]):
        modes = set(grp["mode"])
        designs = []
        if {"text", "weak"} <= modes:
            designs.append(("text_vs_weak", grp[grp["mode"].isin(["text", "weak"])].assign(
                x=lambda d: (d["mode"] == "text").astype(float)), "x"))
        sweep = grp[grp["mode"].str.match(r"^(edit-)?cfg[0-9.]+$")]
        if sweep["mode"].nunique() >= 2:
            sw = sweep.assign(x=np.log(sweep["mode"].str.extract(r"cfg([0-9.]+)$")[0].astype(float)))
            designs.append(("logcfg", sw, "x"))
            sw2 = sw[sw.x > 0]
            if sw2["mode"].nunique() >= 2:
                designs.append(("logcfg_s>=2", sw2, "x"))
        for name, d, term in designs:
            d = d.dropna(subset=["pec_gini", "sal_gini"])
            if d.gid.nunique() < 5:
                continue
            r = dict(dataset=ds, model=model, design=name, n=len(d), n_prompts=d.gid.nunique(),
                     n_units=d.unit.nunique())
            for y in ("pec_gini", "sal_gini", "edge_gini", "resid"):
                if y in d and d[y].notna().all():
                    b, lo, hi, p = fit(d, term, y, [])
                    r.update({f"{y}_b": b, f"{y}_lo": lo, f"{y}_hi": hi, f"{y}_p": p,
                              f"{y}_std_b": b / d[y].std()})
            dd = d.dropna(subset=covs) if covs else d
            if covs and len(dd) > 10:
                b0 = fit(dd, term, "pec_gini", [])[0]
                b, lo, hi, p = fit(dd, term, "pec_gini", covs)
                r.update(pec_adj_b=b, pec_adj_lo=lo, pec_adj_hi=hi, pec_adj_p=p,
                         pec_adj_retained=b / b0 if b0 else np.nan)
                br, lor, hir, pr = fit(dd, term, "resid", covs)
                r.update(resid_adj_b=br, resid_adj_lo=lor, resid_adj_hi=hir, resid_adj_p=pr)
            rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return
    for col in [c for c in t.columns if c.endswith("_p")]:
        t[col + "_holm"] = holm(t[col])
    t.to_csv(f"{out}/confound.csv", index=False)
    for _, r in t.iterrows():
        k = f"conf_{r.dataset}_{r.model}_{r.design}"
        for col in t.columns:
            if col.endswith(("_b", "_lo", "_hi", "_p_holm", "_retained", "_std_b")) and not pd.isna(r[col]):
                put(f"{k}_{col.replace('_holm', '')}", float(r[col]))


def edit_control(df, store, g, gen_roots, out):
    """MagicBrush: evidence inside the human edit mask for the generated edit vs the
    unedited source and the human-edited target, paired by edit (seeds averaged)."""
    roots = {os.path.basename(os.path.normpath(r)): r for r in gen_roots}
    rows = []
    for ds, root in roots.items():
        e = df[(df.corpus == "guidance") & (df.dataset == ds)]
        if not (e.model == "magicbrush").any():
            continue
        for gid, grp in e.groupby("gid"):
            mp = os.path.join(root, "masks", f"{int(gid):05d}_edit.png")
            if not os.path.exists(mp):
                continue
            cov = mask_grid(mp, g)
            for sig in ("pec", "rollout", "loo", "ss"):
                r = dict(dataset=ds, gid=gid, signal=sig, area=float(cov.mean()))
                for cond, sub in grp.groupby(grp.model + "/" + grp["mode"]):
                    vals = [float((M.normalize(store[k][f"g{g}_{sig}"]).reshape(g, g) * cov).sum())
                            for k in sub.key if k in store and f"g{g}_{sig}" in store[k]]
                    r[cond] = float(np.mean(vals)) if vals else np.nan
                rows.append(r)
    per = pd.DataFrame(rows)
    if per.empty:
        return
    per.to_csv(f"{out}/edit_control_perimage.csv", index=False)
    gens = [c for c in per.columns if "/edit-cfg" in c]
    summ = []
    for sig, grp in per.groupby("signal"):
        for gcol in gens:
            for ref in ("magicbrush/source", "magicbrush/human-target"):
                res = paired(grp[gcol], grp[ref])
                summ.append(dict(signal=sig, edit=gcol, ref=ref, n=res["n"], area=grp.area.mean(),
                                 mean_edit=res.get("mean_a"), mean_ref=res.get("mean_b"),
                                 dz=res.get("dz"), p=res.get("p_wilcoxon")))
        res = paired(grp["magicbrush/human-target"], grp["magicbrush/source"])
        summ.append(dict(signal=sig, edit="magicbrush/human-target", ref="magicbrush/source", n=res["n"],
                         area=grp.area.mean(), mean_edit=res.get("mean_a"), mean_ref=res.get("mean_b"),
                         dz=res.get("dz"), p=res.get("p_wilcoxon")))
    t = pd.DataFrame(summ)
    t["p_holm"] = holm(t["p"])
    t.to_csv(f"{out}/edit_control.csv", index=False)
    for _, r in t.iterrows():
        k = f"editctl_{r.signal}_{r.edit.split('/')[-1]}_vs_{r.ref.split('/')[-1]}"
        for col in ("area", "mean_edit", "mean_ref", "dz", "p_holm"):
            put(f"{k}_{col.replace('_holm', '')}", float(r[col]))


def write_numbers(out):
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in NUM.items()}
    with open(f"{out}/numbers.json", "w") as f:
        json.dump(clean, f, indent=1, sort_keys=True, default=float)
    with open(f"{out}/numbers.tex", "w") as f:
        f.write("% generated by scripts/analyze.py -- do not edit\n")
        f.write("\\providecommand{\\EV}[1]{\\csname ev@#1\\endcsname}\n")
        for k in sorted(NUM):
            f.write(f"\\expandafter\\def\\csname ev@{k}\\endcsname{{{fmt(NUM[k])}}}\n")


def external_pairs(df, pairs, out):
    """Generated vs. the paired real image of an external corpus (faces, memes), by caption.

    Like real_vs_generated for COCO: seeds averaged per caption, then a paired test per
    generation condition and map. pairs: [(external dataset, guidance dataset)].
    """
    rows = []
    for ext, gds in pairs:
        real = df[(df.corpus == "external") & (df.dataset == ext)].groupby("gid").mean(numeric_only=True)
        gen = df[(df.corpus == "guidance") & (df.dataset == gds)]
        for (m, mo), grp in gen.groupby(["model", "mode"]):
            grp = grp.groupby("gid").mean(numeric_only=True)
            idx = grp.index.intersection(real.index)
            for s in ("pec", "sal", "rollout", "loo", "ss"):
                c = f"{s}_gini"
                if c in grp and c in real:
                    res = paired(grp.loc[idx, c], real.loc[idx, c])
                    rows.append(dict(dataset=gds, model=m, mode=mo, signal=s, **res))
    t = pd.DataFrame(rows)
    if t.empty:
        return
    t["p_holm"] = holm(t["p_wilcoxon"])
    t.to_csv(f"{out}/external_pairs.csv", index=False)
    for _, r in t.iterrows():
        k = f"extpair_{r.dataset}_{r.model}_{r['mode']}_{r.signal}"
        for col in ("n", "mean_a", "mean_b", "dz", "p_holm"):
            put(f"{k}_{col.replace('_holm', '')}", int(r[col]) if col == "n" else float(r[col]))


def text_control(df, store, g, pairs, out):
    """Real captioned images (memes) vs their generations, with the real image's text masked.

    Real memes carry overlaid text that the generations lack, and CLIP maps are drawn to
    text. For each caption the real image's OCR mask is applied to the real image and, as a
    same-mask control, to every generation of that caption. Reports the share of evidence in
    the text region and the Gini over non-text cells (renormalised), paired by caption with
    seeds averaged. pairs: [(external dataset, guidance dataset, mask dir)].
    """
    rows = []
    for ext, gds, mdir in pairs:
        real = df[(df.corpus == "external") & (df.dataset == ext)].set_index("gid")
        gen = df[(df.corpus == "guidance") & (df.dataset == gds)]
        for gid, rr in real.iterrows():
            mp = os.path.join(mdir, f"{int(gid):05d}_text.png")
            if not os.path.exists(mp):
                continue
            cov = mask_grid(mp, g)
            keep = (cov < 0.5).ravel()
            if keep.sum() < 3:  # almost all text: nothing to compare
                continue

            def feats(key, sig):
                k = f"g{g}_{sig}"
                if key not in store or k not in store[key]:
                    return np.nan, np.nan
                m = M.normalize(store[key][k]).reshape(g, g)
                rest = m.ravel()[keep]
                return float((m * cov).sum()), float(M.gini(rest / max(rest.sum(), 1e-12)))

            for mode, grp in gen[gen.gid == gid].groupby("mode"):
                for sig in ("pec", "sal", "rollout", "loo", "ss"):
                    gv = np.array([feats(k, sig) for k in grp.key], dtype=float)
                    rv = feats(rr.key, sig)
                    rows.append(dict(dataset=gds, mode=mode, gid=gid, signal=sig, text_area=float(cov.mean()),
                                     real_text=rv[0], gen_text=np.nanmean(gv[:, 0]),
                                     real_gini_notext=rv[1], gen_gini_notext=np.nanmean(gv[:, 1]),
                                     real_gini=float(rr.get(f"{sig}_gini", np.nan)),
                                     gen_gini=float(grp[f"{sig}_gini"].mean()) if f"{sig}_gini" in grp else np.nan))
    per = pd.DataFrame(rows)
    if per.empty:
        return
    per.to_csv(f"{out}/text_control_perimage.csv", index=False)
    summ = []
    for (ds, mode, sig), grp in per.groupby(["dataset", "mode", "signal"]):
        r = dict(dataset=ds, mode=mode, signal=sig, n=len(grp), text_area=grp.text_area.mean())
        for what in ("text", "gini", "gini_notext"):
            res = paired(grp[f"real_{what}"], grp[f"gen_{what}"])
            r.update({f"real_{what}": res["mean_a"], f"gen_{what}": res["mean_b"],
                      f"{what}_dz": res["dz"], f"{what}_p": res["p_wilcoxon"]})
        summ.append(r)
    t = pd.DataFrame(summ)
    for what in ("text", "gini", "gini_notext"):
        t[f"{what}_p_holm"] = holm(t[f"{what}_p"])
    for _, r in t.iterrows():
        k = f"textctl_{r.dataset}_{r['mode']}_{r.signal}"
        for col in ("n", "text_area", "real_text", "gen_text", "text_dz", "text_p_holm", "real_gini", "gen_gini",
                    "gini_dz", "gini_p_holm", "real_gini_notext", "gen_gini_notext", "gini_notext_dz",
                    "gini_notext_p_holm"):
            put(f"{k}_{col.replace('_holm', '')}", int(r[col]) if col == "n" else float(r[col]))
    t.to_csv(f"{out}/text_control.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--gen-roots", nargs="*", default=[],
                    help="Generation folders with masks/ (inpainting, MagicBrush editing)")
    ap.add_argument("--correctness", default="",
                    help="CSV dataset,model,mode,gid,seed,correct for failure prediction")
    ap.add_argument("--text-pairs", nargs="*", default=[],
                    help="external_dataset:guidance_dataset:mask_dir, e.g. memes_real:memes:data_ext/memecap/text_masks")
    ap.add_argument("--real-pairs", nargs="*", default=[],
                    help="external_dataset:guidance_dataset, e.g. mmceleba_real:mmceleba")
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
    sp = seed_uncertainty(df, store, args.grid, out)
    tp = trajectory(df, store, args.grid, out)
    failure_prediction(sp, tp, args.correctness, out)
    guidance_sweep(df, sp, out, tp)
    external(df, store, out)
    mask_evidence(df, store, args.grid, args.gen_roots, out)
    mask_control(df, store, args.grid, args.gen_roots, out)
    edit_control(df, store, args.grid, args.gen_roots, out)
    confound(df, out)
    external_pairs(df, [tuple(p.split(":", 1)) for p in args.real_pairs], out)
    text_control(df, store, args.grid, [tuple(p.split(":", 2)) for p in args.text_pairs], out)
    write_numbers(out)
    print(f"wrote {len(NUM)} numbers and {len(glob(out + '/*.csv'))} tables to {out}")


if __name__ == "__main__":
    main()
