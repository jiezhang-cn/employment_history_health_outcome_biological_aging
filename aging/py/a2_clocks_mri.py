"""a2: MRI clocks at the first imaging visit (instance 2) -- C-M organ clocks retrained with the MULTI
method (six non-brain organs) and C-B brain clocks retrained following Tian et al. 2023.
No published weights are applied: both families are retrained on UKB IDPs with the published method.

C-M (per organ, sexes pooled; features = UKB fields of MULTI eFile1; abdominal error indicators set the
affected measurements to missing; complete cases):
  - trained in CN participants only. A within-distribution holdout of holdout_n = 500 CN participants is
    drawn; the remaining CN participants enter n_repeats = 50 outer 80/20 repeated holdouts, each with an
    inner inner_cv = 10-fold grid search (MAE);
  - candidates: LASSO (standardised features, alpha on a log grid) and linear SVR (C in svr_C); the
    family with the smallest |Cohen's d| between training and outer-test MAE is selected (a
    generalisation criterion, not minimum MAE);
  - CN participants are scored by the mean prediction over the repeats that did not train on them;
    others by the selected family refitted on all CN participants with the modal hyperparameter;
  - bias correction (Beheshti): gap = pred - age regressed on age, estimated on the out-of-bag CN
    predictions and applied to everyone.
C-B (Tian et al.: GM / WM / FC / whole brain, by sex):
  - linear SVR on standardised features, C = iqr(age)/1.349 and epsilon = iqr(age)/13.49 (MATLAB fitrsvm
    defaults), no hyperparameter search;
  - 20-fold cross-fitting in CN participants: each fold model scores its held-out CN participants, with
    the bias-correction line from the in-sample gap of its own training set (Tian's glmfit(age_train,
    gap_train)); non-CN participants are scored by the model trained on all CN participants with its own
    training-set correction;
  - connectivity edges (field 25752) are inner-joined with GM / WM by eid; complete cases.
Both families: liblinear penalises the intercept, so y is centred on the training mean before fitting;
gapy = corrected gap (years); gapz = gapy z-scored over all scored participants.

Outputs (derived/): clock_CM.parquet / clock_CB.parquet  eid, sex, age, cn, pred__*, gapy__*, gapz__*
Tables (tables/):   a2_CM_select.csv  per organ x candidate: Cohen's d, MAEs, modal hyperparameter
                    a2_CM_perf.csv / a2_CB_perf.csv  out-of-bag r / MAE in CN (before correction), published MAE

  python aging/py/a2_clocks_mri.py [--only CM|CB]
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from common_aging import (brain_features, clock_file, h1, h2, load_fc, load_measure, mask_multi_errors,
                          multi_features, ok, read_dictionary, read_extract, read_wide, setup, tee, warn)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ── Estimators ─────────────────────────────────────────────────────
class CenteredSVR(BaseEstimator, RegressorMixin):
    """Standardisation + linear epsilon-insensitive SVR; y centred on the training mean (liblinear
    penalises the intercept)."""

    def __init__(self, C=1.0, epsilon=0.0, max_iter=20000):
        self.C, self.epsilon, self.max_iter = C, epsilon, max_iter

    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        self.ym_ = float(np.mean(y))
        self.m_ = LinearSVR(C=self.C, epsilon=self.epsilon, loss="epsilon_insensitive", dual=True,
                            fit_intercept=True, max_iter=self.max_iter, random_state=0)
        self.m_.fit(self.sc_.transform(X), np.asarray(y) - self.ym_)
        return self

    def predict(self, X):
        return self.m_.predict(self.sc_.transform(X)) + self.ym_


def lasso_pipe():
    return make_pipeline(StandardScaler(), Lasso(max_iter=10000))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 else 0.0
    diff = b.mean() - a.mean()
    return float(diff / s) if s > 0 else (0.0 if diff == 0 else float(np.sign(diff) * np.inf))


def gap_line(pred, age):
    """Line of (pred - age) on age (Beheshti / Tian's glmfit). Returns (intercept, slope)."""
    return tuple(np.polyfit(age, pred - age, 1)[::-1])


def correct(pred, age, ab):
    a, b = ab
    return (pred - age) - (a + b * age)


def out_cols(ck, pred, gapy, scale):
    mu, sd = scale
    return {f"pred__{ck}": pred, f"gapy__{ck}": gapy, f"gapz__{ck}": (gapy - mu) / sd}


def frame_base(mea, eids, inst):
    m = mea.loc[eids]
    return pd.DataFrame({"eid": np.asarray(eids), "sex": m["sex"].to_numpy(),
                         "age": m[f"age_i{inst}"].to_numpy("float64"),
                         "cn": m[f"cn_i{inst}"].to_numpy()})


def assemble(parts: dict, mea, inst) -> pd.DataFrame:
    """{clock: DataFrame(index=eid, three columns)} -> one wide table (outer join: complete cases differ)."""
    if not parts:
        return pd.DataFrame(columns=["eid", "sex", "age", "cn"])
    W = pd.concat(parts.values(), axis=1)
    W = W.loc[sorted(W.index)]
    return pd.concat([frame_base(mea, W.index, inst), W.reset_index(drop=True)], axis=1)


def usable_ids(mea, idx, inst):
    """Participants with an age and a CN status at this instance."""
    m = mea.reindex(idx)
    return idx[m[f"age_i{inst}"].notna().to_numpy() & m[f"cn_i{inst}"].notna().to_numpy()]


# ── C-M ────────────────────────────────────────────────────────────
def fit_multi(X, age, cn_mask, cfg, seed):
    """One organ. X: complete-case DataFrame; age / cn_mask aligned with its rows."""
    mc = cfg["multi"]
    rng = np.random.default_rng(seed)
    cn_pos = np.flatnonzero(cn_mask)
    hold = rng.choice(cn_pos, size=min(mc["holdout_n"], len(cn_pos) // 5), replace=False)
    pool = np.setdiff1d(cn_pos, hold)
    Xv, y = X.to_numpy("float64"), age
    alphas = np.logspace(mc["lasso_alphas"]["lo"], mc["lasso_alphas"]["hi"], mc["lasso_alphas"]["n"])
    cands = {"lasso": (lasso_pipe(), {"lasso__alpha": alphas}),
             "svr": (CenteredSVR(max_iter=mc["svr_max_iter"]), {"C": list(mc["svr_C"])})}
    rec = {k: {"train": [], "test": [], "hold": [], "hyper": [], "models": []} for k in cands}
    n = len(X)
    for r in range(mc["n_repeats"]):
        perm = rng.permutation(pool)
        nt = int(round(len(pool) * mc["test_frac"]))
        te, tr = perm[:nt], perm[nt:]
        for k, (est, grid) in cands.items():
            gs = GridSearchCV(clone(est), grid, cv=KFold(mc["inner_cv"], shuffle=True, random_state=seed + r),
                              scoring="neg_mean_absolute_error", n_jobs=mc["n_jobs"], refit=True)
            gs.fit(Xv[tr], y[tr])
            m = gs.best_estimator_
            rec[k]["train"].append(mae(m.predict(Xv[tr]), y[tr]))
            rec[k]["test"].append(mae(m.predict(Xv[te]), y[te]))
            rec[k]["hold"].append(mae(m.predict(Xv[hold]), y[hold]))
            rec[k]["hyper"].append(list(gs.best_params_.values())[0])
            trained = np.zeros(n, bool)
            trained[tr] = True
            rec[k]["models"].append((m, trained))
    sel = []
    for k, v in rec.items():
        hy = pd.Series(v["hyper"]).mode().iloc[0]
        sel.append({"model": k, "cohens_d": cohens_d(v["train"], v["test"]),
                    "train_mae": np.mean(v["train"]), "test_mae": np.mean(v["test"]),
                    "holdout500_mae": np.mean(v["hold"]), "hyper_mode": float(hy)})
    sel = pd.DataFrame(sel)
    best = sel.loc[sel["cohens_d"].abs().idxmin(), "model"]
    hy = float(sel.set_index("model").loc[best, "hyper_mode"])
    est = clone(cands[best][0]).set_params(**{list(cands[best][1])[0]: hy})
    final = est.fit(Xv[cn_pos], y[cn_pos])
    return {"models": rec[best]["models"], "final": final, "best": best, "hyper": hy, "sel": sel,
            "hold": hold, "cn_pos": cn_pos}


def predict_oob(fit, Xv, cn_rows):
    """cn_rows: training-eligible CN rows. CN -> mean over repeats that did not train on them;
    others -> final model."""
    pred = fit["final"].predict(Xv)
    s, c = np.zeros(len(Xv)), np.zeros(len(Xv))
    for m, trained in fit["models"]:
        use = cn_rows & ~trained
        if use.any():
            s[use] += m.predict(Xv[use])
            c[use] += 1
    have = cn_rows & (c > 0)
    pred[have] = s[have] / c[have]
    return pred, int((cn_rows & (c == 0)).sum())


def run_cm(p, cfg, mea, dic):
    h2("C-M: MULTI method, retrained (six non-brain organs)")
    mc = cfg["multi"]
    qc2 = read_extract(p, "imaging_qc", dtype=str).set_index("eid")
    parts2, perf_rows, sel_rows = {}, [], []
    for i, organ in enumerate(mc["organs"]):
        feats = multi_features(dic, organ, 2)
        if not feats:
            warn(f"{organ}: no i2 features in the dictionary; skipped")
            continue
        X = mask_multi_errors(read_wide(p, "mri_multi", feats), qc2, cfg, 2)[feats].dropna()
        X = X.loc[usable_ids(mea, X.index, 2)]
        age = mea.loc[X.index, "age_i2"].to_numpy("float64")
        cn = (mea.loc[X.index, "cn_i2"] == True).to_numpy()                      # noqa: E712
        if cn.sum() < mc["min_train_n"]:
            warn(f"{organ}: {int(cn.sum())} CN complete cases < min_train_n = {mc['min_train_n']}; skipped")
            continue
        fit = fit_multi(X, age, cn, cfg, mc["seed"] + i)
        Xv = X.to_numpy("float64")
        pred, n_noob = predict_oob(fit, Xv, cn)
        if n_noob:
            warn(f"{organ}: {n_noob} CN participants were in the training set of every repeat; "
                 "scored with the final model")
        ab = gap_line(pred[cn], age[cn])                       # Beheshti, on out-of-bag CN predictions
        gapy = correct(pred, age, ab)
        scale = (float(np.mean(gapy)), float(np.std(gapy)))
        parts2[organ] = pd.DataFrame(out_cols(organ, pred, gapy, scale), index=X.index)
        s = fit["sel"].assign(organ=organ, chosen=lambda d: d["model"] == fit["best"])
        sel_rows.append(s)
        pc_, ac_ = pred[cn], age[cn]
        perf_rows.append({"clock": organ, "n_feat": len(feats), "n_all": len(X), "n_cn": int(cn.sum()),
                          "model": fit["best"], "hyper": fit["hyper"],
                          "r_oob": float(np.corrcoef(pc_, ac_)[0, 1]), "mae_oob": mae(pc_, ac_),
                          "paper_mae": mc["paper_mae"].get(organ, np.nan),
                          "beheshti_a": ab[0], "beheshti_b": ab[1], "cn_not_oob": n_noob})
        print(f"    {organ:<9} features {len(feats):>3}  complete {len(X):>6,}  CN {int(cn.sum()):>6,}  "
              f"model {fit['best']} ({fit['hyper']:.4g})  OOB r = {perf_rows[-1]['r_oob']:.3f}  "
              f"MAE = {perf_rows[-1]['mae_oob']:.2f} (published {perf_rows[-1]['paper_mae']})")
    cm = assemble(parts2, mea, 2)
    cm.to_parquet(clock_file(p, "C-M"), index=False)
    pd.DataFrame(perf_rows).to_csv(p["tables"] / "a2_CM_perf.csv", index=False)
    if sel_rows:
        pd.concat(sel_rows).to_csv(p["tables"] / "a2_CM_select.csv", index=False)


# ── C-B ────────────────────────────────────────────────────────────
def tian_svr(age_train, cfg):
    tc = cfg["tian"]
    q = np.subtract(*np.percentile(age_train, [75, 25]))
    return CenteredSVR(C=q / tc["svr_C_div"], epsilon=q / tc["svr_eps_div"], max_iter=tc["svr_max_iter"])


def fit_tian(Xv, age, cn, cfg, seed):
    """K folds over CN participants: one model and one training-set correction line per fold, plus a final
    model on all CN participants."""
    tc = cfg["tian"]
    cn_pos = np.flatnonzero(cn)
    folds = []
    fold_of = np.full(len(Xv), -1)
    kf = KFold(tc["n_folds"], shuffle=True, random_state=seed)
    for j, (tr, te) in enumerate(kf.split(cn_pos)):
        tr, te = cn_pos[tr], cn_pos[te]
        m = tian_svr(age[tr], cfg).fit(Xv[tr], age[tr])
        folds.append((m, gap_line(m.predict(Xv[tr]), age[tr])))
        fold_of[te] = j
    final = tian_svr(age[cn_pos], cfg).fit(Xv[cn_pos], age[cn_pos])
    fin_ab = gap_line(final.predict(Xv[cn_pos]), age[cn_pos])
    return {"folds": folds, "fold_of": fold_of, "final": (final, fin_ab)}


def predict_tian(fit, Xv, age, fold_of):
    pred, gapy = np.empty(len(Xv)), np.empty(len(Xv))
    groups = [(fit["final"], fold_of < 0)] + [(fit["folds"][j], fold_of == j) for j in range(len(fit["folds"]))]
    for (m, ab), mask in groups:
        if mask.any():
            pr = m.predict(Xv[mask])
            pred[mask], gapy[mask] = pr, correct(pr, age[mask], ab)
    return pred, gapy


def run_cb(p, cfg, mea, dic):
    h2("C-B: Tian 2023 method, retrained (by sex, linear SVR, 20 folds, training-set correction)")
    tc = cfg["tian"]
    allc2 = sorted({c for s in tc["models"].values() for c in brain_features(dic, s["subgroups"], 2)})
    XB2 = read_wide(p, "brain_tian", allc2) if allc2 else pd.DataFrame()
    FC2 = load_fc(p, 2)
    print(f"    connectivity (i2): {len(FC2):,} participants x {FC2.shape[1]} edges")
    parts2, perf_rows = {}, []
    sexes_ = ["Female", "Male"] if tc["by_sex"] else [None]
    for i, (name, spec) in enumerate(tc["models"].items()):
        feats = [c for c in brain_features(dic, spec["subgroups"], 2) if c in XB2.columns]
        if spec["subgroups"] and not feats:
            warn(f"{name}: no i2 features for {spec['subgroups']}; skipped")
            continue
        blocks = [XB2[feats]] if feats else []
        if spec["fc"]:
            if not len(FC2):
                warn(f"{name}: no i2 connectivity; skipped")
                continue
            blocks.append(FC2)
        if not blocks:
            warn(f"{name}: no features; skipped")
            continue
        X = pd.concat(blocks, axis=1, join="inner").dropna()
        X = X.loc[usable_ids(mea, X.index, 2)]
        fcols = list(X.columns)
        col_parts2 = []
        for sx in sexes_:
            Xs = X if sx is None else X[mea.loc[X.index, "sex"].to_numpy() == sx]
            age = mea.loc[Xs.index, "age_i2"].to_numpy("float64")
            cn = (mea.loc[Xs.index, "cn_i2"] == True).to_numpy()                 # noqa: E712
            lab = f"{name}/{sx or 'all'}"
            if cn.sum() < tc["min_train_n"]:
                warn(f"{lab}: {int(cn.sum())} CN complete cases < min_train_n = {tc['min_train_n']}; skipped")
                continue
            Xv = Xs.to_numpy("float64")
            fit = fit_tian(Xv, age, cn, cfg, tc["seed"] + i)
            pred, gapy = predict_tian(fit, Xv, age, fit["fold_of"])
            col_parts2.append(pd.DataFrame({"pred": pred, "gapy": gapy}, index=Xs.index))
            perf_rows.append({"clock": name, "sex": sx or "all", "n_feat": len(fcols), "n_all": len(Xs),
                              "n_cn": int(cn.sum()), "C": fit["final"][0].C, "epsilon": fit["final"][0].epsilon,
                              "r_oob": float(np.corrcoef(pred[cn], age[cn])[0, 1]), "mae_oob": mae(pred[cn], age[cn]),
                              "r_all": float(np.corrcoef(pred, age)[0, 1])})
            print(f"    {lab:<14} features {len(fcols):>4}  complete {len(Xs):>6,}  CN {int(cn.sum()):>6,}  "
                  f"OOB r = {perf_rows[-1]['r_oob']:.3f}  MAE = {perf_rows[-1]['mae_oob']:.2f}")
        if not col_parts2:
            continue
        d2 = pd.concat(col_parts2)
        scale = (float(d2["gapy"].mean()), float(d2["gapy"].std(ddof=0)))
        parts2[name] = pd.DataFrame(out_cols(name, d2["pred"].to_numpy(), d2["gapy"].to_numpy(), scale), index=d2.index)
    assemble(parts2, mea, 2).to_parquet(clock_file(p, "C-B"), index=False)
    pd.DataFrame(perf_rows).to_csv(p["tables"] / "a2_CB_perf.csv", index=False)


# ── main ───────────────────────────────────────────────────────────
def main():
    args, cfg, up, p = setup(__doc__, [(("--only",), {"choices": ["CM", "CB"], "default": None,
                                                     "help": "run one family only"})])
    with tee(p, "a2_clocks_mri"):
        h1("a2 MRI clocks: C-M (MULTI, retrained) + C-B (Tian, retrained)")
        mea = load_measure(p).set_index("eid")
        dic = read_dictionary(p)
        if args.only in (None, "CM"):
            run_cm(p, cfg, mea, dic)
        if args.only in (None, "CB"):
            run_cb(p, cfg, mea, dic)
        ok("a2 done")


if __name__ == "__main__":
    main()
