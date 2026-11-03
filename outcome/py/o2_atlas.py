#!/usr/bin/env python3
"""o2: prevalence ratios of prevalent disease at time zero by career type.

For each sex, each career type is compared with all other types of the same
sex. Outcomes are (a) each ICD-10 chapter ("any code in the chapter") and
(b) each three-character code with sex-specific lifetime prevalence
>= op.min_prevalence (from o1's code catalogue). Estimator: adjusted prevalence
ratio from a log-link Poisson model with robust (sandwich) variance (Zou 2004),
one model per type: y ~ covariates + I(type == k). Covariates: restricted cubic
spline of age at time zero, questionnaire year, ethnic group, highest
qualification and assessment centre. Firth's penalty is used when a group has
fewer than op.firth_below cases; cells with fewer than op.min_cell_cases cases
are suppressed. BH-FDR is applied within sex x level (chapter, code).

Global test: career-type labels are permuted within deciles of age at time zero
(separately by sex), and two statistics of the whole type x outcome matrix (sum
of squared log ratios; number of FDR-significant cells) are recomputed from
covariate-adjusted observed/expected ratios. Covariates are fitted once under
the null, so each permutation needs only matrix products; the observed and the
permuted statistics use the same estimator.

Chapter-level outcomes pool all codes of the chapter (not only the selected
ones); chapter II includes the ICD-9-only cancer pseudo-code and excludes C44
(non-melanoma skin cancer, which would otherwise dominate "any cancer").

Inputs: o1 outputs (derived/op_person.parquet, derived/op_disease/,
tables/OP0_code_catalog.csv).
Outputs: tables/OP1_cells.csv, tables/OP2_global_null.csv,
qc/op_design.json, qc/op_null_{sex}_prevalent_{level}.npy.

  python o2_atlas.py --stage all
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common_out import (Z975, bh_fdr, build_design, fail, h1, h2, irls_poisson,
                        load_cfg_out, load_cfg_up, ok, paths_out, sexes, suppress,
                        tee, write_json, z_to_p)
from o1_assemble import load_matrices

PANEL = "prevalent"


# -- Inputs ------------------------------------------------------------------
def load_inputs(p):
    person = pd.read_parquet(p["derived"] / "op_person.parquet").set_index("eid")
    M, meta, codes = load_matrices(p, person, ("prevalent",))
    ch = pd.read_csv(p["derived"] / "op_disease" / "code_chapters.csv")
    cat = pd.read_csv(p["tables"] / "OP0_code_catalog.csv")
    return person, M["prevalent"], codes, ch, cat


def sex_view(person, prev, cat, sex):
    """Rows, type labels (0..K-1, ascending type number) and selected codes of one sex."""
    rows = np.flatnonzero((person["sex"] == sex).to_numpy())
    ps = person.iloc[rows].copy()
    raw = ps["type"].to_numpy()
    types = sorted(int(t) for t in set(raw.tolist()))
    lab = np.array([types.index(int(t)) for t in raw], dtype="int32")
    c = cat[(cat["sex"] == sex) & cat["selected"]].sort_values(
        ["chapter_index", "code"])
    return {"rows": rows, "person": ps, "lab": lab, "types": types,
            "codes": c["code"].tolist(), "prev": prev[rows]}


# -- Estimation ----------------------------------------------------------------
def fit_outcome(y, Xcov, lab, K, cfg, meta_row: dict) -> list:
    """One outcome column -> K rows (each type vs all other types)."""
    cases = np.array([y[lab == k].sum() for k in range(K)], dtype="float64")
    nk = np.array([(lab == k).sum() for k in range(K)], dtype="float64")
    tot_n, tot_c = float(nk.sum()), float(cases.sum())
    out = []
    for k in range(K):
        n_rest, c_rest = tot_n - nk[k], tot_c - cases[k]
        base = {**meta_row, "type_index": k, "n_type": nk[k],
                "cases_type": cases[k], "n_ref": n_rest, "cases_ref": c_rest,
                "prev_type": cases[k] / nk[k] if nk[k] else np.nan,
                "prev_ref": c_rest / n_rest if n_rest else np.nan}
        if nk[k] == 0 or n_rest == 0:
            out.append({**base, "log_pr": np.nan, "se": np.nan, "pr": np.nan,
                        "lo": np.nan, "hi": np.nan, "z": np.nan, "p": np.nan,
                        "q": np.nan, "firth": False, "converged": False})
            continue
        firth = bool(min(cases[k], c_rest) < cfg["op"]["firth_below"])
        X = np.column_stack([Xcov, (lab == k).astype("float64")])
        r = irls_poisson(y, X, firth=firth, robust=not firth)
        bi, si = float(r["beta"][-1]), float(r["se"][-1])
        zi = bi / si if si > 0 else np.nan
        out.append({**base, "log_pr": bi, "se": si, "pr": float(np.exp(bi)),
                    "lo": float(np.exp(bi - Z975 * si)),
                    "hi": float(np.exp(bi + Z975 * si)), "z": zi,
                    "p": float(z_to_p(np.array([zi]))[0]), "q": np.nan,
                    "firth": firth, "converged": r["converged"]})
    return out


def panel_columns(v, ch, cfg, level):
    """Outcome columns: (name, member codes). Code level = selected codes;
    chapter level = all codes of each chapter (C44 excluded from chapter II)."""
    if level == "cell":
        return [(c, [c]) for c in v["codes"]]
    m = ch[ch["chapter"] != "?"]
    drop = set(cfg.get("cancer", {}).get("exclude_from_chapter", []))
    out = []
    for (_, roman), g in m.groupby(["chapter_index", "chapter"], sort=True):
        codes = [c for c in g["code"].tolist() if c not in drop]
        out.append((roman, codes))
    return out


def run_panel(cfg, v, ch, codes_all, sex, level) -> tuple[list, dict]:
    cix = {c: i for i, c in enumerate(codes_all)}
    chi = ch.set_index("code")
    des = build_design(v["person"], cfg)
    Xcov, K = des["X"], len(v["types"])
    rows = []
    for name, members in panel_columns(v, ch, cfg, level):
        j = [cix[c] for c in members if c in cix]
        if not j:
            continue
        y = (np.asarray(v["prev"][:, j].sum(1)).ravel() > 0).astype("float64")
        meta = {"sex": sex, "panel": PANEL, "level": level, "outcome": name,
                "n_codes": len(j), "n_at_risk": int(len(y))}
        if level == "cell":
            r = chi.loc[name]
            meta.update(chapter=r["chapter"], chapter_index=int(r["chapter_index"]),
                        chapter_label=r["chapter_label"])
        else:
            g = ch[ch["chapter"] == name].iloc[0]
            meta.update(chapter=name, chapter_index=int(g["chapter_index"]),
                        chapter_label=g["chapter_label"])
        rows += fit_outcome(y, Xcov, v["lab"], K, cfg, meta)
    return rows, des


def stage_fit(p, cfg, cup) -> pd.DataFrame:
    person, prev, codes, ch, cat = load_inputs(p)
    all_rows, design_log, views = [], {}, {}
    for sex in sexes(cup):
        h2(f"{sex}")
        v = views[sex] = sex_view(person, prev, cat, sex)
        print(f"    K={len(v['types'])}, {len(v['codes'])} selected codes")
        for level in ("chapter", "cell"):
            rows, des = run_panel(cfg, v, ch, codes, sex, level)
            all_rows += rows
            design_log[f"{sex}|{PANEL}|{level}"] = {
                "n_cov_cols": len(des["names"]), "dropped": des["dropped"],
                "age_knots": des["knots"], "refs": des["refs"]}
            print(f"    {level:<8} {len(rows):>6,} cells")

    df = pd.DataFrame(all_rows)
    tmap = {sex: v["types"] for sex, v in views.items()}
    df["ref_type"] = -1.0
    df["type"] = [tmap[s][i] for s, i in zip(df["sex"], df["type_index"])]
    df["reference"] = "rest"

    h2("Suppression and FDR")
    df = suppress(df, cfg["op"]["min_cell_cases"])
    print(f"  suppressed {int(df['suppressed'].sum()):,}/{len(df):,} cells "
          f"(< {cfg['op']['min_cell_cases']} cases)")
    df["family"] = df["sex"] + "|" + df["panel"] + "|" + df["level"]
    for fam, g in df.groupby("family"):
        df.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
    for fam, g in df.groupby("family"):
        sig = int((g["q"] < cfg["op"]["fdr_alpha"]).sum())
        print(f"    {fam:<28} m={int(g['p'].notna().sum()):>5}  "
              f"q<{cfg['op']['fdr_alpha']}: {sig:>4}")

    f = p["tables"] / "OP1_cells.csv"
    df.to_csv(f, index=False)
    ok(f"{f.name}  {len(df):,} rows")
    write_json(p["qc"] / "op_design.json", design_log)
    return df


# -- Global permutation test ----------------------------------------------------
def _group_mat(lab, K):
    from scipy.sparse import csr_matrix
    n = len(lab)
    return csr_matrix((np.ones(n, dtype="float32"), (lab, np.arange(n))),
                      shape=(K, n))


def _stat(lab, K, Y, MU, cfg):
    """Energy (sum of squared log PR) and number of FDR-significant cells.

    log PR_k = log(O_k/E_k) - log(O_rest/E_rest), with E from the
    covariate-only model; SE = sqrt(1/O_k + 1/O_rest).
    """
    G = _group_mat(lab, K)
    O = np.asarray(G @ Y, dtype="float64")
    E = np.asarray(G @ MU, dtype="float64")
    mcc = cfg["op"]["min_cell_cases"]
    Or, Er = O.sum(0) - O, E.sum(0) - E
    lo = np.log(np.clip(O, .5, None) / np.clip(E, 1e-12, None))
    lr = np.log(np.clip(Or, .5, None) / np.clip(Er, 1e-12, None))
    log_pr = lo - lr
    se = np.sqrt(1.0 / np.clip(O, .5, None) + 1.0 / np.clip(Or, .5, None))
    m = (O >= mcc) & (Or >= mcc)
    z = np.where(m, log_pr / se, np.nan)
    q = bh_fdr(z_to_p(z))
    return (float(np.nansum(np.where(m, log_pr, np.nan) ** 2)),
            int(np.nansum(q < cfg["op"]["fdr_alpha"])), log_pr, m)


def permute_within(lab, strata, rng):
    out = lab.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = lab[rng.permutation(idx)]
    return out


def stage_null(p, cfg, cup) -> pd.DataFrame:
    person, prev, codes, ch, cat = load_inputs(p)
    B = int(cfg["op"]["null_n"])
    rng = np.random.default_rng(cfg["op"]["null_seed"])
    cix = {c: i for i, c in enumerate(codes)}
    rows = []
    for sex in sexes(cup):
        v = sex_view(person, prev, cat, sex)
        K = len(v["types"])
        n = len(v["person"])
        strata = pd.qcut(v["person"]["tz_age"], cfg["op"]["null_age_strata"],
                         labels=False, duplicates="drop").to_numpy()
        Xcov = build_design(v["person"], cfg)["X"]
        for level in ("chapter", "cell"):
            cols = panel_columns(v, ch, cfg, level)
            C = len(cols)
            Y = np.zeros((n, C), dtype="float32")
            MU = np.zeros((n, C), dtype="float32")
            for c, (name, members) in enumerate(cols):
                j = [cix[x] for x in members if x in cix]
                y = (np.asarray(v["prev"][:, j].sum(1)).ravel() > 0).astype("float64")
                # covariates only (null of no type effect)
                r = irls_poisson(y, Xcov, robust=False)
                Y[:, c] = y
                MU[:, c] = np.exp(np.clip(Xcov @ r["beta"], -30, 30))
            obs, obs_sig, _, obs_m = _stat(v["lab"], K, Y, MU, cfg)
            nulls = np.empty((B, 2))
            for b in range(B):
                e, s, _, _ = _stat(permute_within(v["lab"], strata, rng),
                                   K, Y, MU, cfg)
                nulls[b] = (e, s)
            pe = (1 + int((nulls[:, 0] >= obs).sum())) / (B + 1)
            psig = (1 + int((nulls[:, 1] >= obs_sig).sum())) / (B + 1)
            rows.append({"sex": sex, "panel": PANEL, "level": level,
                         "n_cells": int(obs_m.sum()),
                         "energy_obs": obs, "energy_null_mean": nulls[:, 0].mean(),
                         "energy_null_sd": nulls[:, 0].std(ddof=1),
                         "energy_ratio": obs / max(nulls[:, 0].mean(), 1e-12),
                         "energy_p": pe, "nsig_obs": obs_sig,
                         "nsig_null_mean": nulls[:, 1].mean(),
                         "nsig_p": psig, "null_n": B})
            print(f"    {sex:<7}{level:<8} energy {obs:.2f} vs null "
                  f"{nulls[:, 0].mean():.2f} +/- {nulls[:, 0].std(ddof=1):.2f} "
                  f"(P={pe:.3f}); significant cells {obs_sig} vs "
                  f"{nulls[:, 1].mean():.1f} (P={psig:.3f})")
            np.save(p["qc"] / f"op_null_{sex}_{PANEL}_{level}.npy", nulls)
    nl = pd.DataFrame(rows)
    nl.to_csv(p["tables"] / "OP2_global_null.csv", index=False)
    ok("OP2_global_null.csv")
    return nl


def main():
    ap = argparse.ArgumentParser(description="Prevalence ratios by career type "
                                             "and global permutation test")
    ap.add_argument("--stage", default="all", choices=["fit", "null", "all"])
    args = ap.parse_args()
    cfg, cup = load_cfg_out(), load_cfg_up()
    p = paths_out(cfg)

    with tee(p, f"o2_atlas_{args.stage}"):
        h1(f"o2 prevalent disease: {args.stage}")
        if not (p["derived"] / "op_person.parquet").exists():
            fail("op_person.parquet missing; run o1_assemble.py --stage all first")
            return 1
        if args.stage in ("fit", "all"):
            h1("Prevalence ratios (chapter and code level)")
            stage_fit(p, cfg, cup)
        if args.stage in ("null", "all"):
            h1("Global permutation test (labels permuted within age deciles)")
            stage_null(p, cfg, cup)
        h2("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
