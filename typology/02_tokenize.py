#!/usr/bin/env python3
"""02: annual panel -> token tensors (per sex).

Inputs (results/typology/derived/): worklife_annual.parquet, cohort.parquet;
baseline covariates (year_birth) from `paths.data`.

Outputs (results/typology/derived/):
  tokens_{sex}.pt       token tensors + eids + metadata (12 input channels)
  vocab.json            SOC vocabulary, gap vocabulary, hierarchy sizes

Sequences are right-truncated at cohort.career_end, the same cut used for the
exposure summaries. Positions are encoded by age, not token index, so that
variable-length sequences share one age axis. Occupation is the 4-digit
SOC2000 unit group (353 groups) with its three parent groups stored as extra
ids for the hierarchical embedding. The intensity vector has 13 dimensions:
ten exposures, weekly hours, shift work and the share of 'do not know' answers,
so that 'do not know' is distinguishable from 'never exposed'.

  python 02_tokenize.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from common import (SOC_DIVISOR, fail, h1, h2, load_cfg, ok, paths, sexes,
                    soc_levels, tee, warn)

# Special tokens. 0 is PAD (nn.Embedding padding_idx).
PAD, CLS, MASK, UNK = 0, 1, 2, 3
N_SPECIAL = 4
N_EXPOSURE = 10
N_INTENS = 13          # 10 exposures + hours + shift + 'do not know' share
IDX_HOURS, IDX_SHIFT, IDX_DK = 10, 11, 12
HOURS_CAP = 100.0      # upper limit for weekly hours

# `dt` = min(tenure in the current SOC spell, TENURE_CAP) / TENURE_CAP.
# Capping keeps "recently changed vs settled" while removing most of the
# correlation of raw tenure with age.
TENURE_CAP = 3.0

# Main-analysis definitions written to the token file.
CHANNELS = "full"
TRUNCATE_AT = "career_end"


def norm_hours(s):
    """Weekly hours -> scaled by the 48 h long-hours threshold, capped first.

    The cap removes physically impossible values (e.g. 168 h/week) that would
    otherwise dominate this input channel.
    """
    return (pd.to_numeric(s, errors="coerce").fillna(0)
            .clip(upper=HOURS_CAP).to_numpy() / 48.0)


def build_vocab(ann, level):
    """SOC vocabulary (main model: soc_l4, 353 groups). Ids start at N_SPECIAL."""
    vals = sorted(int(v) for v in ann[level].dropna().unique())
    return {v: i + N_SPECIAL for i, v in enumerate(vals)}


def build_hier_maps(vocab, level):
    """4-digit vocabulary id -> compact ids of the three parent levels.

    emb(soc4) = E4[id4] + E3[id3] + E2[id2] + E1[id1], so rare unit groups are
    shrunk towards their parent groups. Returns {level: (map array, n classes)},
    with the map indexed by the 4-digit vocabulary id.
    """
    order = ["soc_l1", "soc_l2", "soc_l3", "soc_l4"]
    if level not in order:
        return {}
    base_div = SOC_DIVISOR[level]
    n_tok = max(vocab.values()) + 1 if vocab else N_SPECIAL
    out = {}
    for lv in [x for x in order if SOC_DIVISOR[x] > base_div]:
        div = SOC_DIVISOR[lv] // base_div
        raw = {}
        for code, tid in vocab.items():
            raw.setdefault(int(code) // div, []).append(tid)
        remap = {k: i + 1 for i, k in enumerate(sorted(raw))}  # 0 is PAD
        arr = np.zeros(n_tok, dtype="int64")
        for k, tids in raw.items():
            for t in tids:
                arr[t] = remap[k]
        out[lv] = (arr, len(remap) + 1)
    return out


def gap_vocab(cfg):
    return {c: i + 1 for i, c in enumerate(cfg["codings"]["gap_codes"])}


def add_tenure_and_hazard(a, cap=TENURE_CAP):
    """Add `_dt` (input) and `_chg` (label) to a panel sorted by (eid, age).

    `_dt`  = min(tenure in current SOC spell, cap) / cap
    `_chg` = whether the occupation code changes in the next year (target of
             the occupational-change head)
    Both are computed before the most recent `max_len - 1` years are selected,
    so tenure counts from the true start of the spell and the last kept year
    still has a next year. Gap years do not break a spell: SOC is forward-filled
    within person before comparison.
    """
    soc = pd.to_numeric(a["soc4"], errors="coerce")
    filled = soc.groupby(a["eid"]).ffill()
    new = ((filled != filled.groupby(a["eid"]).shift())
           | (a["eid"].to_numpy() != a["eid"].shift().to_numpy()))
    ten = a.groupby(new.cumsum().to_numpy()).cumcount().to_numpy() + 1
    a = a.copy()
    a["_dt"] = np.minimum(ten.astype("float32"), cap) / cap
    nxt = filled.groupby(a["eid"]).shift(-1)
    a["_chg"] = (nxt.notna() & filled.notna()
                 & (nxt != filled)).to_numpy().astype("float32")
    return a


def truncate(ann, cut):
    """Right truncation at the per-person cut age (cohort.career_end)."""
    c = cut.reindex(ann["eid"]).to_numpy()
    return ann[ann["age"].to_numpy() <= c]


def _empty(n, L, hier):
    """Empty tensors: 12 input channels plus the `soc_change_next` label."""
    out = {"soc": torch.zeros(n, L, dtype=torch.long),
           "gap": torch.zeros(n, L, dtype=torch.long),
           "age": torch.zeros(n, L, dtype=torch.long),
           "dt": torch.zeros(n, L),               # normalised SOC tenure
           "n_jobs": torch.zeros(n, L),           # jobs held in the year
           "is_transition": torch.zeros(n, L),
           "is_parallel": torch.zeros(n, L),
           "intens": torch.zeros(n, L, N_INTENS),
           # Label for the occupational-change head, precomputed here because
           # the raw SOC code is no longer available in the token tensor.
           "soc_change_next": torch.zeros(n, L),
           "pad": torch.ones(n, L, dtype=torch.bool)}
    for lv in hier:
        out[f"soc_{lv[-2:]}"] = torch.zeros(n, L, dtype=torch.long)
    out["soc"][:, 0] = CLS
    out["pad"][:, 0] = False
    return out


def pack_annual(ann, cut, vocab, gvocab, hier, max_len, cfg, level, verbose=True):
    """Annual panel -> fixed-length tensors. One token per person-year; position 0 is [CLS]."""
    a = truncate(ann, cut).sort_values(["eid", "age"])
    if a.empty:
        raise ValueError("empty after truncation; check cohort.career_end")
    exp_names = list(cfg["employment"]["exposures"])
    a = add_tenure_and_hazard(a)

    eids = a["eid"].to_numpy()
    uniq, first, counts = np.unique(eids, return_index=True, return_counts=True)
    n, L = len(uniq), max_len
    room = L - 1
    if verbose:
        nt = int((counts > room).sum())
        print(f"    sequence length median {np.median(counts):.0f}, max {counts.max()}"
              + (f"; {nt} longer than {room}, most recent {room} years kept" if nt else ""))

    # Keep the most recent `room` years
    rank = np.arange(len(eids)) - np.repeat(first, counts)
    keep_from = np.maximum(counts - room, 0)
    sel = rank >= np.repeat(keep_from, counts)
    a = a[sel]
    eids, rank = eids[sel], (rank - np.repeat(keep_from, counts))[sel]
    pos = rank + 1
    row = np.searchsorted(uniq, eids)

    out = _empty(n, L, hier)
    soc = a[level].map(vocab).fillna(UNK).to_numpy().astype("int64")
    # Gap years carry no occupation, so occupation and being in work are not entangled
    in_job = a["in_job"].to_numpy() == 1
    soc_tok = np.where(in_job, soc, PAD)
    out["soc"][row, pos] = torch.as_tensor(soc_tok, dtype=torch.long)
    for lv, (arr, _) in hier.items():
        out[f"soc_{lv[-2:]}"][row, pos] = torch.as_tensor(
            arr[np.clip(soc_tok, 0, len(arr) - 1)], dtype=torch.long)
    out["gap"][row, pos] = torch.as_tensor(
        pd.to_numeric(a["gap_code"], errors="coerce").map(gvocab)
        .fillna(0).to_numpy(), dtype=torch.long)
    out["age"][row, pos] = torch.as_tensor(np.clip(a["age"].to_numpy(), 0, 100),
                                           dtype=torch.long)
    out["dt"][row, pos] = torch.as_tensor(a["_dt"].to_numpy(), dtype=torch.float)
    out["n_jobs"][row, pos] = torch.as_tensor(
        a["n_jobs_year"].fillna(0).to_numpy(), dtype=torch.float)
    for k in ("is_transition", "is_parallel"):
        if k in a.columns:
            out[k][row, pos] = torch.as_tensor(
                pd.to_numeric(a[k], errors="coerce").fillna(0).to_numpy(),
                dtype=torch.float)
    out["soc_change_next"][row, pos] = torch.as_tensor(a["_chg"].to_numpy(),
                                                       dtype=torch.float)

    it = np.zeros((len(a), N_INTENS), dtype=np.float32)
    dk = np.zeros(len(a), dtype=np.float32)
    for i, nm in enumerate(exp_names):
        col = a[f"exp_{nm}"]
        it[:, i] = col.fillna(0).to_numpy()
        dk += col.isna().to_numpy().astype(np.float32)
    it[:, IDX_HOURS] = norm_hours(a["hours"])
    it[:, IDX_SHIFT] = pd.to_numeric(a["shift_level"], errors="coerce"
                                     ).fillna(0).to_numpy() / 2.0
    # Share of the ten agents answered 'do not know' in this person-year
    it[:, IDX_DK] = dk / N_EXPOSURE
    it[~in_job] = 0.0    # no exposure information in gap years
    out["intens"][row, pos] = torch.as_tensor(it)
    out["pad"][row, pos] = False
    return out, uniq


def main():
    cfg = load_cfg()
    p = paths(cfg)
    level = cfg["model"]["soc_level"]

    with tee(p, "02_tokenize_main"):
        h1(f"02: tokenisation (level={level})")

        d = p["derived"]
        if not (d / "cohort.parquet").exists():
            fail("missing cohort.parquet; run 01_worklife.py first")
            return 1
        ann = soc_levels(pd.read_parquet(d / "worklife_annual.parquet"))
        coh = pd.read_parquet(d / "cohort.parquet").set_index("eid")
        b = pd.read_csv(p["data"] / cfg["files"]["cov_baseline"]).set_index("eid")
        base = coh.join(b[["year_birth"]], how="left")
        print(f"  annual panel {len(ann):,} person-years; cohort {len(coh):,}")

        cut = coh["career_end"]
        vocab, gvocab = build_vocab(ann, level), gap_vocab(cfg)
        hier = build_hier_maps(vocab, level) if cfg["model"]["hierarchical_soc"] else {}
        print(f"  SOC vocabulary ({level}) {len(vocab)}; gap vocabulary {len(gvocab)}")
        if hier:
            print("  parent levels: " + ", ".join(
                f"{lv} ({n})" for lv, (_, n) in hier.items()))

        for sex in sexes(cfg):
            h2(f"{sex}")
            eids = base.index[base["sex"] == sex]
            if not len(eids):
                warn(f"no participants for {sex}")
                continue
            cut_s = cut.reindex(eids)
            tok, uniq = pack_annual(ann[ann["eid"].isin(eids)], cut_s, vocab,
                                    gvocab, hier, cfg["model"]["max_len"], cfg,
                                    level)
            L = tok["pad"].shape[1]
            real = (~tok["pad"]).sum(1)
            print(f"    {len(uniq):,} participants x {L} tokens; "
                  f"length median {int(real.median())}, max {int(real.max())}")

            torch.save({
                "tok": tok, "eid": uniq, "level": level, "sex": sex,
                "channels": CHANNELS, "truncate_at": TRUNCATE_AT,
                "vocab": vocab, "gap_vocab": gvocab,
                "hier_sizes": {lv: int(n) for lv, (_, n) in hier.items()},
                "n_intens": N_INTENS, "tenure_cap": TENURE_CAP,
                "career_end_cap": str(coh["career_end_cap"].iloc[0]),
                "schema_version": "v5.0",
            }, d / f"tokens_{sex}.pt")
            ok(f"tokens_{sex}.pt")


        (d / "vocab.json").write_text(json.dumps(
            {"level": level,
             "soc_vocab": {str(k): v for k, v in vocab.items()},
             "gap_vocab": {str(k): v for k, v in gvocab.items()},
             "hier_sizes": {lv: int(n) for lv, (_, n) in hier.items()}},
            indent=2, ensure_ascii=False), encoding="utf-8")
        ok("vocab.json")
        return 0


if __name__ == "__main__":
    sys.exit(main())
