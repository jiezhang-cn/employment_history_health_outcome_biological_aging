#!/usr/bin/env python3
"""Occupation embedding and per-person occupational movement from the frozen encoder.

No retraining and no outcome data: inputs are the token files, the encoder
weights and the annual work-history table. The checkpoint hash
(`encoder_{sex}.pt` -> `sha256`) must equal `model_sha256` in
`latent_{sex}.meta.json`; otherwise the script stops for that sex.

Steps (per sex):
  1. effective occupation embedding of every SOC code in the vocabulary:
     E = E4[code] + E3[l3] + E2[l2] + E1[l1] (own embedding plus the embeddings
     of its parent groups, which carry most of the signal for sparse codes);
  2. observed occupation changes: consecutive in-job years with different
     codes, right-truncated at career end as in the tokens (a change across an
     intervening break counts as one move);
  3. per-person movement in the embedding: n_moves, path length, largest and
     mean single jump, ages of first and last move, and net displacement
     (distance between the first and the last occupation).

People who never changed code have n_moves = 0 and missing (not zero)
distance measures; downstream analyses must treat them as a separate group.

Inputs:  results/typology/derived/tokens_{sex}.pt, latent_{sex}.parquet
         (+ .meta.json), cohort.parquet, worklife_annual.parquet;
         results/typology/models/encoder_{sex}.pt
Output:  results/typology/derived/seqmove_{sex}.parquet

Run:
  python typology/05_derived.py               # both sexes
  python typology/05_derived.py --sex Female
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import (SOC_DIVISOR, fail, h1, h2, import_numbered, load_cfg,
                    load_latent, ok, paths, sexes, tee)


def load_encoder(p, sex, device):
    """Restore the frozen encoder and check its hash against the latent meta.

    A mismatch means the weights are not those that produced the frozen
    representation, so nothing derived from them would be consistent.
    """
    import torch
    pre = import_numbered("03_pretrain")
    if pre is None:
        fail("03_pretrain.py not found")
        return None
    f = p["models"] / f"encoder_{sex}.pt"
    if not f.exists():
        fail(f"missing {f.name}; run 03_pretrain.py first")
        return None
    # map_location lets a GPU-saved checkpoint load on a CPU-only machine
    ck = torch.load(f, map_location=device, weights_only=False)
    _, _, meta = load_latent(p["derived"] / f"latent_{sex}.parquet")
    if meta.get("model_sha256") and ck.get("sha256") != meta["model_sha256"]:
        fail(f"weight hash mismatch: checkpoint {str(ck.get('sha256'))[:12]} vs "
             f"latent meta {meta['model_sha256'][:12]}; stopping")
        return None
    cfg = load_cfg()
    model = pre.WorkHistoryEncoder(cfg, ck["n_soc"], ck["n_gap"], ck["n_intens"],
                                   ck.get("hier_sizes") or {}).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    ok(f"{f.name} loaded, hash verified ({str(ck.get('sha256'))[:12]})")
    return model, ck


def soc_embedding(model, blob, cfg):
    """Effective embedding of every vocabulary code: E4[id] plus the parent-level
    embeddings E3/E2/E1. Returns (E, soc_codes)."""
    import torch
    vocab = blob["vocab"]                       # {SOC code: token id}
    level = blob.get("level") or cfg["model"]["soc_level"]
    ids = torch.tensor(sorted(vocab.values()), dtype=torch.long)
    codes = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        E = model.soc_emb.weight[ids].clone()
        hier = import_numbered("02_tokenize").build_hier_maps(vocab, level)
        for lv, emb in model.hier_emb.items():
            mp = hier.get(lv)
            if mp is None:
                continue
            arr = torch.as_tensor(mp[0], dtype=torch.long)
            E = E + emb.weight.detach()[arr[ids]]
    E = E.float().cpu().numpy()
    soc_codes = np.array([codes[int(i)] for i in ids])
    return E, soc_codes


def transitions(ann, coh, eids_set, level):
    """Observed occupation changes between in-job years, up to career end."""
    a = ann[ann["eid"].isin(eids_set) & (ann["in_job"] == 1)].copy()
    cut = coh["career_end"].reindex(a["eid"]).to_numpy()
    a = a[a["age"].to_numpy() <= cut]
    div = SOC_DIVISOR.get(level, 1)
    a["code"] = pd.to_numeric(a["soc4"], errors="coerce") // div
    a = a.dropna(subset=["code"]).sort_values(["eid", "age"])
    prv = a.groupby("eid")["code"].shift()
    m = prv.notna() & (prv != a["code"])
    return pd.DataFrame({"eid": a.loc[m, "eid"].to_numpy(),
                         "age": a.loc[m, "age"].to_numpy(),
                         "frm": prv[m].to_numpy().astype("int64"),
                         "to": a.loc[m, "code"].to_numpy().astype("int64")})


def seq_movement(E, soc_codes, tr, eids, p, sex):
    """Per-person movement in the embedding; writes seqmove_{sex}.parquet."""
    pos = {int(c): i for i, c in enumerate(soc_codes)}
    t = tr[tr["frm"].isin(pos) & tr["to"].isin(pos)].copy()
    i = t["frm"].map(pos).to_numpy()
    j = t["to"].map(pos).to_numpy()
    t["d"] = np.linalg.norm(E[i] - E[j], axis=1)
    g = t.groupby("eid")
    out = pd.DataFrame({"n_moves": g.size(), "path_length": g["d"].sum(),
                        "max_single_jump": g["d"].max(), "mean_jump": g["d"].mean(),
                        "first_move_age": g["age"].min(),
                        "last_move_age": g["age"].max()})
    # detour ratio = path length / straight-line distance from first to last code
    fr = g["frm"].first().map(pos).to_numpy()
    to = g["to"].last().map(pos).to_numpy()
    net = np.linalg.norm(E[fr] - E[to], axis=1)
    out["net_displacement"] = net
    out["detour_ratio"] = np.where(net > 1e-6, out["path_length"] / net, np.nan)
    out = out.reindex(pd.Index(eids, name="eid"))
    out["n_moves"] = out["n_moves"].fillna(0).astype(int)
    n0 = int((out["n_moves"] == 0).sum())
    print(f"    {n0:,}/{len(out):,} ({n0 / len(out):.1%}) never changed code; "
          "distance measures set to missing")
    out.loc[out["n_moves"] == 0,
            ["path_length", "max_single_jump", "mean_jump", "detour_ratio",
             "net_displacement"]] = np.nan
    out.reset_index().to_parquet(p["derived"] / f"seqmove_{sex}.parquet", index=False)
    ok(f"seqmove_{sex}.parquet")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", default="both")
    args = ap.parse_args()
    cfg = load_cfg()
    p = paths(cfg)
    tgt = sexes(cfg) if args.sex == "both" else (args.sex,)

    with tee(p, "05_derived"):
        h1("05: occupation embedding and occupational movement")
        try:
            import torch
        except ImportError:
            fail("torch is required")
            return 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        coh = pd.read_parquet(p["derived"] / "cohort.parquet").set_index("eid")
        ann = pd.read_parquet(p["derived"] / "worklife_annual.parquet",
                              columns=["eid", "age", "soc4", "in_job"])

        for sex in tgt:
            h2(sex)
            tf = p["derived"] / f"tokens_{sex}.pt"
            if not tf.exists():
                fail(f"missing {tf.name}; run 02_tokenize.py first")
                continue
            blob = torch.load(tf, weights_only=False)
            eids = np.asarray(blob["eid"])
            got = load_encoder(p, sex, device)
            if got is None:
                continue
            model, _ = got
            lv = blob.get("level") or cfg["model"]["soc_level"]

            E, soc_codes = soc_embedding(model, blob, cfg)
            print(f"  embedding: {E.shape[0]} codes x {E.shape[1]} dims")
            tr = transitions(ann, coh, set(eids), lv)
            print(f"  {len(tr):,} occupation changes ({tr['eid'].nunique():,} people)")
            seq_movement(E, soc_codes, tr, eids, p, sex)
        return 0


if __name__ == "__main__":
    sys.exit(main())
