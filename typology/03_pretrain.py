#!/usr/bin/env python3
"""03: self-supervised pre-training -> frozen 192-d career representation.

No health, mortality or biomarker variable is read by this script.

Self-supervised objectives:
  T1 masked SOC      mask 15% of tokens and predict their occupation at all
                     four SOC2000 levels (weighted 0.1 / 0.2 / 0.3 / 1.0)
  T2 masked span     mask 2-5 consecutive years; reconstruct SOC and intensities
  T3 next event      next year's occupation + occupational-change hazard
  T4 bag of occupations
                     predict every occupation ever held (multi-label) from the
                     sequence-level [CLS] token; the only objective that
                     supervises [CLS], because T1-T3 never mask position 0

Training runs in FP32 with deterministic algorithms (see common.set_seed).
Early stopping uses a relative threshold on the validation loss of a 15%
held-out set. After training the weights are frozen, their SHA256 is stored
with the representation, and downstream scripts verify it (common.load_latent).

Inputs:  results/typology/derived/tokens_{sex}.pt
Outputs: results/typology/derived/latent_{sex}.parquet (+ .meta.json)
         results/typology/models/encoder_{sex}.pt
         results/typology/qc/latent_diagnostics_{sex}.json

  python 03_pretrain.py            # both sexes
  python 03_pretrain.py --sex Female
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common import (effective_rank, fail, h1, h2, load_cfg, model_hash, ok,
                    paths, save_latent, set_seed, sexes, tee, warn)

# Must match 02_tokenize.py (ids 0..3 reserved; occupation codes start at 4)
PAD, CLS, MASK, UNK = 0, 1, 2, 3
N_SPECIAL = 4


class WorkHistoryEncoder(nn.Module):
    """Work-history encoder with four self-supervised heads (no outcome head)."""

    def __init__(self, cfg, n_soc, n_gap, n_intens, hier_sizes=None,
                 n_layers=None, dropout=None, cls_layernorm=None):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        if cls_layernorm is None:
            cls_layernorm = m.get("cls_layernorm", True)
        self.d, self.n_soc = d, n_soc
        self.hier_sizes = dict(hier_sizes or {})
        self.soc_emb = nn.Embedding(n_soc, d, padding_idx=PAD)
        # Hierarchical SOC embedding: parent-group components are shared across
        # unit groups, so rare 4-digit codes shrink towards their parent groups.
        self.hier_emb = nn.ModuleDict({
            lv: nn.Embedding(n, d, padding_idx=0)
            for lv, n in self.hier_sizes.items()})
        self.gap_emb = nn.Embedding(n_gap, d)
        self.age_emb = nn.Embedding(101, d)        # position = age in years
        self.dt_proj = nn.Linear(1, d)             # normalised SOC tenure
        self.n_jobs_proj = nn.Linear(1, d)         # jobs held in the year
        self.flag_proj = nn.Linear(2, d)           # is_transition + is_parallel
        self.intens_proj = nn.Linear(n_intens, d)
        layer = nn.TransformerEncoderLayer(
            d, m["n_heads"], m["ffn"],
            dropout=m["dropout"] if dropout is None else dropout,
            batch_first=True, norm_first=True, activation="gelu")
        # The nested-tensor fast path supports post-LN only; we use pre-LN.
        self.enc = nn.TransformerEncoder(
            layer, m["n_layers"] if n_layers is None else n_layers,
            enable_nested_tensor=False)
        # LayerNorm on [CLS]. Every token carries age_emb(age), so attention
        # pooling at [CLS] is largely an average of age embeddings over
        # [16, career_end], i.e. a function of career length. Normalisation
        # removes the mean and scale of the [CLS] vector, where this average
        # mainly lives. It must be present during training.
        self.cls_norm = nn.LayerNorm(d) if cls_layernorm else nn.Identity()
        self.head_soc = nn.Linear(d, n_soc)        # T1 and the SOC part of T2 (L4)
        self.head_hier = nn.ModuleDict({
            lv: nn.Linear(d, n) for lv, n in self.hier_sizes.items()})
        self.head_intens = nn.Linear(d, n_intens)  # intensity part of T2
        self.head_next = nn.Linear(d, n_soc)       # T3 next occupation
        self.head_hazard = nn.Linear(d, 1)         # T3 occupational change (Bernoulli)
        # T4: multi-label prediction of the set of occupations ever held from
        # [CLS]. Labels come from the input itself.
        self.head_bag = nn.Linear(d, n_soc)

    def embed(self, b):
        e = (self.soc_emb(b["soc"]) + self.gap_emb(b["gap"])
             + self.age_emb(b["age"])
             + self.dt_proj(b["dt"].unsqueeze(-1))
             + self.n_jobs_proj(b["n_jobs"].unsqueeze(-1))
             + self.flag_proj(torch.stack([b["is_transition"],
                                           b["is_parallel"]], dim=-1))
             + self.intens_proj(b["intens"]))
        for lv, emb in self.hier_emb.items():
            e = e + emb(b[f"soc_{lv[-2:]}"])
        return e

    def forward(self, b, x=None):
        return self.enc(self.embed(b) if x is None else x,
                        src_key_padding_mask=b["pad"])

    def represent(self, b):
        """Normalised [CLS] output: the career representation.

        Passed through cls_norm so that it is the same vector T4 supervises.
        """
        return self.cls_norm(self.forward(b)[:, 0])


def make_targets(b, cfg, rng, device, hier_keys=()):
    """Input corruption and labels for the four objectives.

    Masks apply to real tokens only (not padding, not [CLS]).
    """
    s = cfg["ssl"]["tasks"]
    real = (~b["pad"]).clone()
    real[:, 0] = False
    x = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    t = {}

    r = torch.rand(b["soc"].shape, generator=rng, device=device)
    m1 = real & (r < s["masked_soc"]["mask_ratio"])
    t["m1"], t["y1"] = m1, b["soc"].clone()
    for k in hier_keys:
        t[f"y1_{k}"] = b[k].clone()

    # T2 span mask, vectorised (no per-sample host synchronisation)
    B, L = b["soc"].shape
    lens = (~b["pad"]).sum(1)
    lo, hi = s["masked_span"]["span_min"], s["masked_span"]["span_max"]
    span = torch.randint(lo, hi + 1, (B,), generator=rng, device=device)
    valid = (lens - span - 2) > 0      # keep some context in short sequences
    u = torch.rand(B, generator=rng, device=device)
    start = 1 + (u * (lens - span - 1).clamp(min=1).float()).long()
    pos = torch.arange(L, device=device).unsqueeze(0)
    m2 = ((pos >= start.unsqueeze(1)) & (pos < (start + span).unsqueeze(1))
          & valid.unsqueeze(1) & real)
    t["m2"], t["y2_soc"], t["y2_intens"] = m2, b["soc"].clone(), b["intens"].clone()

    mm = m1 | m2
    x["soc"] = torch.where(mm, torch.full_like(x["soc"], MASK), x["soc"])
    # Parent-group ids are masked too; otherwise they leak the masked code
    for k in hier_keys:
        x[k] = torch.where(mm, torch.zeros_like(x[k]), x[k])
    x["intens"] = torch.where(m2.unsqueeze(-1), torch.zeros_like(x["intens"]),
                              x["intens"])

    nxt = real & torch.cat([real[:, 1:], torch.zeros_like(real[:, :1])], 1)
    t["m3"] = nxt
    t["y3_soc"] = torch.cat([b["soc"][:, 1:], torch.zeros_like(b["soc"][:, :1])], 1)
    # Change label precomputed in 02_tokenize (gap years do not count as changes)
    t["y3_hazard"] = b["soc_change_next"].clone()

    # T4 label: set of occupations in the unmasked sequence
    t["bag_src"] = b["soc"].clone()
    t["bag_valid"] = real & (b["soc"] > UNK)
    return x, t


def _bag_target(t, n_soc):
    """(B, L) occupation ids -> (B, n_soc) multi-hot."""
    src, valid = t["bag_src"], t["bag_valid"]
    y = torch.zeros(src.shape[0], n_soc, device=src.device)
    idx = torch.where(valid, src, torch.zeros_like(src))
    y.scatter_(1, idx, torch.where(valid, torch.ones_like(src, dtype=y.dtype),
                                   torch.zeros_like(src, dtype=y.dtype)))
    y[:, :N_SPECIAL] = 0.0    # special tokens are not targets
    return y


def ssl_loss(model, h, t, cfg):
    w = cfg["ssl"]["tasks"]
    lw = cfg["model"].get("soc_loss_weights", {})
    parts = {}

    def _ce(lg, y, m):
        return (nn.functional.cross_entropy(lg[m], y[m]) if m.any()
                else h.new_zeros(()))

    lo_soc = model.head_soc(h)
    l1 = _ce(lo_soc, t["y1"], t["m1"]) * lw.get("soc_l4", 1.0)
    for lv, head in model.head_hier.items():
        k = f"y1_soc_{lv[-2:]}"
        if k in t:
            l1 = l1 + _ce(head(h), t[k], t["m1"]) * lw.get(lv, 0.0)
    parts["T1"] = l1 * w["masked_soc"]["weight"]

    l2 = _ce(lo_soc, t["y2_soc"], t["m2"])
    if t["m2"].any():
        l2 = l2 + nn.functional.mse_loss(model.head_intens(h)[t["m2"]],
                                         t["y2_intens"][t["m2"]])
    parts["T2"] = l2 * w["masked_span"]["weight"]

    l3 = _ce(model.head_next(h), t["y3_soc"], t["m3"])
    if t["m3"].any():
        lg = model.head_hazard(h).squeeze(-1)[t["m3"]]
        l3 = l3 + nn.functional.binary_cross_entropy_with_logits(
            lg, t["y3_hazard"][t["m3"]],
            pos_weight=h.new_tensor(w["next_event"]["hazard_pos_weight"]))
    parts["T3"] = l3 * w["next_event"]["weight"]

    if "bag_of_occupations" in w:
        z = model.cls_norm(h[:, 0])
        # pos_weight offsets the imbalance: few positives among 353 classes
        parts["T4"] = nn.functional.binary_cross_entropy_with_logits(
            model.head_bag(z), _bag_target(t, model.n_soc),
            pos_weight=h.new_tensor(w["bag_of_occupations"]["pos_weight"])
        ) * w["bag_of_occupations"]["weight"]
    return sum(parts.values()), parts


def batches(tok, idx, bs, shuffle, gen=None, device="cpu"):
    order = idx[torch.randperm(len(idx), generator=gen)] if shuffle else idx
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        yield {k: v[sel].to(device, non_blocking=True) for k, v in tok.items()}


def train_one(tok, cfg, device, seed, n_soc, n_gap, n_intens, hier_sizes=None,
              verbose=True):
    """Train one encoder (FP32, deterministic). Returns the best-validation model."""
    set_seed(seed)
    model = WorkHistoryEncoder(cfg, n_soc, n_gap, n_intens, hier_sizes).to(device)
    hier_keys = tuple(k for k in tok if k.startswith("soc_l"))
    if verbose:
        npar = sum(p.numel() for p in model.parameters())
        print(f"    parameters {npar / 1e6:.2f} M")

    sc = cfg["ssl"]
    n = tok["soc"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    nv = max(int(n * sc["val_frac"]), 1)
    val_idx, tr_idx = perm[:nv], perm[nv:]
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"],
                            weight_decay=sc["weight_decay"])
    rng = torch.Generator(device=device).manual_seed(seed)
    rtol = sc["early_stop_rtol"]     # relative improvement threshold
    best, bad, hist, best_state, best_ep = float("inf"), 0, [], None, -1

    zero = torch.zeros((), device=device)
    for ep in range(sc["epochs"]):
        model.train()
        t0, tl, nb = time.perf_counter(), zero.clone(), 0
        for b in batches(tok, tr_idx, sc["batch_size"], True, g, device):
            opt.zero_grad(set_to_none=True)
            x, t = make_targets(b, cfg, rng, device, hier_keys)
            loss, _ = ssl_loss(model, model(b, x=model.embed(x)), t, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl = tl + loss.detach()
            nb += 1
        model.eval()
        vl, vb, vlp = zero.clone(), 0, {}
        with torch.no_grad():
            for b in batches(tok, val_idx, sc["batch_size"], False, device=device):
                x, t = make_targets(b, cfg, rng, device, hier_keys)
                loss, tp = ssl_loss(model, model(b, x=model.embed(x)), t, cfg)
                vl = vl + loss.detach()
                for k, v in tp.items():
                    vlp[k] = vlp.get(k, zero.clone()) + v.detach()
                vb += 1
        tr_loss = float(tl) / max(nb, 1)
        vl = float(vl) / max(vb, 1)
        vparts = {k: float(v) / max(vb, 1) for k, v in sorted(vlp.items())}
        hist.append({"epoch": ep, "train": tr_loss, "val": vl, "val_parts": vparts,
                     "sec": round(time.perf_counter() - t0, 1)})
        if verbose and (ep % 5 == 0 or ep == sc["epochs"] - 1):
            det = "  ".join(f"{k} {v:.3f}" for k, v in vparts.items())
            print(f"    ep {ep:>2}  train {tr_loss:.4f}  val {vl:.4f}"
                  f"  [{det}]  {hist[-1]['sec']:.0f}s")
        if vl < best * (1.0 - rtol):
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= sc["patience"]:
                if verbose:
                    print(f"    early stop at epoch {ep} "
                          f"(best val {best:.4f} at epoch {best_ep})")
                break
    else:
        if verbose:
            warn(f"reached the epoch limit ({sc['epochs']}) without early stopping "
                 f"(best val {best:.4f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def extract(model, tok, device, bs=1024):
    model.eval()
    zs = []
    for i in range(0, tok["soc"].shape[0], bs):
        b = {k: v[i:i + bs].to(device) for k, v in tok.items()}
        zs.append(model.represent(b).float().cpu().numpy())
    return np.concatenate(zs, 0)


def _dominance(z, y):
    """How strongly a scalar y organises the geometry of z.

    var_frac  share of the total variance of z explained by regressing each
              dimension of z on y (reverse regression)
    pc_r2     R^2 of y on each of the first three principal components
    r2_by_k   R^2 of y on the first k principal components
    """
    zc = z - z.mean(0)
    yc = y - np.nanmean(y)
    good = np.isfinite(yc)
    zc, yc = zc[good], yc[good]
    tot = float((zc ** 2).sum())
    yy = float(yc @ yc)
    if yy < 1e-12 or tot <= 0:
        var_frac = 0.0
    else:
        b = zc.T @ yc / yy
        var_frac = float(((yc[:, None] * b[None, :]) ** 2).sum() / tot)
    u, s, _ = np.linalg.svd(zc, full_matrices=False)
    pc_all = u * s
    pc_r2 = []
    for j in range(min(3, pc_all.shape[1])):
        x = pc_all[:, j]
        pc_r2.append(0.0 if x.std() < 1e-12
                     else float(np.corrcoef(x, yc)[0, 1] ** 2))
    r2_by_k = {}
    for k in (1, 5, 20, min(50, pc_all.shape[1])):
        if k > pc_all.shape[1]:
            continue
        x = pc_all[:, :k]
        resid = yc - x @ np.linalg.lstsq(x, yc, rcond=None)[0]
        r2_by_k[k] = float(1.0 - (resid @ resid) / max(yy, 1e-12))
    return {"var_frac": var_frac, "pc_r2": pc_r2, "r2_by_k": r2_by_k}


def diagnose(z, p, cfg, sex, seq_len):
    """Representation diagnostics: dimensional collapse and sequence-length dominance.

    Thresholds are set in config.ssl.collapse_checks. z is not modified.
    """
    cc = cfg["ssl"]["collapse_checks"]
    res = {}

    er = effective_rank(z)
    res["effective_rank"] = er
    res["cls_layernorm"] = bool(cfg["model"].get("cls_layernorm", True))
    res["t4_enabled"] = "bag_of_occupations" in cfg["ssl"]["tasks"]
    thr = cc["min_effective_rank"]
    (ok if er >= thr else fail)(
        f"effective rank {er:.1f} (threshold >= {thr}, {z.shape[1]} dimensions)")

    # Sequence length = number of person-year tokens after truncation
    sd = _dominance(z, np.asarray(seq_len, dtype="float64"))
    res.update({f"seq_len_{k}": v for k, v in sd.items()})
    thr_pc1 = cc["max_seqlen_pc1_r2"]
    res["seqlen_gate_pass"] = bool(sd["pc_r2"][0] <= thr_pc1)
    (ok if res["seqlen_gate_pass"] else fail)(
        f"R^2 of PC1 on sequence length {sd['pc_r2'][0]:.3f} (threshold <= {thr_pc1}); "
        f"first 5 PCs {sd['r2_by_k'].get(5, float('nan')):.3f}")

    (p["qc"] / f"latent_diagnostics_{sex}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", default="both")
    args = ap.parse_args()
    cfg = load_cfg()
    p = paths(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tgt = sexes(cfg) if args.sex == "both" else (args.sex,)
    seed = cfg["ssl"]["seed"]

    with tee(p, "03_pretrain_main"):
        h1("03: self-supervised pre-training")
        if device.type == "cuda":
            cap = torch.cuda.get_device_capability(0)
            print(f"  {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
        else:
            warn("CUDA not available; running on CPU")

        for sex in tgt:
            f = p["derived"] / f"tokens_{sex}.pt"
            if not f.exists():
                fail(f"missing {f.name}; run 02_tokenize.py first")
                continue
            blob = torch.load(f, weights_only=False)
            tok, eids = blob["tok"], blob["eid"]
            n_soc = max(int(tok["soc"].max()) + 1, len(blob["vocab"]) + N_SPECIAL)
            n_gap = max(int(tok["gap"].max()) + 1, len(blob["gap_vocab"]) + 1)
            n_intens = blob.get("n_intens", tok["intens"].shape[-1])
            hier = blob.get("hier_sizes", {})
            h2(f"{sex} ({len(eids):,} participants; SOC vocabulary {n_soc}, "
               f"gap {n_gap}, intensity {n_intens})")

            model, hist = train_one(tok, cfg, device, seed, n_soc, n_gap,
                                    n_intens, hier)
            z = extract(model, tok, device)
            sha = model_hash(model)
            sl = (~tok["pad"]).sum(1).numpy().astype("float64") - 1  # minus [CLS]
            diag = diagnose(z, p, cfg, sex, sl)
            save_latent(p["derived"] / f"latent_{sex}.parquet", eids, z, {
                "model_sha256": sha, "sex": sex, "seed": seed,
                "n_layers": cfg["model"]["n_layers"],
                "dropout": cfg["model"]["dropout"],
                "tokens_file": f.name, "level": blob.get("level"),
                "channels": blob.get("channels"),
                "truncate_at": blob.get("truncate_at"),
                "career_end_cap": blob.get("career_end_cap"),
                "hierarchical_soc": bool(hier),
                "ssl_tasks": cfg["ssl"]["tasks"],
                "best_val": min(h["val"] for h in hist),
                "epochs_run": len(hist), "diagnostics": diag,
                "outcome_blind": True})
            torch.save({"state": model.state_dict(), "hist": hist, "sha256": sha,
                        "n_soc": n_soc, "n_gap": n_gap, "n_intens": n_intens,
                        "hier_sizes": hier},
                       p["models"] / f"encoder_{sex}.pt")
        return 0


if __name__ == "__main__":
    sys.exit(main())
