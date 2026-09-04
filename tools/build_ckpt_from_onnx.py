#!/usr/bin/env python3
"""Aegis ONNX -> trainable piper1-gpl .ckpt.

The Aegis voice ships only as an inference ONNX. piper.train resumes from a
Lightning .ckpt. This rebuilds the generator (the whole voice: enc_p, dp,
flow, dec) from the ONNX initializers and writes a .ckpt whose `state_dict`
holds `model_g.*`.

296 ONNX tensors carry their module name and map 1:1. The other 53 are
`onnx::Conv_*` / `onnx::ConvTranspose_*` — weight-norm-folded convs whose names
were dropped by constant-folding at export. Route each by its ONNX NODE NAME
(the name is the address), NOT by graph order: the exporter traces the residual-
coupling flow in reverse, so the folded flow convs appear flows.6, flows.4,
flows.2, flows.0, then dec (forward). Getting this wrong = exact weights in the
wrong slots = pure noise.
  /flow/flows.6/enc/in_layers.0/Conv -> flow.flows.6.enc.in_layers.0.weight_v
  /dec/ups.0/ConvTranspose           -> dec.ups.0.weight_v
For each: weight_v := W_onnx and weight_g := ||W_onnx|| over dims!=0, so the
weight_norm forward reproduces W_onnx exactly.

Plus dp.flows.0.logs (ElementwiseAffine): the exporter folds Neg(logs) to an
initializer feeding an Exp node, so onnx::Exp_* == -logs exactly.

Verified bit-exact against the ONNX end to end (tools/diff_intermediates.py).
enc_q, the discriminator, dp.post_* and dp.flows.1 are training-only, absent
from the ONNX, and left at fresh init (loaded strict=False).
"""
from __future__ import annotations
import argparse, json, pathlib, sys

import os
PIPER_SRC = os.environ.get("PIPER_SRC", "/content/piper1-gpl/src")

ARCH = dict(
    spec_channels=513, segment_size=8192,
    inter_channels=192, hidden_channels=192, filter_channels=768,
    n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
    resblock="2", resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
    upsample_rates=(8, 8, 4), upsample_initial_channel=256,
    upsample_kernel_sizes=(16, 16, 8),
    n_speakers=1, gin_channels=0, use_sdp=True,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--piper-src", default=PIPER_SRC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.piper_src)
    import numpy as np, onnx, torch
    from onnx import numpy_helper
    from piper.train.vits.models import SynthesizerTrn

    g = onnx.load(str(args.onnx))
    inits = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
             for t in g.graph.initializer}

    # opaque (weight-norm-folded) convs: derive the destination param straight
    # from the ONNX node name, e.g.
    #   /flow/flows.6/enc/in_layers.0/Conv  -> flow.flows.6.enc.in_layers.0.weight_v
    #   /dec/ups.0/ConvTranspose            -> dec.ups.0.weight_v
    #   /dec/resblocks.0/convs.1/Conv       -> dec.resblocks.0.convs.1.weight_v
    # No ordering assumptions — the name IS the address. (An earlier version
    # matched by graph order and got the flow blocks reversed: the exporter
    # traces the flow in reverse, so flows.6 comes before flows.0.)
    opaque = []  # (target_weight_v_key, onnx_initializer_name)
    for node in g.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        oin = [i for i in node.input if i.startswith("onnx::") and i in inits]
        if not oin:
            continue
        key = node.name.strip("/").rsplit("/", 1)[0].replace("/", ".") + ".weight_v"
        opaque.append((key, oin[0]))
    named = {n: w for n, w in inits.items() if not n.startswith("onnx::")}
    print(f"ONNX: {len(inits)} initializers | {len(named)} named | "
          f"{len(opaque)} opaque conv weights")

    model = SynthesizerTrn(n_vocab=json.load(open(str(args.onnx) + ".json"))["num_symbols"],
                           **ARCH)
    sd = model.state_dict()

    new, report = {}, {"named_ok": 0, "named_shapebad": [], "named_miss": [],
                       "opaque_ok": 0, "opaque_bad": []}

    for n, w in named.items():
        if n not in sd:
            report["named_miss"].append(n); continue
        if tuple(sd[n].shape) != tuple(w.shape):
            report["named_shapebad"].append((n, tuple(sd[n].shape), tuple(w.shape))); continue
        new[n] = w.to(sd[n].dtype)
        report["named_ok"] += 1

    # dp.flows.0 is ElementwiseAffine. Its reverse does exp(-self.logs); the
    # exporter constant-folds Neg(logs) into an initializer feeding an Exp node
    # (node /dp/flows.0/Exp), i.e. onnx::Exp_* == -logs exactly. m comes through
    # named as dp.flows.0.m.
    for en in list(inits):
        if en.startswith("onnx::Exp_"):
            new["dp.flows.0.logs"] = (-inits[en]).reshape(sd["dp.flows.0.logs"].shape).to(sd["dp.flows.0.logs"].dtype)
            report.setdefault("extra", []).append(f"dp.flows.0.logs <- -{en}")

    for tgt, oname in opaque:
        w = inits[oname]
        if tgt not in sd:
            report["opaque_bad"].append((tgt, oname, "no such key")); continue
        if tuple(sd[tgt].shape) != tuple(w.shape):
            report["opaque_bad"].append((tgt, oname, f"{tuple(sd[tgt].shape)} vs {tuple(w.shape)}")); continue
        new[tgt] = w.to(sd[tgt].dtype)
        gk = tgt[:-1] + "g"          # weight_v -> weight_g
        gv = torch.linalg.vector_norm(w.reshape(w.shape[0], -1), dim=1)
        new[gk] = gv.reshape(sd[gk].shape).to(sd[gk].dtype)
        report["opaque_ok"] += 1

    covered = set(new)
    missing = [k for k in sd if k not in covered
               and not k.startswith(("enc_q.", "emb_g."))]
    print(f"\nnamed matched : {report['named_ok']}/{len(named)}")
    for n, a, b in report["named_shapebad"]: print(f"   shape  {n}: model{a} onnx{b}")
    for n in report["named_miss"]: print(f"   missNM {n}")
    print(f"opaque matched: {report['opaque_ok']}/{len(opaque)}")
    for t, o, why in report["opaque_bad"]: print(f"   bad {t} <- {o}: {why}")
    print(f"generator keys still unset (excl. enc_q/emb_g): {len(missing)}")
    for k in missing[:30]: print(f"   .. {k}  {tuple(sd[k].shape)}")

    if args.dry_run:
        print("\n--dry-run, nothing written"); return 0

    # fill enc_q + anything left from fresh init (training-only, not the voice)
    full = dict(sd); full.update(new)
    ckpt = {
        "state_dict": {f"model_g.{k}": v for k, v in full.items()},
        "global_step": 0, "epoch": 0,
        "pytorch-lightning_version": "2.0.0",
        "hyper_parameters": {},
        "loops": {}, "callbacks": {},
        "lr_schedulers": [], "optimizer_states": [],
    }
    out = args.out or args.onnx.with_suffix(".ckpt")
    torch.save(ckpt, str(out))
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
