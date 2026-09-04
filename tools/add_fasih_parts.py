#!/usr/bin/env python3
"""Add the training-only parts (posterior encoder + discriminator) to the
Aegis-graft checkpoint, taken from the public Fasih checkpoint.

The Aegis ONNX has no enc_q / model_d / dp.post_* / dp.flows.1 — they're used
only during training, not inference. Left at random init, the *frozen* enc_q
feeds a garbage posterior into the KL term, which is the ONLY loss with a
gradient path to the LoRA adapters -> the adapters learn garbage -> noise.

Fasih is the exact checkpoint Aegis was fine-tuned from (same architecture,
piper medium, 22 kHz, espeak ur). Grafting its enc_q gives training a real
posterior; its model_d a real discriminator. The generator inference path
(enc_p / dp flows 0,3,5,7 / flow / dec) is untouched, so infer() stays
bit-exact to the Aegis ONNX and the voice identity is unchanged.
"""
import sys, torch

GRAFT = sys.argv[1] if len(sys.argv) > 1 else "aegis-female.ckpt"
FASIH = sys.argv[2]
OUT   = sys.argv[3] if len(sys.argv) > 3 else GRAFT

TAKE = ("model_g.enc_q.", "model_g.dp.post_", "model_g.dp.flows.1.", "model_d.")

g = torch.load(GRAFT, map_location="cpu")
f = torch.load(FASIH, map_location="cpu")["state_dict"]
gsd = g["state_dict"]

added, reshaped, skipped = 0, 0, 0
for k, v in f.items():
    if not k.startswith(TAKE):
        continue
    if k in gsd and tuple(gsd[k].shape) != tuple(v.shape):
        skipped += 1
        print(f"  shape mismatch, skip {k}: {tuple(gsd[k].shape)} vs {tuple(v.shape)}")
        continue
    gsd[k] = v
    added += 1

g["state_dict"] = gsd
torch.save(g, OUT)
print(f"added {added} tensors from Fasih ({skipped} shape-skipped) -> {OUT}")
mg = sum(k.startswith("model_g.") for k in gsd)
md = sum(k.startswith("model_d.") for k in gsd)
print(f"checkpoint now: {mg} model_g + {md} model_d keys, {len(gsd)} total")
