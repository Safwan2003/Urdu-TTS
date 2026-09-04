#!/usr/bin/env python3
"""Compare ONNX (ground truth) vs grafted torch model stage by stage.
Kills all noise (scales = [0,1,0]) so both are deterministic.
Prints, at each boundary, max/mean abs diff. First boundary that diverges
localises the broken graft component.
"""
import json, subprocess, sys, numpy as np
import os
# piper1-gpl source tree (for piper.train.vits). Override with PIPER_SRC=...
PIPER_SRC = os.environ.get("PIPER_SRC", "/content/piper1-gpl/src")

ONNX = "ur-aegis-female/ur_PK-aegis_female-medium.onnx"
SCR  = "./_scratch"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{SCR}/aegis-female.ckpt"
LINE = "اپنے فون کے کی پیڈ پر چار ہندسے دبائیں۔"

TAPS = {
    "enc_p.x        ": "/enc_p/encoder/Mul_2_output_0",
    "enc_p.m_p      ": "/enc_p/Split_output_0",
    "enc_p.logs_p   ": "/enc_p/Split_output_1",
    "flow.out (z*m) ": "/Mul_7_output_0",
    "dec.conv_pre   ": "/dec/conv_pre/Conv_output_0",
    "dec.ups.0      ": "/dec/ups.0/ConvTranspose_output_0",
    "audio          ": "output",
}

# ---- phonemise with the installed inference piper ----
S1 = r'''
import json,sys
from piper import PiperVoice
v=PiperVoice.load(sys.argv[1]); pid=json.load(open(sys.argv[1]+".json"))["phoneme_id_map"]
ids=[1]
for p in (x for s in v.phonemize(sys.argv[2]) for x in s):
    if p in pid: ids+=pid[p]+[0]
ids+=[2]
print(json.dumps(ids))
'''
ids = json.loads(subprocess.run([sys.executable, "-c", S1, ONNX, LINE],
                                capture_output=True, text=True).stdout.strip().splitlines()[-1])
ids = np.array([ids], dtype=np.int64)
lens = np.array([ids.shape[1]], dtype=np.int64)
scales = np.array([0.0, 1.0, 0.0], dtype=np.float32)
print(f"{ids.shape[1]} phoneme ids")

# ---- ONNX: expose taps, run ORT ----
import onnx, onnxruntime as ort
m = onnx.load(ONNX)
present = {vi.name for vi in list(m.graph.value_info) + list(m.graph.output)}
add = []
for name, t in TAPS.items():
    if t == "output":
        continue
    vi = onnx.helper.ValueInfoProto(); vi.name = t
    m.graph.output.append(vi); add.append(t)
onnx.save(m, f"{SCR}/aegis_taps.onnx")
sess = ort.InferenceSession(f"{SCR}/aegis_taps.onnx", providers=["CPUExecutionProvider"])
outs = sess.run(None, {"input": ids, "input_lengths": lens, "scales": scales})
onames = [o.name for o in sess.get_outputs()]
O = {n: v for n, v in zip(onames, outs)}
print("ONNX taps:", {k: O[v].shape for k, v in TAPS.items() if v in O})

# ---- torch grafted model, step by step ----
sys.path.insert(0, PIPER_SRC)
import torch
from piper.train.vits.models import SynthesizerTrn
from piper.train.vits import commons

ARCH = dict(spec_channels=513, segment_size=8192, inter_channels=192,
    hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
    kernel_size=3, p_dropout=0.1, resblock="2", resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)), upsample_rates=(8, 8, 4),
    upsample_initial_channel=256, upsample_kernel_sizes=(16, 16, 8),
    n_speakers=1, gin_channels=0, use_sdp=True)
net = SynthesizerTrn(n_vocab=256, **ARCH)
raw = torch.load(CKPT, map_location="cpu")["state_dict"]
miss, unexp = net.load_state_dict({k[8:]: v for k, v in raw.items()
                                   if k.startswith("model_g.")}, strict=False)
net.eval()

T = {}
with torch.no_grad():
    x = torch.LongTensor(ids); xl = torch.LongTensor(lens)
    xe, m_p, logs_p, x_mask = net.enc_p(x, xl)
    T["enc_p.x        "] = (xe * x_mask)
    T["enc_p.m_p      "] = m_p
    T["enc_p.logs_p   "] = logs_p
    logw = net.dp(xe, x_mask, g=None, reverse=True, noise_scale=0.0)
    w = torch.exp(logw) * x_mask * 1.0
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)
    m_pe = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_pe = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    z_p = m_pe  # noise_scale 0
    z = net.flow(z_p, y_mask, g=None, reverse=True)
    T["flow.out (z*m) "] = (z * y_mask)
    o = net.dec(z * y_mask, g=None)
    T["audio          "] = o
    # dec internals
    dx = net.dec.conv_pre(z * y_mask)
    T["dec.conv_pre   "] = dx
    import torch.nn.functional as F
    T["dec.ups.0      "] = net.dec.ups[0](F.leaky_relu(dx, net.dec.LRELU_SLOPE))

print(f"\ntorch durations sum (frames): {int(w_ceil.sum())}   ONNX audio frames: {O['output'].shape[-1]//256 if 'output' in O else '?'}")
print(f"{'boundary':16s} {'onnx shape':18s} {'torch shape':18s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'onnx‖·‖':>10s}")
for k, tname in TAPS.items():
    ov = O.get(tname if tname != "output" else "output")
    if ov is None:
        print(f"{k} (onnx tap missing)"); continue
    tv = T[k].numpy()
    a, b = ov.squeeze(), tv.squeeze()
    n = min(a.shape[-1], b.shape[-1]) if a.ndim else 1
    if a.ndim >= 1:
        a2, b2 = a[..., :n], b[..., :n]
    else:
        a2, b2 = a, b
    d = np.abs(a2 - b2)
    print(f"{k} {str(ov.shape):18s} {str(tv.shape):18s} {d.max():10.4f} {d.mean():10.5f} {np.abs(a2).mean():10.4f}")
