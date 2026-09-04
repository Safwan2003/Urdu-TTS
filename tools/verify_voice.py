#!/usr/bin/env python3
"""Render test lines from the grafted ckpt vs the Aegis ONNX.

Stage 1 (installed inference piper): phonemise + ONNX reference audio.
Stage 2 (piper1-gpl checkout): load SynthesizerTrn from the ckpt, infer.
"""
import json, os, subprocess, sys, numpy as np, soundfile as sf
import os
# piper1-gpl source tree (for piper.train.vits). Override with PIPER_SRC=...
PIPER_SRC = os.environ.get("PIPER_SRC", "/content/piper1-gpl/src")

ONNX = "ur-aegis-female/ur_PK-aegis_female-medium.onnx"
SCR  = "./_scratch"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{SCR}/aegis-female.ckpt"
OUT  = f"{SCR}/verify"
os.makedirs(OUT, exist_ok=True)

LINES = [
    "آج میں آپ کی کیا مدد کر سکتی ہوں؟",
    "اپنے فون کے کی پیڈ پر چار ہندسے دبائیں۔",
    "آپ کے کارڈ پر ایک فراڈ ٹرانزیکشن ہے۔",
]

STAGE1 = r'''
import json, sys, numpy as np, soundfile as sf
from piper import PiperVoice
ONNX, OUT = sys.argv[1], sys.argv[2]
LINES = json.loads(sys.argv[3])
v = PiperVoice.load(ONNX)
cfg = json.load(open(ONNX + ".json")); pid = cfg["phoneme_id_map"]
out = []
for i, t in enumerate(LINES):
    ids = [1]
    for p in (x for s in v.phonemize(t) for x in s):
        if p in pid: ids += pid[p] + [0]
    ids += [2]
    out.append(ids)
    ch = [np.frombuffer(x.audio_int16_bytes, dtype=np.int16).astype(np.float32)/32768
          for x in v.synthesize(t)]
    sf.write(f"{OUT}/onnx_{i}.wav", np.concatenate(ch), 22050)
print(json.dumps(out))
'''

r = subprocess.run([sys.executable, "-c", STAGE1, ONNX, OUT, json.dumps(LINES)],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr); sys.exit(1)
id_seqs = json.loads(r.stdout.strip().splitlines()[-1])

sys.path.insert(0, PIPER_SRC)
import torch
from piper.train.vits.models import SynthesizerTrn

CFG = json.load(open(ONNX + ".json"))
ARCH = dict(spec_channels=513, segment_size=8192, inter_channels=192,
    hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
    kernel_size=3, p_dropout=0.1, resblock="2", resblock_kernel_sizes=(3, 5, 7),
    resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)), upsample_rates=(8, 8, 4),
    upsample_initial_channel=256, upsample_kernel_sizes=(16, 16, 8),
    n_speakers=1, gin_channels=0, use_sdp=True)

g = SynthesizerTrn(n_vocab=CFG["num_symbols"], **ARCH)
raw = torch.load(CKPT, map_location="cpu")
sd = raw.get("state_dict", raw)
sd = {k[len("model_g."):]: v for k, v in sd.items() if k.startswith("model_g.")}
miss, unexp = g.load_state_dict(sd, strict=False)
print(f"load: {len(miss)} missing, {len(unexp)} unexpected")
g.eval()


def logmel(x, n=1024, hop=256):
    w = np.hanning(n)
    fr = [np.abs(np.fft.rfft(w * x[i:i+n])) for i in range(0, len(x)-n, hop)]
    return np.log(np.array(fr) + 1e-6)


torch.manual_seed(1234)
for i, ids in enumerate(id_seqs):
    with torch.no_grad():
        a = g.infer(torch.LongTensor([ids]), torch.LongTensor([len(ids)]),
                    noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8)[0][0, 0].numpy()
    sf.write(f"{OUT}/ckpt_{i}.wav", a, 22050)
    ref, _ = sf.read(f"{OUT}/onnx_{i}.wav")
    n = min(len(a), len(ref))
    d = float(np.abs(logmel(a[:n]) - logmel(ref[:n])).mean()) if n > 1100 else -1
    print(f"[{i}] ckpt={len(a):6d} onnx={len(ref):6d}  logmel-L1≈{d:.3f}  rms={np.sqrt((a**2).mean()):.3f}  {LINES[i]}")

print(f"\nWAVs in {OUT}/  (ckpt_*.wav vs onnx_*.wav)")
