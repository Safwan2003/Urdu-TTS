# -*- coding: utf-8 -*-
"""Paste this into a Colab cell (any time after cell 4 / "Base model") and run.
It checks whether espeak-ng Urdu phonemisation is actually working on this
runtime -- the thing that decides if the A/B renders (and the training itself)
are meaningful. Paste the whole output back.

HEALTHY markers:
  - espeak-ng-data dir exists, ~400 entries, has `phontab` and `ur_dict`
  - each test sentence -> 40-90 phonemes, 0 dropped, ~90-180 ids
BROKEN markers (the bug we are chasing):
  - espeak-ng-data missing / tiny  OR  ~10-15 phonemes per sentence
  - short ids -> 0.6 s renders, logmel L1 ~2.7
"""
import os, json, subprocess, pathlib
import piper
from piper import PiperVoice

pkg = pathlib.Path(piper.__file__).parent
edir = pkg / "espeak-ng-data"
print("piper           :", piper.__file__)
try:
    import importlib.metadata as _m
    print("piper-tts ver   :", _m.version("piper-tts"))
except Exception as e:
    print("piper-tts ver   : ?", e)

print("espeak-ng-data  :", edir, "| exists:", edir.is_dir())
if edir.is_dir():
    entries = list(edir.iterdir())
    print("   entries      :", len(entries),
          "| phontab:", (edir / "phontab").exists(),
          "| ur_dict:", (edir / "ur_dict").exists(),
          "| lang/inc/ur:", (edir / "lang" / "inc" / "ur").exists())
for p in ("/usr/share/espeak-ng-data", "/usr/lib/x86_64-linux-gnu/espeak-ng-data"):
    print("   system copy  :", p, "exists:", os.path.isdir(p))
try:
    print("apt espeak-ng   :", subprocess.run(["espeak-ng", "--version"],
          capture_output=True, text=True).stdout.strip())
except Exception as e:
    print("apt espeak-ng   : none", e)

v = PiperVoice.load("/content/aegis.onnx")
print("voice data dir  :", getattr(v, "espeak_data_dir", "?"))
pid = json.load(open("/content/aegis.onnx.json"))["phoneme_id_map"]

for t in ["آپ کے کارڈ پر ایک فراڈ ٹرانزیکشن ہے۔",
          "براہِ کرم اپنے موبائل کے کی پیڈ سے پرانا پن مٹا دیں۔"]:
    ph = v.phonemize(t)
    flat = [x for s in ph for x in s]
    dropped = sorted(set(p for p in flat if p not in pid))
    kept = [p for p in flat if p in pid]
    print(f"\n{t}")
    print("   raw         :", ph)
    print(f"   phonemes={len(flat)}  kept={len(kept)}  dropped={dropped}"
          f"  -> {1 + 2*len(kept) + 1} ids   (healthy ~90-180)")

cache = pathlib.Path("/content/cache")
if cache.is_dir():
    import numpy as np
    npys = sorted(cache.rglob("*.npy"))
    print(f"\ntraining cache  : {len(npys)} .npy files under {cache}")
    for f in npys[:4]:
        try:
            a = np.load(f, allow_pickle=True)
            r = np.asarray(a).ravel()
            print(f"   {f.name:32s} len={r.size:4d}  head={r[:32]}")
        except Exception as e:
            print(f"   {f.name}: {e}")
else:
    print("\ntraining cache  : /content/cache not found")
