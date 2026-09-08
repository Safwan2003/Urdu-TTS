# -*- coding: utf-8 -*-
"""Synthesize the 135 enrichment clips with Uplift `helpdesk-agent` and append
them to dataset/metadata.csv. Idempotent: existing 0124+.wav are skipped."""
import os, sys, json, time, wave, pathlib, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(__file__))
from enrichment_sentences import NEW

KEY   = os.environ["UPLIFT_API_KEY"]
VOICE = "helpdesk-agent"
URL   = "https://api.upliftai.org/v1/synthesis/text-to-speech"
REPO  = pathlib.Path("/data/my_projects/aegis-urdu-loanword")
WAV   = REPO / "dataset" / "wav"
META  = REPO / "dataset" / "metadata.csv"
START = 124  # existing clips are 0001..0123

def synth(text, tries=4):
    body = json.dumps({"voiceId": VOICE, "text": text,
                       "outputFormat": "WAV_22050_16"}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 2000:
                raise RuntimeError(f"suspiciously small response ({len(data)} B)")
            return data
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"   retry {k+1}: {e}")
            time.sleep(2 * (k + 1))

def check(p):
    with wave.open(str(p)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    dur = n / sr
    assert sr == 22050, f"{p.name}: {sr} Hz"
    assert ch == 1,     f"{p.name}: {ch} ch"
    assert sw == 2,     f"{p.name}: {sw*8}-bit"
    assert 0.6 <= dur <= 14.0, f"{p.name}: {dur:.2f}s out of range"
    return sr, dur

BASE = [l for l in META.read_text(encoding="utf-8").splitlines()
        if "|" in l and int(pathlib.Path(l.split("|", 1)[0]).stem) < START]

def sync_meta():
    """Rewrite metadata.csv = the untouched 1..123 base + one row per new wav
    that actually exists on disk. Safe to call after every clip."""
    rows = list(BASE)
    for off2, (_c, txt) in enumerate(NEW):
        i2 = START + off2
        if (WAV / f"{i2:04d}.wav").is_file():
            rows.append(f"wav/{i2:04d}.wav|{txt}")
    META.write_text("\n".join(rows) + "\n", encoding="utf-8")

done = skipped = 0
try:
    for off, (cat, text) in enumerate(NEW):
        idx = START + off
        p = WAV / f"{idx:04d}.wav"
        if p.is_file():
            try:
                check(p); skipped += 1; continue
            except Exception:
                p.unlink()
        data = synth(text)
        p.write_bytes(data)
        _sr, dur = check(p)
        done += 1
        print(f"{idx:04d} [{cat}] {dur:5.2f}s  «{text[:48]}…»")
        if done % 10 == 0:
            sync_meta()
        time.sleep(0.35)
finally:
    sync_meta()

total_wavs = len(list(WAV.glob("*.wav")))
total_rows = len([l for l in META.read_text(encoding="utf-8").splitlines() if l.strip()])
print(f"\nsynth={done} skipped={skipped} appended={len(add)}")
print(f"metadata.csv rows={total_rows}  wav files={total_wavs}")
assert total_rows == total_wavs, "row / wav count mismatch"
