# LoRA fine-tune the Urdu voice for English loanwords

One notebook. Nothing to install locally.

## Files

- **`finetune_low_rank_adaptation_colab.ipynb`** — the whole pipeline
- **`dataset/`** — 123 loanword-dense Urdu clips (Uplift AI `helpdesk-agent`,
  22 050 Hz mono) + `metadata.csv`. Ready to use; `zip -r dataset.zip metadata.csv wav`
  from inside `dataset/` to make the upload for Colab.
- **`tools/`** — `build_ckpt_from_onnx.py` (rebuild the trainable checkpoint
  from the Aegis ONNX), `diff_intermediates.py` (prove it bit-exact),
  `verify_voice.py` (A/B render). The notebook inlines the builder.
- `finetuning_brief.txt` — problem statement, evidence, guarantees

## The base checkpoint

Piper trains from a `.ckpt`; Aegis ships only the inference ONNX. **Cell 5 of
the notebook rebuilds a trainable checkpoint from that ONNX** — it grafts every
weight (text encoder, duration predictor, flow, vocoder) onto a fresh piper1-gpl
generator, verified **bit-exact** against the ONNX. Nothing to upload. If you do
have a real Aegis `.ckpt`, drop it at `/content/aegis-female.ckpt` and cell 5
uses it as-is.

## How to run

1. `cd dataset && zip -r /tmp/dataset.zip metadata.csv wav`
   (or bring your own — `metadata.csv` lines `wav/0001.wav|<urdu text>`
   `|`-delimited + a `wav/` folder of **22 050 Hz mono** WAVs).
2. Open `finetune_low_rank_adaptation_colab.ipynb` in Google Colab.
   **Runtime → Change runtime type → GPU (T4)**.
3. Upload `dataset.zip` (Files pane).
4. **Run all.** ~30–45 min.
5. **Cell 5b** plays the rebuilt base voice — confirm it is clean and female
   before the training cell spends 20 min on it.
6. Download `ur_PK-aegis_female-medium.onnx` + `.onnx.json` (last cell).

## What it does

| step | |
|---|---|
| 2 | installs piper1-gpl (training) — `scikit-build` first, then the build |
| 3 | unzips your dataset (or rebuilds it from Uplift AI) |
| 4 | pulls the Aegis ONNX (phonemisation + config) |
| 5 | **rebuilds `/content/aegis-female.ckpt` from the ONNX** (bit-exact graft) |
| 5b | **plays the base voice — must be clean female speech, else stop** |
| 6–7 | injects low-rank adapters into the text encoder (~1% of weights), **freezes everything else**, trains only the adapters |
| 8 | A/B: plain-Urdu output must be identical to base (mel-L1 ≈ 0); loanwords should move |
| 9 | folds the adapter into the weights → a normal Piper ONNX, `phoneme_id_map` unchanged |
| 10 | final listen + download |

## Ship

Drop the two files into `models/piper/ur-aegis-female/` as **v2**, keep v1 for
rollback. If the adapter over-corrects, re-run cell 9 with `SCALE = 0.5`.

## Why LoRA, not a full fine-tune

Every base weight stays byte-identical, so plain Urdu can't regress and the
voice identity is preserved. `set_lora_scale(0)` = the exact original voice.
Verified: zero-init adapter is a no-op, merge is bit-identical to the adapted
forward pass. See `finetuning_brief.txt`.

## Attribution

The Aegis voice is MIT-licensed by **Muhammad Mahwiz Khalil (Proxima AI)**
(<https://huggingface.co/mahwizzzz>). Any redistributed derivative must keep
that copyright and attribution.
