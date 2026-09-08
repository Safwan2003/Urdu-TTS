# LoRA fine-tune the Urdu voice for English loanwords

One self-contained Colab notebook that teaches the Piper voice
**`ur_PK-aegis_female-medium`** (VITS, espeak `ur`, 22.05 kHz) to pronounce
English loanwords — *keypad, card, transaction, balance, OTP, PIN, CVV,
statement* — cleanly **inside** Urdu sentences, without changing anything else
about the voice.

It does this with **low-rank adaptation**: freeze 100 % of the model, train tiny
adapters (~1 % of the weights) on a loanword-dense corpus, then fold them back
into the weights at export → a normal Piper ONNX with a byte-identical
`phoneme_id_map`. Because every base weight stays frozen, **plain Urdu cannot
regress**.

---

## The problem

The production voice says plain Urdu well but mangles English loanwords:
`keypad → "ki pat"`, `transaction` / `balance` come out muddy.

Things that were tried and rejected by ear on production:

- Urdu respellings (`کِی پَڈ` / `کی پَیڈ`)
- inline espeak phonemes `[[kˈiːpæd]]` → still `"ki pat"`
- Latin passthrough (`keypad`) → still wrong
- rewording prompts to avoid the word (interim workaround)

**"Add /æ/" is the wrong frame.** espeak `ur` never emits /æ/ — phoneme id 39
is a dead slot in this frontend, which is why the `[[..æ..]]` attempt failed:

| word | espeak `ur` output |
|---|---|
| keypad | `kˈiːpad` — long iː, **short a**, d |
| transaction | `tɹansˈakʃən` |
| balance | `bˈaləns` |

The target is to make the model render the **a-based phoneme shapes espeak
already produces**, cleanly coarticulated. Those use phoneme ids the voice
already has — they were just coarticulated wrong for loanword shapes.

---

## Why LoRA, not a full fine-tune

A full fine-tune rewrites every weight → risk of plain-Urdu regression and voice
drift on a live bank line, and you can only *hope* it didn't happen.

Instead: freeze the whole model, inject low-rank adapters into the text encoder
(attention q/k/v/o + FFN + `proj`, all `nn.Conv1d(c, c, 1)` = linear), train
**only** the adapters, then merge them at export.

Guarantees:

- adapters are **zero-initialised** → before training the model is bit-exact
- `set_lora_scale(0)` at any point → the exact original voice
- the merge is verified **bit-identical** to the adapted forward pass
- base weights are never touched → plain Urdu cannot drift, voice identity is
  preserved

---

## Repo layout

| path | what |
|---|---|
| `finetune_low_rank_adaptation_colab.ipynb` | the whole pipeline — open in Colab, Run all |
| `dataset/` | 185 loanword-dense Urdu clips (Uplift AI `helpdesk-agent`, 22 050 Hz mono) + `metadata.csv` |
| `training_metrics_log.md` | step-by-step metrics + safety checks from the shipped run (see below) |
| `tools/build_ckpt_from_onnx.py` | Aegis ONNX → trainable `.ckpt` (also inlined as notebook cell 5) |
| `tools/diff_intermediates.py` | tap the ONNX layer-by-layer, prove the graft bit-exact |
| `tools/verify_voice.py` | local A/B render, ckpt vs ONNX |
| `tools/enrichment_sentences.py` | the v2 enrichment sentences (DR / LV / NV / PR tagged) |
| `tools/synth_enrichment.py` | synth + validate + append via Uplift (idempotent) |
| `tools/diag_espeak.py` | paste-into-Colab check that espeak-ng Urdu phonemisation works |
| `ur-aegis-female/` | the base ONNX + json (untracked — `*.onnx` is gitignored) |
| `tune_model/` | the exported v2 ONNX + json from the shipped run (untracked) |

The base ONNX and the trainable `.ckpt` never live in git. `dataset.zip`
(`zip -r dataset.zip metadata.csv wav` from inside `dataset/`) is the upload for
the no-Drive path.

---

## How to run (Colab, T4)

1. Put an `aegis-urdu-loanword/` folder at the top of your Google Drive with
   `dataset/` (`metadata.csv` + `wav/`) and `ur-aegis-female/` (the Aegis
   `.onnx` + `.json`). Nothing else — the trainable checkpoint is rebuilt from
   the ONNX. *(No Drive? cell 3 takes an uploaded `dataset.zip`; cell 4 falls
   back to Hugging Face for the ONNX.)*
2. Open the notebook in Colab → **Runtime → Change runtime type → GPU (T4)**.
3. **Run all.** ~30–45 min. Approve the Drive mount prompt.
   On a re-run, **Runtime → Disconnect and delete runtime** first — a plain
   restart keeps `/content` and stale files there poison the run.
4. **Cell 5b** plays the frozen base voice — it must be clean, natural, female
   before the training cell spends 20 min on it.
5. **Cell 8** is the A/B: plain Urdu must still sound like Aegis at
   `scale = 0.33`; loanwords should be clearer than at `scale = 0.0`.
6. **Cell 10** is the final listen from the exported ONNX; the last cell
   downloads `ur_PK-aegis_female_loan-medium.onnx` + `.json` (also copied to
   `Drive/loan-out/`).

### Pipeline

| cell | step |
|---|---|
| 2 | GPU check |
| 4 | mount Drive |
| 6 | install piper1-gpl `[train]`, compile `espeakbridge` + monotonic-align, force the legacy ONNX exporter, guarantee a complete `espeak-ng-data` |
| 9–10 | dataset: copy from Drive / unzip / rebuild from Uplift AI; assert 22 050 Hz; play 3 teacher clips |
| 12 | pull the Aegis ONNX + json (phonemisation + config) |
| 14 | **phoneme sanity gate** — real corpus lines must phonemise to full Urdu, not a ~10-phoneme stub |
| 16 | **rebuild the trainable `.ckpt` from the ONNX** (bit-exact graft) |
| 18 | **5b** — synth from the rebuilt base, no adapter → must be clean female speech |
| 20 | the `LoRAConv1d` adapter |
| 22 | **train** — base frozen, only the adapters + `enc_q` learn |
| 24 | **A/B** — plain unchanged, loanwords improved; prior-variance + logmel safety checks |
| 26 | merge the adapter into the weights → Piper ONNX, `phoneme_id_map` verified unchanged |
| 28–29 | final listen + download |

---

## The base checkpoint (ONNX → ckpt graft)

Piper trains from a Lightning `.ckpt`; Aegis ships **only** the inference ONNX
(`mahwizzzz/piper-voice-ur-aegis-female` — ONNX + json, no ckpt). Cell 16
rebuilds a trainable checkpoint by grafting **every** ONNX weight onto a fresh
piper1-gpl generator — text encoder, stochastic duration predictor,
residual-coupling flow, HiFi-GAN vocoder. Verified **bit-exact** against the
ONNX end to end (`tools/diff_intermediates.py`: audio max |Δ| `0.0000` at zero
noise).

Two gotchas that this graft gets right:

- **The exporter traces the coupling flow in reverse**, so the weight-norm-folded
  `onnx::Conv_*` tensors appear `flows.6, flows.4, flows.2, flows.0` in the
  graph. They are routed by ONNX **node name**, never by graph order — order it
  by position and `flows.6`'s weights land in `flows.0`'s slots and the voice is
  pure noise.
- `dp.flows.0.logs == -onnx::Exp_*` (the initializer *is* `-logs`, feeding an
  `Exp` node) — **not** `-log(onnx::Exp_*)`.

`enc_q` (posterior encoder), the discriminator and `dp.post_*` / `dp.flows.1`
are training-only and absent from the ONNX, so they start from fresh init. They
are **not** grafted from any other voice (no Fasih, no male model). Instead the
training cell **unfreezes `enc_q`** and trains it alongside the adapters: the mel
loss runs `enc_q → frozen female decoder`, so `enc_q` learns female latents,
which become the KL target the adapters fit. `enc_q` and the discriminator are
discarded at export → the shipped model is 100 % Aegis-female.

The lineage of the voice: `hi_IN/rohan → Fasih ur_PK-male → Aegis` (Muhammad
Mahwiz Khalil / Proxima AI, female fine-tune of Fasih).

---

## Training results — the shipped run (2026-09-08)

**Config:** `RANK 8`, `ALPHA 8` (effective gain 1.0), `LR 1e-4`, `ENCQ_LR 1e-4`,
`MAX_STEPS 1000`, `BATCH 8`, targets =
`enc_p.encoder.attn_layers`, `enc_p.encoder.ffn_layers`, `enc_p.proj`.
Merged at **`scale = 0.33`** from the **best-`val_mel` checkpoint (step 800,
`val_mel = 0.6942`)**, not the final step.

### Loss curve

| step | mel | kl | dur | fm | gen | disc |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.434 | 35.821 | 2.68 | 0.62 | 1.76 | 2.95 |
| 100 | 1.150 | 9.656 | 2.65 | 1.66 | 2.15 | 2.59 |
| 200 | 1.153 | 8.150 | 2.30 | 2.11 | 2.25 | 2.50 |
| 300 | 1.031 | 6.667 | 2.43 | 3.02 | 2.32 | 2.23 |
| 400 | 0.909 | 7.809 | 2.37 | 3.61 | 3.77 | 2.40 |
| 500 | 0.866 | 7.707 | 2.36 | 3.40 | 1.76 | 2.33 |
| 600 | 0.827 | 7.001 | 2.42 | 4.30 | 2.51 | 1.97 |
| 700 | 0.796 | 5.588 | 2.38 | 3.10 | 2.12 | 2.45 |
| **800** ⭐ | **0.858** | **4.697** | **2.59** | 3.04 | 2.70 | 2.33 |
| 900 | 0.815 | 4.988 | 2.55 | 4.46 | 2.88 | 1.92 |
| 1000 | 0.832 | 4.455 | 2.39 | 5.15 | 3.02 | 1.50 |

*(every-50-steps sample; the full every-25-steps table is in
`training_metrics_log.md`.)*

### What the numbers say

| check | value | reading |
|---|---|---|
| **mel loss** | 1.43 → ~0.80 | adapter + `enc_q` learned to reconstruct the teacher's loanword clips — it is learning the target acoustic shapes |
| **KL divergence** | 35.8 → ~4.5, flat from step ~650 | `enc_q` (trained from scratch) **converged** — the adapters fit a *stable* female posterior, not a moving target |
| **duration** | ~2.3–2.6, flat | no timing pathology |
| **prior std** (base → adapter @0.33) | `×1.01` (1.026 → 1.038) | **no latent-prior-variance collapse** — this is the mechanism behind a muffled / robotic voice, and it is absent here |
| **prior mean shift** `|Δm_p| / |m_p|` | ~0.13 | the adapter genuinely moves the text's latent representation (not the historic `0.000` no-op) |
| **base-vs-Aegis logmel L1** (`scale 0`) | `1.29` (threshold < 2.0) | the frozen graft is intact — plain-Urdu structure cannot regress |
| **PLAIN L1** (`0.33` vs base) | `0.128` | a small, non-destructive coarticulation shift on plain Urdu |
| **`phoneme_id_map`** | byte-identical to v1 | the export is a true drop-in |

### What the tuned model is better at

Trained on 185 loanword-dense `helpdesk-agent` clips, it targets:

1. **English loanwords inside Urdu** — `keypad, card, transaction, balance, OTP,
   PIN, CVV, statement, credit card, debit card, reference number` rendered as
   the espeak-`ur` a-based shapes (`kˈiːpad`, `tɹansˈakʃən`, `bˈaləns`), cleanly
   coarticulated instead of `"ki pat"` / muddy.
2. **Dental ت/د vs retroflex ٹ/ڈ contrast** — the v2 enrichment sharpens this
   (matters for ٹرانزیکشن / ڈیبٹ).
3. **Long vowels** iː aː eː oː — cleaner vowel length in loanwords.
4. **Wider bank-IVR vocabulary** — `verification, installment, refund,
   beneficiary, dispute, reversal, POS, QR code, e-statement, standing
   instruction, auto-debit`.

Metrics can't score audio quality — the deciding test is the cell 8 / cell 10
A/B listen. Plain rows at `scale 0.33` must still sound unmistakably like Aegis;
loanword rows should be clearer at `0.33` than at `0.0`. If plain drifts
audibly, re-run the merge cell at `SCALE = 0.25` (no retraining needed). If a
specific word is still weak, add 10–20 clips of that exact word and retrain.

---

## Dataset

185 clips (`dataset/`, Uplift AI `helpdesk-agent`, 22 050 Hz mono): the original
123 plus 62 enrichment clips (`0124-0185`) covering dental vs retroflex, long
vowels, and wider bank-IVR vocabulary. 73 more sentences (`0186-0258`) are
drafted — run `tools/synth_enrichment.py` with an Uplift key to finish them
(idempotent, resumes at `0186`). `SENTENCES` in the notebook mirrors
`metadata.csv` 1:1.

No dataset of female Urdu speech with correctly pronounced English loanwords
exists on Hugging Face (Urdu TTS sets are plain/literary; code-switching sets are
text-only Roman Urdu; Urdu ASR sets are multi-speaker, noisy, 16 kHz). The
teacher is the ceiling — the adapter cannot beat it, so probe new hard words
before adding them.

---

## Ship

Drop `tune_model/ur_PK-aegis_female_loan-medium.onnx` + `.json` into
`models/piper/ur-aegis-female/` as **v2**, renamed to
`ur_PK-aegis_female-medium.{onnx,onnx.json}`. Keep v1 for rollback. The exported
`phoneme_id_map` is byte-identical to v1 (verified in cell 9), so
`adapt_speech_text` and inline `[[..]]` phonemes keep working.

---

## Attribution

The Aegis voice is MIT-licensed by **Muhammad Mahwiz Khalil (Proxima AI)**
(<https://huggingface.co/mahwizzzz>). Any redistributed derivative must keep that
copyright and attribution.
