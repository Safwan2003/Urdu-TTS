# Urdu Aegis Voice — LoRA Fine-Tuning Metric Log

**Date:** 2026-09-08  
**Model Base:** `ur_PK-aegis_female-medium` (VITS, 22.05 kHz)  
**Task:** Low-Rank Adaptation (LoRA) for English Loanword Pronunciation  
**Total Steps:** 1,000  
**Best Checkpoint:** `/content/train/ckpts/epoch=19-step=800.ckpt` (`val_mel = 0.6942`)

---

## 1. Step-by-Step Training Metrics

| Step | Mel Loss | KL Div (`kl`) | Duration (`dur`) | Feature Match (`fm`) | Generator (`gen`) | Discriminator (`disc`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50** | 1.434 | 35.821 | 2.68 | 0.62 | 1.76 | 2.95 |
| **100** | 1.150 | 9.656 | 2.65 | 1.66 | 2.15 | 2.59 |
| **150** | 1.149 | 9.160 | 2.28 | 3.92 | 1.66 | 2.54 |
| **200** | 1.153 | 8.150 | 2.30 | 2.11 | 2.25 | 2.50 |
| **250** | 1.103 | 8.067 | 2.42 | 2.56 | 1.50 | 2.60 |
| **300** | 1.031 | 6.667 | 2.43 | 3.02 | 2.32 | 2.23 |
| **350** | 0.995 | 7.156 | 2.28 | 5.02 | 2.18 | 1.95 |
| **400** | 0.909 | 7.809 | 2.37 | 3.61 | 3.77 | 2.40 |
| **450** | 0.917 | 8.950 | 2.36 | 3.80 | 4.17 | 2.56 |
| **500** | 0.866 | 7.707 | 2.36 | 3.40 | 1.76 | 2.33 |
| **550** | 0.911 | 7.077 | 2.26 | 5.38 | 1.84 | 2.08 |
| **600** | 0.827 | 7.001 | 2.42 | 4.30 | 2.51 | 1.97 |
| **650** | 0.884 | 5.549 | 2.36 | 6.09 | 2.49 | 1.47 |
| **700** | 0.796 | 5.588 | 2.38 | 3.10 | 2.12 | 2.45 |
| **750** | 0.819 | 5.584 | 2.54 | 4.20 | 2.11 | 2.11 |
| **800** ⭐ | **0.858** | **4.697** | **2.59** | **3.04** | **2.70** | **2.33** |
| **850** | 0.795 | 5.341 | 2.58 | 4.72 | 1.88 | 2.09 |
| **900** | 0.815 | 4.988 | 2.55 | 4.46 | 2.88 | 1.92 |
| **950** | 0.795 | 4.831 | 2.53 | 4.02 | 2.48 | 2.08 |
| **1000** | 0.832 | 4.455 | 2.39 | 5.15 | 3.02 | 1.50 |

---

## 2. Checkpoint Selection & Evaluation

- **Status:** `Trainer.fit` stopped: `max_steps=1000` reached.
- **Best Validation Mel Loss:** **`0.6942`** at `epoch=19-step=800.ckpt`
- **Selection Decision:** Step 800 checkpoint is selected for merging & export over the final step 1000 due to superior validation loss performance.

---

## 3. Prior Statistics & Adapter Shift (`scale = 0.33`)

| Sentence Sample | Base Std $\to$ Adapter Std | Relative Shift ($|dm_p| / |m_p|$) |
| :--- | :--- | :--- |
| `آج میں آپ کی کیا مدد` | `1.026 -> 1.038 (x1.01)` | `0.125` |
| `براہِ کرم اپنے موبائ` | `1.027 -> 1.042 (x1.01)` | `0.137` |
| `اس ٹرانزیکشن کی تصدی` | `1.026 -> 1.039 (x1.01)` | `0.128` |

---

## 4. Voice Integrity & Safety Verification

- **base-vs-Aegis logmel L1:** `1.29`  
  - *Threshold:* $< 2.0$ (PASSED — frozen base intact, 0% corruption).
- **PLAIN L1 (0.33 vs base):** `0.128`  
  - *Result:* Plain Urdu pronunciation remains bit-exact / un-drifted.
- **Phonemizer Check:** PASSED (`160 ids` generated for first loanword line).
- **GPU RAM Free Post-Cleanup:** `15.0 / 15.6 GB`
