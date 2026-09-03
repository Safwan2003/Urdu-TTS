# Request to Muhammad Mahwiz Khalil (Proxima AI) for the Aegis checkpoint

**Where to send:** open a Discussion on
<https://huggingface.co/mahwizzzz/piper-voice-ur-aegis-female/discussions>
(title: "Request: student .ckpt for a small loanword adapter fine-tune"),
or DM / email if a contact is listed on his HF profile.

---

Subject: Request for the Aegis student checkpoint (ur_PK-aegis_female-medium)

Hi Mahwiz,

Thank you for releasing Aegis and for writing up the OmniVoice distillation
process — the blog post was very clear, and the `aegis-tts` training code was
helpful to read through.

We use `ur_PK-aegis_female-medium` in production for an Urdu-language customer
support line (banking IVR). It is the female Urdu voice we have standardised on.
Our one remaining issue is English loanword pronunciation — words like
"keypad", "transaction", "card", "balance" that appear mid-sentence in Urdu.
espeak-ng's `ur` frontend maps them to Urdu phoneme shapes and the voice
renders them with a strong Urdu vowel colour, which our listeners find hard to
parse.

We would like to train a **small low-rank adapter** (~1% of the generator,
text-encoder only, base weights frozen and merged back to a normal Piper ONNX
afterwards) on a ~120-clip loanword-dense dataset, so plain-Urdu quality and
the voice identity are provably untouched. For that we need a **trainable
checkpoint** that matches the published ONNX.

(The dataset is ~120 short Urdu banking sentences with embedded English
loanwords, synthesised with Uplift AI's Urdu `helpdesk-agent` voice at
22.05 kHz. Since the adapter only touches the text encoder and the decoder
stays frozen at your weights, the voice identity stays yours — the adapter
just carries the loanword pronunciation/timing.)

Would you be willing to share any of:

1. the final Aegis student `.ckpt` (or the last checkpoint before export), or
2. a mid-training checkpoint, or
3. failing that, your student `config.json` / Lightning training config for
   the run (we already have the public Fasih epoch-3206 ckpt from IhorShevchuk
   as the warm-start), so we can reproduce the student and stop at a usable
   checkpoint ourselves.

We will:

- keep the MIT copyright and attribution to "Muhammad Mahwiz Khalil
  (Proxima AI)" on anything we redistribute,
- share the loanword adapter and the dataset back with you, and
- credit Aegis in our deployment notes.

Happy to sign anything or take this over email if you prefer.

Thanks again for the work,
Safwan
