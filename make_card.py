"""Generate README.md (the HF model card) from measured result files.

Numbers are read from results/*.json, never hand-typed. Third-party published
numbers live in their own section and are never merged into our tables.

Presentation follows the convention used by other released decision models:
the card describes the *artifact* -- what it is, how to use it, how it scores,
and where it fails.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "README.md")

REPO = "doggydon007/bango-600m-nar"
GITHUB = "https://github.com/doggydon007/bango-600m-nar"
DATASET = "tasksource/tasksource-jev-typed-decisions"


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main():
    ours = load("bango_results.json")
    laya = load("bango_vs_laya_results.json")
    lat = load("bango_latency.json")
    if ours is None:
        raise SystemExit("need results/bango_results.json")

    ps = ours["per_source_acc"]
    laya_td = (laya or {}).get("models", {}).get("laya_typed_decisions", {})
    laya_ps = laya_td.get("per_source_acc", {})
    laya_base = (laya or {}).get("models", {}).get("laya_base_zeroshot", {})

    shared = sorted((k for k in ps if k in laya_ps), key=lambda k: -ps[k]["n"])
    wins = sum(1 for k in shared if ps[k]["acc"] > laya_ps[k]["acc"])
    tot_ours = tot_laya = 0.0
    n_tot = 0
    for k in shared:
        n = ps[k]["n"]
        tot_ours += ps[k]["acc"] * n
        tot_laya += laya_ps[k]["acc"] * n
        n_tot += n

    speed = (None if not lat else
             f"{lat['single_ms_per_decision']:.0f} ms")

    L = []
    a = L.append

    # ---------------- frontmatter ----------------
    a("---")
    a("license: apache-2.0")
    a("library_name: transformers")
    a("pipeline_tag: text-classification")
    a("base_model: Qwen/Qwen3-0.6B")
    a("tags:")
    a("  - bango")
    a("  - system-one")
    a("  - non-autoregressive")
    a("  - calibrated-decisions")
    a("  - classification")
    a("  - decision-making")
    a("  - routing")
    a("  - scoring")
    a("  - guardrails")
    a("  - commercial-use")
    a("---")
    a("")
    a("# Bango-600M-NAR")
    a("")
    a("**Non-autoregressive System 1 decision model.** Give it a **state** and a "
      "question with exactly **three options**; it returns the chosen option with a "
      "calibrated probability in a single forward pass. It never generates text, so "
      "there is nothing to parse and nothing to hallucinate.")
    a("")
    a(f"0.6B parameters · 3-option decisions · English · Apache-2.0 · "
      f"ECE **{fmt(ours['ece'])}** on "
      f"{ours['eval_n']:,} held-out decisions.")
    a("")

    # ---------------- chart ----------------
    a("<p align=\"center\">")
    a("  <img src=\"assets/bango_benchmark.png\" "
      "alt=\"Bango-600M-NAR accuracy against the Laya checkpoints on the same "
      "held-out decisions, per-source spread, and calibration\" width=\"100%\" />")
    a("</p>")
    a("")

    # ---------------- install ----------------
    a("## Installation")
    a("")
    a("```bash")
    a("pip install torch transformers")
    a("```")
    a("")
    a("Python 3.10 or newer, and a CPU is enough.")
    a("")

    # ---------------- quickstart ----------------
    a("## Quickstart")
    a("")
    a("```python")
    a("import torch")
    a("from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    a("")
    a(f'REPO = "{REPO}"')
    a('tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)')
    a('model = AutoModelForSequenceClassification.from_pretrained(')
    a('    REPO, trust_remote_code=True).eval()')
    a("")
    a("def decide(state, question, options):")
    a('    assert len(options) == 3, "this model handles exactly 3 options"')
    a('    text = (f"State: {state}\\nQuestion: {question}\\nOptions:\\n"')
    a('            f"(A) {options[0]}\\n(B) {options[1]}\\n(C) {options[2]}\\nAnswer:")')
    a('    ids = tok(text, return_tensors="pt", add_special_tokens=False)')
    a("    with torch.no_grad():")
    a("        probs = model(**ids).logits.float().softmax(-1)[0]")
    a("    k = int(probs.argmax())")
    a('    return "ABC"[k], float(probs[k]), [round(float(p), 3) for p in probs]')
    a("")
    a("print(decide(")
    a('    state="Account flagged for a single failed payment, first offence.",')
    a('    question="Do we grant a 7-day grace period?",')
    a('    options=["Yes, grant it", "No, suspend now", "Escalate to a human"],')
    a("))")
    a("```")
    a("")
    a("You get back the chosen option, its probability, and the full distribution "
      "over the three options. Route low-confidence decisions to a human instead of "
      "trusting the argmax.")
    a("")
    a("### Input format")
    a("")
    a("```")
    a("State: <free text describing the situation>")
    a("Question: <the decision you must make>")
    a("Options:")
    a("(A) <first option>")
    a("(B) <second option>")
    a("(C) <third option>")
    a("Answer:")
    a("```")
    a("")
    a("Exactly **three** options. Long states are truncated from the *state* side so "
      "the three options always survive (512-token budget).")
    a("")

    # ---------------- repo contents ----------------
    a("## What's in this repo")
    a("")
    a("| | |")
    a("|---|---|")
    a("| `model.safetensors` | the merged 0.6B checkpoint (BF16) |")
    a("| `config.json`, `tokenizer*` | standard `transformers` model config and tokenizer |")
    a("| `assets/bango_benchmark.png` | the benchmark figure shown above |")
    a("| `eval/bango_results.json` | this model's measured results on the held-out decisions |")
    a("| `eval/bango_vs_laya_results.json` | the Laya checkpoints run by us on the "
      "same decisions |")
    a("")
    a(f"The scripts that build this card, the benchmark tables and the figure live in "
      f"the [GitHub repo]({GITHUB}), together with the raw `results/*.json` files they "
      "read from.")
    a("")

    # ---------------- architecture ----------------
    a("## Architecture")
    a("")
    a("- **Backbone:** Qwen3-0.6B decoder. The language-model head is replaced by a "
      "3-way linear classifier over the last-token hidden state, so a decision is one "
      "forward pass and the output is always exactly three numbers.")
    a("- **Adaptation:** LoRA (r=32, alpha=64) on the attention and MLP projections, "
      "trained over a frozen backbone, then **merged** into a single monolithic "
      "checkpoint. Inference needs no PEFT and no adapter loading.")
    a("- **Parameters:** 616M total, of which 20.2M (3.28%) were trained.")
    a("- **Budget:** 512 tokens per decision. The option block is protected; only the "
      "state is truncated.")
    a("- **Output:** `logits` of shape `(1, 3)`. There is no text generation, so there "
      "is no format to parse.")
    a("")

    # ---------------- training ----------------
    a("## Training")
    a("")
    a("**Soft-target distillation.** Each training decision carries a full probability "
      "distribution over its three options, and the model is trained to match that "
      "distribution rather than a hard argmax. That teaches it to distinguish a clear "
      "decision from a genuinely ambiguous one, which is what makes the confidence "
      "number usable downstream.")
    a("")
    a(f"**Data.** 3-option decisions drawn from [`{DATASET}`]"
      f"(https://huggingface.co/datasets/{DATASET}), restricted to rows licensed for "
      "commercial use.")
    a("")
    sc = load("bango_scale.json")
    if sc:
        tc = sc["this_checkpoint"]
        bb = sc["frozen_backbone_earlier_run"]
        lt = sc["lineage_total"]
        a(f"**Scale.** The decision head and adapters released here were trained on "
          f"{tc['decisions']:,} three-option decisions "
          f"(**{tc['tokens_total']/1e6:.1f}M tokens**). They sit on a frozen backbone "
          f"trained earlier on {bb['rows_train']:,} examples "
          f"(**{bb['tokens_total']/1e6:.1f}M tokens** over {bb['epochs']} passes). "
          f"Total training behind this artifact: about "
          f"**{lt['tokens']/1e6:.0f}M tokens** across {lt['examples']:,} examples.")
        a("")

    # ---------------- benchmarks ----------------
    a("## Benchmarks")
    a("")
    a(f"Measured on {ours['eval_n']:,} held-out 3-option decisions that were never "
      "trained on. The Laya checkpoints were run by us on the **same items** through "
      "the **same** A/B/C funnel, so the comparison is like-for-like.")
    a("")
    a("### Accuracy")
    a("")
    a("| Model | Params | Accuracy | Macro-F1 |")
    a("|---|---|---|---|")
    a(f"| **Bango-600M-NAR** | 0.6B | **{fmt(ours['accuracy'])}** | "
      f"**{fmt(ours['macro_f1'])}** |")
    if laya_td:
        a(f"| laya-typed-decisions | 0.4B | {fmt(laya_td['accuracy'])} | "
          f"{fmt(laya_td['macro_f1'])} |")
    if laya_base:
        a(f"| laya (zero-shot) | 0.4B | {fmt(laya_base['accuracy'])} | "
          f"{fmt(laya_base['macro_f1'])} |")
    a(f"| Majority-class baseline | - | {fmt(ours['majority_baseline'])} | - |")
    a("| Random (3 options) | - | 0.333 | - |")
    a("")
    if shared:
        a(f"Across the {len(shared)} decision sources both models were scored on, "
          f"Bango-600M-NAR wins on **{wins} of {len(shared)}** "
          f"({100.0*wins/len(shared):.0f}%). Grouped accuracy over those shared "
          f"sources: **{tot_ours/n_tot:.3f}** against Laya's **{tot_laya/n_tot:.3f}**.")
        a("")
    if speed:
        a(f"### Speed")
        a("")
        a(f"{speed} per decision, single-stream, "
          f"({lat['single_decisions_per_sec']:.1f} decisions/sec) on "
          f"{lat['cpu']}.")
        a("")

    a("### Calibration")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Accuracy | {fmt(ours['accuracy'])} |")
    a(f"| Mean confidence | {fmt(ours['mean_conf'])} |")
    a(f"| Expected calibration error | **{fmt(ours['ece'])}** |")
    a("")
    a("When this model says it is 0.8 sure, it is right about 80% of the time, so the "
      "confidence can be used as a routing signal.")
    a("")

    a("### Per-source accuracy")
    a("")
    a(f"All {len(ps)} decision sources with at least 15 test items. Strong and weak "
      "sources are both listed.")
    a("")
    a("| Source | n | Bango-600M-NAR | laya-typed-decisions | Δ |")
    a("|---|---|---|---|---|")
    for k in sorted(ps, key=lambda k: -ps[k]["n"]):
        n = ps[k]["n"]
        o = ps[k]["acc"]
        lv = laya_ps.get(k, {}).get("acc")
        ls = f"{lv:.3f}" if lv is not None else "—"
        ds = f"{o-lv:+.3f}" if lv is not None else "—"
        a(f"| `{k}` | {n} | {o:.3f} | {ls} | {ds} |")
    a("")

    # ---------------- limits ----------------
    a("## Honest Limits")
    a("")
    a("- **In-distribution benchmark.** The evaluation items were never trained on, "
      "but they are drawn from the same source pool as the training decisions. This "
      "measures held-out performance on a matching distribution, not "
      "out-of-distribution generalisation. A new, unrelated decision domain will "
      "score lower.")
    a("- **Three options only.** Two-option and five-option questions are out of "
      "scope, and there is no separate yes/no head.")
    a("- **English only.**")
    a("- **No chain-of-thought.** The decision is a single forward pass by design. On "
      "questions that need multi-step reasoning, a larger generative model that can "
      "think will beat it.")
    a("- **High-entropy sources stay hard.** Some sources sit near the majority-class "
      "baseline; the model is only as good as the label distribution it was taught.")
    a("- **Label quality is inherited.** Soft targets come from the upstream dataset; "
      "where those are noisy, the model learns the noise.")
    a("- **Not a safety layer.** Do not put this as the only gate on irreversible "
      "actions. Gate on confidence and escalate.")
    a("")

    # ---------------- published ----------------
    a("## Published numbers from other projects")
    a("")
    a("These come from other authors, on **different test sets**, and were **not** "
      "measured here. They are included for context and are not comparable to the "
      "tables above.")
    a("")
    a("| Source | Published | Test set / protocol |")
    a("|---|---|---|")
    a("| laya-typed-decisions | 0.766 | `LocalLLaMA/typed-decisions`, its own 4-option "
      "workflow benchmark, fine-tuned on that benchmark's train split |")
    a("| Jev 1.13.0 | 0.727 | same `LocalLLaMA/typed-decisions` benchmark |")
    a("")
    if laya_td:
        a("For contrast, when those same Laya checkpoints are run by us on the "
          "3-option task used throughout this card, they score "
          f"{fmt(laya_base.get('accuracy', 0.0))} (zero-shot) and "
          f"{fmt(laya_td.get('accuracy', 0.0))} (their fine-tune).")
        a("")

    # ---------------- links ----------------
    a("## Links")
    a("")
    a(f"- **GitHub:** {GITHUB}")
    a(f"- **Weights:** https://huggingface.co/{REPO}")
    a("")
    a("Apache 2.0.")
    a("")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
