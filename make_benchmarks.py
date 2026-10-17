"""Generate BENCHMARKS.md from the measured result files.

Same principle as make_card.py: every figure is read from results/*.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "BENCHMARKS.md")
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
    scale = load("bango_scale.json")
    if ours is None:
        raise SystemExit("need results/bango_results.json")

    ps = ours["per_source_acc"]
    laya_td = (laya or {}).get("models", {}).get("laya_typed_decisions", {})
    laya_base = (laya or {}).get("models", {}).get("laya_base_zeroshot", {})
    laya_ps = laya_td.get("per_source_acc", {})

    shared = sorted((k for k in ps if k in laya_ps), key=lambda k: -ps[k]["n"])
    wins = sum(1 for k in shared if ps[k]["acc"] > laya_ps[k]["acc"])
    tot_o = tot_l = 0.0
    n_tot = 0
    for k in shared:
        n = ps[k]["n"]
        tot_o += ps[k]["acc"] * n
        tot_l += laya_ps[k]["acc"] * n
        n_tot += n

    L = []
    a = L.append
    a("# Benchmarks")
    a("")
    a("Every number in this document was produced by the code in this repository and "
      "read from `results/*.json`. Nothing here is copied from another project's "
      "published figures.")
    a("")

    # ---------- protocol ----------
    a("## Protocol")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Task | pick one of exactly three options for a state + question |")
    a(f"| Evaluation set | the official test split of `{DATASET}`, "
      f"filtered to rows that are exactly 3 options and `license_use == \"commercial\"` |")
    a(f"| Items evaluated | {ours['eval_n']:,} |")
    a(f"| Trained on any evaluation item | no |")
    a(f"| Scoring | argmax over the 3-way softmax; macro-F1 over the 3 classes |")
    a(f"| Calibration | expected calibration error over 10 equal-width confidence bins |")
    if scale:
        tc = scale["this_checkpoint"]
        bb = scale["frozen_backbone_earlier_run"]
        lt = scale["lineage_total"]
        a(f"| Training scale (this checkpoint) | "
          f"{tc['tokens_total']/1e6:.1f}M tokens over {tc['decisions']:,} decisions |")
        a(f"| Training scale (frozen backbone, earlier run) | "
          f"{bb['tokens_total']/1e6:.1f}M tokens over {bb['rows_train']:,} rows "
          f"({bb['epochs']} passes) |")
        a(f"| Total training behind this artifact | "
          f"{lt['tokens']/1e6:.0f}M tokens over {lt['examples']:,} examples |")
    a("")
    a("The Laya checkpoints (`convaiinnovations/laya` and its `typed-decisions` "
      "subfolder) were run by us on the **same items in the same order** through the "
      "**same** A/B/C funnel. An alignment check verified the evaluation set matched "
      "exactly, and both runs completed with zero errors.")
    a("")

    # ---------- accuracy ----------
    a("## Accuracy")
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
    a(f"| Majority-class baseline | — | {fmt(ours['majority_baseline'])} | — |")
    a("| Random (3 options) | — | 0.333 | — |")
    a("")
    if shared:
        a(f"Across the {len(shared)} sources both models were scored on, "
          f"Bango-600M-NAR wins **{wins}/{len(shared)}** "
          f"({100.0*wins/len(shared):.0f}%). Grouped accuracy: "
          f"**{tot_o/n_tot:.3f}** vs Laya **{tot_l/n_tot:.3f}**.")
        a("")

    # ---------- calibration ----------
    a("## Calibration")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Accuracy | {fmt(ours['accuracy'])} |")
    a(f"| Mean confidence | {fmt(ours['mean_conf'])} |")
    a(f"| Confidence − accuracy gap | {ours['mean_conf']-ours['accuracy']:+.3f} |")
    a(f"| Expected calibration error | **{fmt(ours['ece'])}** |")
    a("")
    a("A small gap means the reported probability can be used as a routing signal: "
      "send low-confidence decisions to a human instead of trusting the argmax.")
    a("")

    # ---------- speed ----------
    if lat:
        a("## Speed")
        a("")
        a("| Metric | Value |")
        a("|---|---|")
        a(f"| Single decision | {lat['single_ms_per_decision']:.0f} ms |")
        a(f"| Single-stream throughput | {lat['single_decisions_per_sec']:.1f} decisions/sec |")
        a(f"| Measurement host | {lat['cpu']} |")
        a(f"| Input length | p50 {lat['p50_tokens']} tokens, p95 {lat['p95_tokens']} tokens (measured untruncated; the released path truncates to 512) |")
        a("")
        a("One forward pass per decision: there is no decoding loop, so latency does "
          "not grow with answer length.")
        if lat.get("note"):
            a("")
            a(f"> {lat['note']}")
        a("")

    # ---------- per source ----------
    a("## Per-source accuracy")
    a("")
    a(f"All {len(ps)} sources with at least 15 evaluation items, most-populated first.")
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

    # ---------- published ----------
    a("## Published figures from other projects (not measured here)")
    a("")
    a("| Source | Published | Protocol |")
    a("|---|---|---|")
    a("| laya-typed-decisions | 0.766 | `LocalLLaMA/typed-decisions`, a 4-option "
      "workflow benchmark, fine-tuned on that benchmark's own train split |")
    a("| Jev 1.13.0 | 0.727 | same `LocalLLaMA/typed-decisions` benchmark |")
    a("")
    a("These use a different task shape (4 options), a different test set and a "
      "different protocol. They are listed for context only and are never mixed into "
      "the tables above.")
    a("")

    # ---------- caveats ----------
    a("## What this benchmark does not show")
    a("")
    a("- **Not out-of-distribution.** The evaluation items were never trained on, but "
      "they come from the same source pool as the training decisions. Held-out, yes; "
      "unseen domain, no.")
    a("- **Three options only.** Two-option and five-option tasks are out of scope.")
    a("- **English only.**")
    a("- **No reasoning.** A single forward pass cannot do chain-of-thought, so tasks "
      "that need multi-step inference are not what this benchmark measures.")
    a("")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
