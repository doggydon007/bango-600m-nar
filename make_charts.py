"""Benchmark figure for the Bango-600M-NAR model card.

Reads the two result JSONs (our model + Laya head-to-head, measured on the SAME
held-out 3-option decision test set) and renders assets/bango_benchmark.png.

Everything in the figure is a measured number. Published third-party numbers are
never mixed in here -- they are labelled as published wherever they appear.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

C_OURS = "#16a34a"
C_LAYA = "#2563eb"
C_BASE = "#94a3b8"
C_WEAK = "#dc2626"
INK = "#0f172a"


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ours = load("bango_results.json")
    laya = load("bango_vs_laya_results.json")
    if ours is None:
        raise SystemExit("need results/bango_results.json")

    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(15.5, 9.8), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30,
                          left=0.155, right=0.965, top=0.845, bottom=0.155)

    # ---------------- panel A: overall accuracy vs baselines ----------------
    ax = fig.add_subplot(gs[0, 0])
    names, vals, cols = [], [], []
    names.append("Bango-600M-NAR\n(this model, 0.6B)")
    vals.append(ours["accuracy"])
    cols.append(C_OURS)
    if laya and "models" in laya:
        for key, label in (("laya_typed_decisions", "laya-typed-decisions\n(their fine-tune, 0.4B)"),
                           ("laya_base_zeroshot", "laya, zero-shot\n(flagship, 0.4B)")):
            m = laya["models"].get(key)
            if m and "accuracy" in m:
                names.append(label)
                vals.append(m["accuracy"])
                cols.append(C_LAYA)
    names.append("majority-class\nbaseline")
    vals.append(ours["majority_baseline"])
    cols.append(C_BASE)
    names.append("random\n(3 options)")
    vals.append(1 / 3)
    cols.append(C_BASE)

    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=11,
                fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("accuracy  (3-option decisions)", fontsize=10)
    ax.set_title(f"A. Accuracy on the same held-out test set  (n={ours['eval_n']:,})",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # ---------------- panel B: per-source accuracy (our model) ----------------
    # Honest range: show the strongest AND the weakest sources, not just winners.
    ax = fig.add_subplot(gs[0, 1])
    ps = ours.get("per_source_acc", {})
    ranked = sorted(ps.items(), key=lambda kv: -kv[1]["acc"])
    best = ranked[:7]
    worst = ranked[-4:]
    items = best + [("...", None)] + worst
    labels, accs, ns, cols2 = [], [], [], []
    for k, v in items:
        if v is None:
            labels.append(""); accs.append(0.0); ns.append(0); cols2.append("white")
            continue
        labels.append(k if len(k) <= 34 else k[:32] + "..")
        accs.append(v["acc"]); ns.append(v["n"])
        cols2.append(C_OURS if v["acc"] >= ours["majority_baseline"] else C_WEAK)
    y2 = np.arange(len(items))[::-1]
    ax.barh(y2, accs, color=cols2, height=0.62)
    for yy, a, n, v in zip(y2, accs, ns, items):
        if v[1] is None:
            continue
        ax.text(a + 0.012, yy, f"{a:.2f}  (n={n})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y2)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axvline(ours["majority_baseline"], color=C_BASE, linestyle="--", linewidth=1.4)
    ax.text(ours["majority_baseline"] + 0.01, len(items) - 0.55, "majority baseline",
            fontsize=8.5, color="#64748b")
    ax.set_xlabel("accuracy", fontsize=10)
    ax.set_title(f"B. Honest range: 7 strongest and 4 weakest sources  (of {len(ps)})",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # ---------------- panel C: calibration ----------------
    ax = fig.add_subplot(gs[1, 0])
    conf = ours["mean_conf"]
    acc = ours["accuracy"]
    ax.bar([0], [acc], color=C_OURS, width=0.42, label="accuracy")
    ax.bar([1], [conf], color="#0ea5e9", width=0.42, label="mean confidence")
    ax.bar([2], [ours["ece"]], color="#f59e0b", width=0.42, label="ECE (lower = better)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["accuracy", "mean\nconfidence", "ECE"], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.text(0, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(1, conf + 0.02, f"{conf:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(2, ours["ece"] + 0.02, f"{ours['ece']:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("C. Are the probabilities honest?  (confidence vs accuracy gap)",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------- panel D: what it is ----------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    rows = [
        ("Architecture", "Qwen3-0.6B decoder, LoRA r=32 (merged)"),
        ("Head", "Linear(3) -- option A / B / C, single forward pass"),
        ("Params", "616 M total, non-autoregressive, BF16"),
        ("Objective", "soft-target distillation on 3-option decisions"),
        ("Test set", f"{ours['eval_n']:,} decisions, {len(ps)} scored sources"),
        ("License", "Apache-2.0 (model) - commercial-licensed rows only"),
        ("Runs on", "CPU or any GPU; 0.6B BF16 weights"),
        ("Scope", "3-option decisions only; English; no generation"),
    ]
    lat = None
    _lp = os.path.join(RES, "bango_latency.json")
    if os.path.exists(_lp):
        with open(_lp) as f:
            lat = json.load(f)
    if lat:
        rows.insert(4, ("Speed", f"{lat['single_ms_per_decision']:.0f} ms per decision "
                                 f"({lat['single_decisions_per_sec']:.1f}/s, "
                                 f"single-stream, CPU)"))
    ax.set_title("D. What it actually is", fontsize=12.5, fontweight="bold",
                 color=INK, loc="left", pad=10)
    n = len(rows)
    step = (0.90 - 0.02) / (n - 1)
    for i, (k, v) in enumerate(rows):
        yy = 0.90 - i * step
        ax.text(0.0, yy, k, fontsize=10, fontweight="bold", color=INK, va="top")
        ax.text(0.30, yy, v, fontsize=9.5, color="#334155", va="top", wrap=True)
    box = FancyBboxPatch((0.0, -0.02), 1.0, 1.0, boxstyle="round,pad=0.02",
                         linewidth=1.1, edgecolor="#e2e8f0", facecolor="#f8fafc",
                         transform=ax.transAxes, zorder=0)
    ax.add_patch(box)

    # ---------------- header / footer ----------------
    fig.text(0.045, 0.955, "Bango-600M-NAR", fontsize=25, fontweight="bold", color=INK)
    fig.text(0.045, 0.918, "a 0.6B non-autoregressive decision model that answers "
                           "typed 3-option decisions in one forward pass",
             fontsize=12, color="#475569")
    fig.text(0.045, 0.893, "every number measured on the same held-out test set; "
                           "Laya checkpoints run at the same items through the same funnel",
             fontsize=9.5, color="#64748b", style="italic")

    foot = (f"Data: tasksource/tasksource-jev-typed-decisions, commercial-licence rows "
            f"with exactly 3 options. Split: official test split; no test item was ever "
            f"trained on, but test sources overlap the training pool (in-distribution "
            f"held-out), so this is not an out-of-distribution generalisation claim. "
            f"Laya checkpoints were run by us on the same items through the same funnel; "
            f"their published figures are on a different benchmark. No published "
            f"third-party number is presented as our measurement.")
    fig.text(0.045, 0.030, foot, fontsize=8.0, color="#64748b", wrap=True)

    path = os.path.join(OUT, "bango_benchmark.png")
    fig.savefig(path, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
