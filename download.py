"""Download everything needed to run and reproduce Bango-600M-NAR.

    python download.py          # model + evaluation decisions + a demo run
    python download.py --no-data
    python download.py --out DIR

Needs:  pip install huggingface_hub datasets torch transformers
"""
import argparse
import json
import os
import sys

REPO = "doggydon007/bango-600m-nar"
DATASET = "tasksource/tasksource-jev-typed-decisions"


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fetch_model(out):
    print(f"\n[1/3] model  <-  {REPO}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")
    local = os.path.join(out, "bango-600m-nar")
    snapshot_download(repo_id=REPO, local_dir=local)
    weights = os.path.join(local, "model.safetensors")
    if not os.path.exists(weights):
        sys.exit("model.safetensors missing after download")
    print(f"      -> {local}  ({human(dir_size(local))})")
    return local


def fetch_data(out, limit=None):
    print(f"\n[2/3] evaluation decisions  <-  {DATASET}")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")
    ddir = os.path.join(out, "eval-data")
    os.makedirs(ddir, exist_ok=True)
    items = []
    ds = load_dataset(DATASET, split="test", streaming=True)
    for r in ds:
        opts = r.get("options") or []
        tgt = r.get("target") or []
        if len(opts) != 3 or len(tgt) != 3:
            continue
        if str(r.get("license_use", "")) != "commercial":
            continue
        items.append({
            "state": str(r.get("state", "")).strip(),
            "question": str(r.get("question", "")).strip(),
            "options": [str(o) for o in opts],
            "target": [float(t) for t in tgt],
            "label": int(max(range(3), key=lambda i: float(tgt[i]))),
            "source": str(r.get("source", "")),
        })
        if limit and len(items) >= limit:
            break
    path = os.path.join(ddir, "test_items.json")
    with open(path, "w") as f:
        json.dump(items, f)
    print(f"      -> {path}  ({len(items)} items)")
    return items


def demo(model_dir, items):
    print("\n[3/3] demo")
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except ImportError:
        print("      (skipped: pip install torch transformers)")
        return
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True).eval()

    def decide(state, question, options):
        text = (f"State: {state}\nQuestion: {question}\nOptions:\n"
                f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer:")
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            probs = model(**ids).logits.float().softmax(-1)[0]
        k = int(probs.argmax())
        return k, float(probs[k]), [round(float(p), 3) for p in probs]

    for it in items[:3]:
        k, conf, dist = decide(it["state"], it["question"], it["options"])
        ok = "hit " if k == it["label"] else "miss"
        print(f"      [{ok}] picked {it['options'][k]!r} (p={conf:.3f})  "
              f"dist={dist}  | source={it['source']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".", help="download directory")
    ap.add_argument("--no-data", action="store_true",
                    help="skip the evaluation decisions")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of evaluation decisions")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_dir = fetch_model(args.out)

    items = []
    if not args.no_data:
        try:
            items = fetch_data(args.out, args.limit)
        except Exception as e:                            # noqa: BLE001
            print(f"      (data step failed: {e})")
    else:
        print("\n[2/3] evaluation decisions  (skipped)")

    if items:
        try:
            demo(model_dir, items)
        except Exception as e:                            # noqa: BLE001
            print(f"      (demo failed: {e})")
    else:
        print("\n[3/3] demo  (skipped, no items)")

    print("\nDone.")


if __name__ == "__main__":
    main()
