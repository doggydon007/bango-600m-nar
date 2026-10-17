# Benchmarks

Every number in this document was produced by the code in this repository and read from `results/*.json`. Nothing here is copied from another project's published figures.

## Protocol

| | |
|---|---|
| Task | pick one of exactly three options for a state + question |
| Evaluation set | the official test split of `tasksource/tasksource-jev-typed-decisions`, filtered to rows that are exactly 3 options and `license_use == "commercial"` |
| Items evaluated | 1,340 |
| Trained on any evaluation item | no |
| Scoring | argmax over the 3-way softmax; macro-F1 over the 3 classes |
| Calibration | expected calibration error over 10 equal-width confidence bins |
| Training scale (this checkpoint) | 12.4M tokens over 58,368 decisions |
| Training scale (frozen backbone, earlier run) | 36.3M tokens over 292,780 rows (2 passes) |
| Total training behind this artifact | 49M tokens over 351,148 examples |

The Laya checkpoints (`convaiinnovations/laya` and its `typed-decisions` subfolder) were run by us on the **same items in the same order** through the **same** A/B/C funnel. An alignment check verified the evaluation set matched exactly, and both runs completed with zero errors.

## Accuracy

| Model | Params | Accuracy | Macro-F1 |
|---|---|---|---|
| **Bango-600M-NAR** | 0.6B | **0.669** | **0.667** |
| laya-typed-decisions | 0.4B | 0.487 | 0.485 |
| laya (zero-shot) | 0.4B | 0.472 | 0.472 |
| Majority-class baseline | — | 0.364 | — |
| Random (3 options) | — | 0.333 | — |

Across the 42 sources both models were scored on, Bango-600M-NAR wins **33/42** (79%). Grouped accuracy: **0.686** vs Laya **0.499**.

## Calibration

| Metric | Value |
|---|---|
| Accuracy | 0.669 |
| Mean confidence | 0.680 |
| Confidence − accuracy gap | +0.011 |
| Expected calibration error | **0.029** |

A small gap means the reported probability can be used as a routing signal: send low-confidence decisions to a human instead of trusting the argmax.

## Speed

| Metric | Value |
|---|---|
| Single decision | 533 ms |
| Single-stream throughput | 1.9 decisions/sec |
| Measurement host | 8 vCPU (torch float32) |
| Input length | p50 124 tokens, p95 766 tokens (measured untruncated; the released path truncates to 512) |

One forward pass per decision: there is no decoding loop, so latency does not grow with answer length.

> The batched figure pads each batch to its longest item (p95 = 766 tokens), so it is not a throughput-optimised measurement. Quote the single-stream figure.

## Per-source accuracy

All 42 sources with at least 15 evaluation items, most-populated first.

| Source | n | Bango-600M-NAR | laya-typed-decisions | Δ |
|---|---|---|---|---|
| `WANLI` | 69 | 0.739 | 0.652 | +0.087 |
| `FOL-nli` | 62 | 0.645 | 0.371 | +0.274 |
| `knowref60k` | 56 | 0.661 | 0.554 | +0.107 |
| `social_i_qa` | 51 | 0.647 | 0.451 | +0.196 |
| `procedural-typed-decisions/policy_applicability` | 37 | 0.595 | 0.270 | +0.324 |
| `spartqa-yn` | 35 | 0.657 | 0.486 | +0.171 |
| `corr2cause` | 35 | 0.657 | 0.257 | +0.400 |
| `procedural-typed-decisions/record_eligibility` | 30 | 0.700 | 0.400 | +0.300 |
| `procedural-typed-decisions/event_state_reconstruction` | 29 | 1.000 | 0.759 | +0.241 |
| `procedural-typed-decisions/multi_view_adjudication` | 29 | 0.966 | 0.379 | +0.586 |
| `procedural-typed-decisions/state_perturbation` | 28 | 0.500 | 0.500 | +0.000 |
| `procedural-typed-decisions/policy_under_uncertainty` | 28 | 0.893 | 0.536 | +0.357 |
| `hatexplain/majority` | 24 | 0.417 | 0.417 | +0.000 |
| `multilingual/multilingual-sentiments/all` | 24 | 0.625 | 0.458 | +0.167 |
| `twitter-financial-news-sentiment` | 24 | 0.833 | 0.667 | +0.167 |
| `breaking_nli` | 24 | 1.000 | 0.875 | +0.125 |
| `SpaceNLI` | 24 | 0.958 | 0.458 | +0.500 |
| `newsmtsc` | 24 | 0.583 | 0.667 | -0.083 |
| `wands` | 24 | 0.542 | 0.250 | +0.292 |
| `CONDAQA` | 24 | 0.500 | 0.417 | +0.083 |
| `headline_cause/en_simple` | 24 | 0.708 | 0.083 | +0.625 |
| `vitaminc` | 23 | 0.826 | 0.609 | +0.217 |
| `gap` | 23 | 0.478 | 0.522 | -0.043 |
| `lonli` | 23 | 0.913 | 0.565 | +0.348 |
| `wice` | 23 | 0.696 | 0.696 | +0.000 |
| `scientific-exaggeration-detection` | 23 | 0.609 | 0.522 | +0.087 |
| `tab/identifier` | 23 | 0.522 | 0.391 | +0.130 |
| `folio` | 23 | 0.391 | 0.478 | -0.087 |
| `naturallogic` | 22 | 0.818 | 0.500 | +0.318 |
| `multilingual/tweet_sentiment_multilingual` | 21 | 0.524 | 0.476 | +0.048 |
| `snli` | 21 | 0.952 | 0.857 | +0.095 |
| `multilingual/americas_nli/all_languages` | 21 | 0.429 | 0.476 | -0.048 |
| `sentfin` | 21 | 0.762 | 0.667 | +0.095 |
| `hope_edi/english` | 21 | 0.809 | 0.476 | +0.333 |
| `redial/preference` | 21 | 0.714 | 0.809 | -0.095 |
| `equate` | 21 | 0.524 | 0.381 | +0.143 |
| `propsegment/nli` | 21 | 0.857 | 0.619 | +0.238 |
| `mams/atsa` | 20 | 0.550 | 0.450 | +0.100 |
| `shell-safety-v2` | 20 | 0.650 | 0.250 | +0.400 |
| `syntactic-augmentation-nli` | 20 | 0.500 | 0.500 | +0.000 |
| `temporal-nli` | 18 | 0.667 | 0.500 | +0.167 |
| `nan-nli` | 15 | 0.733 | 0.467 | +0.267 |

## Published figures from other projects (not measured here)

| Source | Published | Protocol |
|---|---|---|
| laya-typed-decisions | 0.766 | `LocalLLaMA/typed-decisions`, a 4-option workflow benchmark, fine-tuned on that benchmark's own train split |
| Jev 1.13.0 | 0.727 | same `LocalLLaMA/typed-decisions` benchmark |

These use a different task shape (4 options), a different test set and a different protocol. They are listed for context only and are never mixed into the tables above.

## What this benchmark does not show

- **Not out-of-distribution.** The evaluation items were never trained on, but they come from the same source pool as the training decisions. Held-out, yes; unseen domain, no.
- **Three options only.** Two-option and five-option tasks are out of scope.
- **English only.**
- **No reasoning.** A single forward pass cannot do chain-of-thought, so tasks that need multi-step inference are not what this benchmark measures.

