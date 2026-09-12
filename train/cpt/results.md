# Step 4 rehearsal — continued pretraining on the Mac (2026-09-11)

Qwen3-1.7B-bf16, LoRA rank 16 on all 28 layers (17.4M trainable, 1.01%),
batch 2 × grad-accumulation 2, seq 2048, cleaned dataset from step 3
(946k train tokens: 868k KUKA + 131k FineWeb-Edu). ~0.16 it/s, ~200 tok/s,
peak 10.2 GB. Each iteration is one batch (~1,250 tokens).

## Run 1 — peak LR 1e-4 (stopped at 250: unstable)

| Iter | Train | Val | LR |
|-----:|------:|----:|---:|
| 1 | – | 2.379 | warmup |
| 50 | 2.19 | 2.029 | 1.0e-4 |
| 100 | 2.10 | **2.012** | 1.0e-4 |
| 150 | 2.05 | 2.012 | 9.9e-5 |
| 200 | 1.91 | 2.213 ↑ | 9.8e-5 |
| 250 | 2.19 | 2.315 ↑ | 9.6e-5 |

Train loss flat (1.8–2.4) while val rose → instability, not overfitting.
Log and checkpoints kept in `train/runs/cpt-mac-run1/` (gitignored).

## Run 2 — resumed from run-1 iter 100, peak LR 3e-5, cosine to 3e-6 over 600

| Iter | Train | Val | LR |
|-----:|------:|----:|---:|
| 1 | – | 2.016 | warmup |
| 50 | 1.445 | 2.029 | 3.0e-5 |
| 100 | 1.642 | 1.999 | 3.0e-5 |
| 150 | 1.920 | 1.897 | 2.9e-5 |
| 200 | 1.855 | 1.903 | 2.9e-5 |
| 250 | 1.967 | 1.869 | 2.8e-5 |
| 300 | 1.737 | 1.838 | 2.7e-5 |
| 350 | 1.875 | 1.828 | 2.5e-5 |
| 400 | 1.809 | 1.813 | 2.4e-5 |
| 450 | 2.002 | 1.804 | 2.2e-5 |
| 500 | 1.717 | 1.796 | 2.1e-5 |
| 550 | 1.643 | 1.791 | 1.9e-5 |
| 600 | 1.925 | **1.786** | 1.7e-5 |

**Test loss 1.860, perplexity 6.42.** Total tokens seen across both runs
≈ 960k ≈ 1.0 epoch. Untrained → final: val 2.379 → 1.786 (ppl 10.8 → 6.0).

## Generation note: no chat template for a completion model

`mlx_lm generate` applies the Qwen3 chat template by default. The adapter
was trained on raw text, so chat-formatted prompts are out of distribution
and greedy decoding collapsed to `![](https://p.1000…` on every prompt
(`after-chat-template.txt`). With `--ignore-chat-template` the same adapter
completes normally. `run-mac.sh` now passes the flag for both before/after.

## Before / after (raw completion, greedy, 100 tokens)

**The battery of the KMP 3000P is charged by**
- before: "a 12V DC power supply. The battery is a 12V 12Ah battery. The battery is connected to a 12V 12Ah battery…" (loop)
- after: "a manual charger. The manual charger is a DC-DC converter that can be connected to the battery terminals of the KMP 3000P…" (fluent, wrong)

**When the vehicle reports a localization error, the first step is to**
- before: "check the GPS signal. If the GPS signal is not working, the system will automatically switch to the internal clock…"
- after: "check the localization module. If the localization module is normal, the next step is to check the vehicle's position…"

**To create a new map in KUKA.AMR Fleet, open**
- before: "the KUKA.AMR application, go to the "Map" tab, and click on "New Map". Then, select the "Map" type… use it for simulation or testing."
- after: "the Fleet web interface and navigate to the Map tab. Click the New Map button to enter the map creation page. Fill in the map name and select the map type. Click the Select button to upload a local map file…"

**The KMP 1500P … Before starting the vehicle, the operator must**
- before: "check the following: (1) The vehicle's condition is good… (2) The vehicle's fuel tank is full, and the oil, water, and coolant are sufficient…"
- after: "check the following:\n• The vehicle is not overloaded.\n• The vehicle is not carrying a load.\n• The load is secured to prevent it from falling off…" (loop)

Reading: vocabulary and register moved decisively into the AMR domain (no
more fuel, GPS, 12V); procedures are plausible; specific facts are still
invented; a 1.7B model under greedy decoding loops. Facts need the 8B model
on the Spark, the Q&A stage (steps 5–6), and sampling at inference.
