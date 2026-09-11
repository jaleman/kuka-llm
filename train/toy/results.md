# Step 2 toy run — results (2026-09-11, Mac mini M4 16 GB)

Config: `train/toy/lora.yaml` — Qwen3-0.6B-bf16, LoRA on 16 of 28 layers,
2.88M trainable params (0.48%), batch 2, lr 1e-4, 300 iters, seq 2048.
Data: 270 train / 15 valid / 15 test chunks from `../kuka-mcp/knowledge`
(front matter NOT stripped — see lesson 2).

Throughput: ~0.26 it/s, ~640 tokens/s, peak 7.1 GB. Wall time ≈ 20 min.
730k tokens seen ≈ 1.5 epochs over the 270 training chunks.

| Iter | Train loss | Val loss |
|-----:|-----------:|---------:|
|    1 |          – |    2.108 |
|   10 |      2.148 |          |
|   50 |      1.686 |    1.463 |
|  100 |      1.434 |    1.346 |
|  150 |      1.087 |    1.308 |
|  200 |      1.007 |    1.302 |
|  250 |      1.160 |    1.233 |
|  300 |      0.785 |    1.261 |

Test loss 1.464, test perplexity 4.32.

Prompt (greedy, temp 0): *"The KMP 1500P is an autonomous mobile platform.
Before starting the vehicle, the operator must"*

**Before (base model):** enters Qwen3's `<think>` mode and reasons about
"autonomous vehicles" generically — sensors, cameras, licences.

**After (base + adapter):** drops thinking, answers in manual voice:
"…must ensure the following preconditions and requirements are met.
1. Check the battery level. 2. Check the network connection. 3. Check the GPS
device. 4. Check the LED strip. 5. Check the charging station. 6. Check the
charging interface. 7. Check the charging mode. 8. Check the charging voltage…"

Reading: style and vocabulary transferred in 300 steps; facts did not (KMP
vehicles use laser localization, not GPS; the list degenerates into a
"charging …" loop). Train/val gap opened after iter 200 → early overfitting.
