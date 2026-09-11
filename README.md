# kuka-llm — a KUKA AMR model with the documentation baked in

Companion project to [kuka-mcp](../kuka-mcp). That repo turns official KUKA
PDFs into a cleaned, chunked markdown corpus and serves it over MCP. This
repo consumes that corpus and produces an open-weights model that has been
taught KUKA AMR content directly, so it can answer without retrieval.

The two projects stay linked:

- **Input**: `../kuka-mcp/knowledge/*.md` is the raw training text.
- **Output**: a merged, quantized model served behind an OpenAI-compatible
  endpoint that the MCP server can later call.

## Target hardware

NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64 Linux). LoRA on an
8B–14B base in bf16 is the sweet spot; QLoRA on 70B is possible but slow.
Use NVIDIA NGC containers for PyTorch — aarch64 pip wheels lag behind x86.

Development of data and eval scripts also works on a Mac or any Linux box;
only `train/` needs the GPU.

## Layout

```
data/     turn knowledge/*.md into training sets (outputs in data/out/, gitignored)
train/    continued-pretraining and SFT configs and launch scripts
eval/     held-out questions and the LLM-judge scorer
serve/    vLLM / llama.cpp configs, OpenAI-compatible endpoint
lessons/  side-by-side lessons (Java/Rust reader learning Python + ML)
docs/     background notes
PLAN.md   status dashboard + progress log — read before doing planned work
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: count tokens in the corpus

```bash
python data/count_tokens.py ../kuka-mcp/knowledge
```

See [PLAN.md](PLAN.md) for the full step list.
