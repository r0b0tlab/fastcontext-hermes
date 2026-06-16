# FastContext-Hermes: Async Subagent Repository Exploration on NVIDIA GB10

> Specialized repo exploration for coding agents — local FastContext 4B-RL on GB10 + async Hermes subagents + optional NVFP4 quantization.

**Model:** [microsoft/FastContext-1.0-4B-RL](https://huggingface.co/microsoft/FastContext-1.0-4B-RL)
**Paper:** [arXiv:2606.14066](https://arxiv.org/abs/2606.14066) — *FastContext: Training Efficient Repository Explorer for Coding Agents*
**Hardware:** NVIDIA DGX Spark (GB10), 128GB unified memory, single-node

## What This Does

FastContext separates repository **exploration** from **solving**. A 4B model trained with GRPO explores codebases using READ/GLOB/GREP and returns compact `file:line` citations. This project integrates it into the Hermes Agent workflow via async background subagents:

```
Main Agent (API) ──delegate_task(bg)──▶ FastContext (Local GB10)
     ▲                                        │
     └──────── file:line citations ───────────┘
```

**Result:** Up to 60% fewer API tokens, improved task accuracy, zero local compute cost.

## Quick Start

```bash
# 1. Serve FastContext on GB10
python3 -m sglang.launch_server \
    --model-path microsoft/FastContext-1.0-4B-RL \
    --tool-call-parser qwen \
    --context-length 262144 \
    --dtype bfloat16 \
    --host 0.0.0.0 --port 30000

# 2. Install FastContext CLI
uv tool install git+https://github.com/microsoft/fastcontext.git

# 3. Configure
export FASTCONTEXT_BASE_URL=http://127.0.0.1:30000/v1
export FASTCONTEXT_MODEL=FastContext-1.0-4B-RL
export FASTCONTEXT_API_KEY=local

# 4. Explore
fastcontext --query "Find the authentication middleware and token validation logic" \
    --citation --max-turns 6
```

## Project Structure

```
fastcontext-hermes/
├── PLAN.md                    # Full implementation plan (read this first)
├── README.md                  # This file
├── configs/
│   ├── fastcontext-sglang.yaml    # SGLang serving config for GB10
│   └── fastcontext-vllm.yaml      # vLLM alternative
├── docker/
│   └── Dockerfile.sgl          # Containerized serving
├── hermes/
│   └── integration/
│       ├── async_explorer.py   # delegate_task(background=True) wrapper
│       └── skill_template.md   # Hermes skill definition
├── benchmarks/
│   ├── token_comparison.py     # API token usage with/without FC
│   ├── latency_profile.py      # P50/P95 exploration latency
│   └── results/                # JSON + CSV benchmark outputs
└── docs/
    ├── architecture.md         # Full architecture diagram
    └── nvfp4_plan.md           # NVFP4 quantization roadmap
```

## License

MIT — same as upstream FastContext
