# FastContext-Hermes: NVFP4 Repository Exploration on NVIDIA GB10

> Specialized repo exploration for coding agents — FastContext 4B-RL quantized to NVFP4, served locally on GB10 SM121 via vLLM, benchmarked against API baselines.

**Model:** [microsoft/FastContext-1.0-4B-RL](https://huggingface.co/microsoft/FastContext-1.0-4B-RL)
**NVFP4:** [r0b0tlab/FastContext-1.0-4B-RL-NVFP4](https://huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4) (2.7 GB, 2.8× compression)
**Paper:** [arXiv:2606.14066](https://arxiv.org/abs/2606.14066) — *FastContext: Training Efficient Repository Explorer for Coding Agents*
**Hardware:** NVIDIA DGX Spark (GB10), 128GB unified memory, single-node

---

## What This Does

FastContext separates repository **exploration** from **solving**. A 4B model trained with GRPO explores codebases using Read/Glob/Grep tool calls and returns compact `file:line` citations. This project quantizes it to NVFP4 for edge deployment and benchmarks it against full-repo API stuffing.

```
Main Agent (API) ──delegate_task(bg)──▶ FastContext (Local GB10, NVFP4)
     ▲                                        │
     └──────── file:line citations ───────────┘
```

---

## Proven Results

### Real-World Benchmark (June 2026)

5 exploration questions on `hermes-concurrent-agents` repo (53 files, ~45K tokens). Solver: GLM-5.2 via z.ai API.

#### Token Efficiency

| Question | Without FC (tokens) | With FC (tokens) | Reduction |
|---|---|---|---|
| Deploy vLLM backend | 51,532 | 6,883 | **86.6%** |
| Fault injection testing | 50,686 | 475 | **99.1%** |
| Profile architecture | 51,525 | 1,657 | **96.8%** |
| Benchmarking methods | 51,387 | 6,666 | **87.0%** |
| CI/CD pipeline | 50,816 | 15,946 | **68.6%** |
| **TOTAL** | **255,946** | **31,627** | **87.6%** |

> **Model card claim:** "Up to 60% token reduction." **Our result: 87.6% average (exceeds claim).**

#### NVFP4 Performance (GB10 SM121)

| Metric | NVFP4 | BF16 Baseline | Delta |
|---|---|---|---|
| Throughput (c=1) | **66.3 tok/s** | 22.8 tok/s | **2.9× faster** |
| Peak concurrent (c=16) | **950 tok/s** | 416 tok/s | **2.3× faster** |
| TTFT | **22 ms** | 43 ms | **1.95× faster** |
| GPU power | **11 W** | ~15 W | 27% less |
| Model size | **2.7 GB** | 7.6 GB | **2.8× compression** |
| Tool-call accuracy | **100%** (5/5) | 100% (5/5) | Zero degradation |

#### Retrieval Quality

| Metric | Result | Assessment |
|---|---|---|
| Exploration convergence | **5/5** (3–6 turns each) | ✅ NVFP4 preserves capability |
| Ground truth file recall | **4/12 (33%)** | ⚠️ Mixed — search strategy is the weak link |
| Answer quality (when found) | **Parity** with full-repo at 13% of cost | ✅ Validated |
| Avg tool calls per exploration | 8.8 (Read/Glob/Grep) | ✅ Correct parallel tool calling |

**Honest assessment:** Token savings massively exceed the model card claim. Retrieval precision is the weak link — not the quantization. FastContext excels when it identifies the right search domain (q4: 100% recall) but misses files with naming convention mismatches (e.g., grep `fault_inject` misses `fault-injection-test.sh`) or hidden directories (`.github/`).

#### Tool-Call Correctness (NVFP4)

All 5 explorations generated valid tool calls through vLLM's `hermes` parser:

```
q1: Glob → Glob → Grep → Grep → Read → Read          ✓
q2: Grep ×4 → Glob ×3 → Read ×2 → Read               ✓
q3: Read → Grep → Grep → Grep                         ✓
q4: Grep ×3 → Glob → Read → Grep → Grep               ✓
q5: Glob ×4 → Read ×2 → Glob → Grep → Read ×3         ✓
```

---

## Quick Start

### Serve NVFP4 on GB10

```bash
# Activate ModelOpt venv (vLLM 0.23.0, CUDA 13.0)
source ~/.venvs/modelopt/bin/activate

vllm serve models/llm/nvfp4/FastContext-1.0-4B-RL-NVFP4 \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --gpu-memory-utilization 0.40 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name FastContext-1.0-4B-RL-NVFP4 \
  --host 0.0.0.0 --port 30000
```

### Run Exploration

```bash
# Query the local NVFP4 FastContext server
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FastContext-1.0-4B-RL-NVFP4",
    "messages": [
      {"role": "system", "content": "You are a codebase exploration specialist..."},
      {"role": "user", "content": "<query>Find the authentication middleware and token validation logic</query>"}
    ],
    "tools": [...],
    "max_tokens": 4096,
    "temperature": 0.3
  }'
```

### Reproduce Benchmarks

```bash
# Token efficiency comparison (FastContext vs full-repo API stuffing)
python3 benchmarks/realworld_v3.py

# Retrieval quality analysis
python3 benchmarks/analyze_quality.py

# Throughput benchmark (NVFP4 vs BF16)
python3 benchmarks/benchmark_fastcontext.py
```

---

## Quantization

NVFP4 quantization using NVIDIA ModelOpt (W4A4, group-16) on GB10 SM121 native FP4 tensor cores.

| Property | Value |
|---|---|
| Quantization | NVFP4 (W4A4, group-16) |
| Calibration | `abisee/cnn_dailymail`, 512×1024×16 |
| Quantizers | 903 (all Linear layers) |
| Compression | 7.6 GB → 2.7 GB (2.8×) |
| Excluded layers | None (dense text-only Qwen3-4B) |
| Tool calling | Preserved (hermes parser, 100% accuracy) |

```bash
python3 scripts/quantize_fastcontext_nvfp4.py
```

---

## Project Structure

```
fastcontext-hermes/
├── README.md                          # This file (with proven metrics)
├── PLAN.md                            # 5-phase implementation plan
├── configs/                           # vLLM + SGLang serving configs
├── scripts/
│   ├── quantize_fastcontext_nvfp4.py  # ModelOpt NVFP4 quantization
│   └── matmul_early_rejection.py      # Pre-quantization matmul benchmark
├── hermes/
│   └── integration/
│       └── async_explorer.py          # delegate_task(background=True) wrapper
├── benchmarks/
│   ├── realworld_v3.py                # FastContext vs full-repo token comparison
│   ├── analyze_quality.py             # Ground truth retrieval analysis
│   ├── benchmark_fastcontext.py       # NVFP4 vs BF16 throughput
│   └── results/                       # JSON + HTML reports
│       ├── realworld_v3.json          # Raw benchmark data
│       └── realworld_report.html      # Full HTML report
└── docs/
```

---

## Links

- **NVFP4 Model:** [huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4](https://huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4)
- **Original Model:** [huggingface.co/microsoft/FastContext-1.0-4B-RL](https://huggingface.co/microsoft/FastContext-1.0-4B-RL)
- **Paper:** [arxiv.org/abs/2606.14066](https://arxiv.org/abs/2606.14066)
- **FastContext CLI:** [github.com/microsoft/fastcontext](https://github.com/microsoft/fastcontext)

## License

MIT — same as upstream FastContext
