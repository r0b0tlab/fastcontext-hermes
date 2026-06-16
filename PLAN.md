# FastContext-Hermes: Optimized Repo Exploration via Async Subagent Integration

> **Status:** Planning — Phase 0
> **Created:** 2026-06-16
> **Repo target:** github.com/r0b0tlab/fastcontext-hermes
> **Model:** microsoft/FastContext-1.0-4B-RL (Qwen3-4B-Instruct base, MIT)
> **Paper:** arXiv:2606.14066 — *FastContext: Training Efficient Repository Explorer for Coding Agents*

---

## 1. Model Review

### What FastContext Is

A **specialized repository-exploration subagent** that separates codebase search from task solving. Instead of the main coding agent burning context tokens on file reads and grep, it delegates exploration to FastContext, which:

1. Takes a natural-language query ("find the authentication middleware")
2. Issues **parallel** READ / GLOB / GREP tool calls (read-only, never edits)
3. Returns **compact file:line citations** — not raw code dumps

```
Main Agent ──query──▶  FastContext (4B)  ──read/search──▶  Repository
     ▲                       │
     └──── file:line ────────┘
```

### Why It Matters (The Numbers)

| Metric | Without FastContext | With FC-4B-RL |
|--------|--------------------|---------------|
| GPT-5.4 token usage | 418K–818K per task | **166K–701K (↓14–50%)** |
| SWE-bench Pro accuracy (GPT-5.4) | 46.0 | **48.5 (+2.5)** |
| SWE-bench Pro accuracy (GLM-5.1) | 17.5 | **22.5 (+5.0)** |
| Exploration turns as % of all tool calls | **56.2%** | Offloaded entirely |

### Model Specs

| Property | Value |
|----------|-------|
| Base | Qwen/Qwen3-4B-Instruct-2507 |
| Params | 4B, BF16 |
| Context | 262K tokens |
| Tools | READ, GLOB, GREP (read-only) |
| Training | SFT (exploration traces) → GRPO (file/line F1 reward) |
| Output format | `<final_answer>` block with `path/to/file.py:42-58` citations |
| Serving | SGLang or any OpenAI-compatible endpoint |
| License | MIT |

### Strengths

- **Tiny but specialized** — 4B model outperforms 30B-SFT in several benchmarks (GLM-5.1 SWE-bench Pro: 22.5 vs 20.0)
- **Tool-call native** — trained specifically for parallel READ/GLOB/GREP, not generic function calling
- **Compact output** — returns line ranges, not full file contents, keeping solver context clean
- **262K context** — can handle massive repos in a single exploration session
- **Open everything** — MIT model + MIT code + paper + training data

### Limitations to Validate

- **Read-only** — cannot make changes, only locate. The main agent still does the editing.
- **GB10 tool-call validation** — Qwen3 tool-calling format needs verification on GB10 SGLang/vLLM
- **No native NVFP4 weights** — BF16 only; NVFP4 quantization is our value-add
- **SWE-bench-specific training** — may need adaptation for non-SWE repos (ML codebases, infra configs)
- **Only 32 downloads** — very new (released 2026-06-15), limited community validation

---

## 2. Architecture: Hybrid API + Local via Async Subagents

### Core Insight

FastContext's paper architecture maps **exactly** to Hermes's `delegate_task(background=true)` pattern. The main agent (expensive API model) stays focused on reasoning and solving; FastContext (cheap local model on GB10) handles exploration in the background.

```
┌─────────────────────────────────────────────────────────────┐
│                    HERMES CONTROLLER                         │
│                   (Main Agent: API Model)                    │
│              GPT-5.4 / GLM-5.1 / Kimi-K2.6                   │
├──────────────┬──────────────┬───────────────────────────────┤
│              │              │                                │
│  User Task   │  delegate_   │  Receives citations back,      │
│  → parse     │  task(bg)    │  uses focused context to edit  │
│              │      ↓       │  / test / commit               │
├──────────────┴──────────────┴───────────────────────────────┤
│                                                              │
│  ┌─── Subagent A (bg) ───┐  ┌─── Subagent B (bg) ───┐      │
│  │ "Find auth middleware" │  │ "Find test patterns"  │      │
│  │     ↓                  │  │     ↓                  │      │
│  │ FastContext 4B-RL      │  │ FastContext 4B-RL      │      │
│  │ (GB10 localhost:30000) │  │ (GB10 localhost:30000) │      │
│  │ READ/GLOB/GREP         │  │ READ/GLOB/GREP         │      │
│  └────────────────────────┘  └────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            GB10 DGX Spark (Node A)                    │   │
│  │  vLLM 0.23.0 serving FastContext-1.0-4B-RL-NVFP4     │   │
│  │  ~2.4 GB VRAM NVFP4 (W4A4), FlashInfer attn,         │   │
│  │  FP8 KV cache, SM121 native FP4 tensor cores         │   │
│  │  Parallel exploration requests batched automatically │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Three Integration Modes

#### Mode 1: Direct API Wrapper (Simplest)
```python
# Main agent calls FastContext via simple HTTP when it needs context
# FastContext runs as standalone SGLang server on GB10
from fastcontext.agent.agent_factory import make_fastcontext_agent

agent = make_fastcontext_agent(
    trajectory_file=".fastcontext/trajectory.jsonl",
    work_dir="/path/to/repo",
)
answer = await agent.run(
    prompt="Find where database migrations are defined",
    max_turns=6,
    citation=True,
)
# answer = "src/migrations/001_init.py:1-45\nsrc/models/base.py:12-30"
```

#### Mode 2: Hermes Async Subagent (Recommended)
```python
# Main agent dispatches FastContext as background subagent
# Multiple parallel explorations, results arrive asynchronously
result = delegate_task(
    goal="Explore the repo at /path/to/repo using FastContext CLI. "
         "Run: fastcontext --query 'Find authentication middleware and "
         "related test files' --citation --max-turns 6 "
         "--traj /tmp/fc-auth.jsonl. Return the final_answer citations.",
    background=True,  # Non-blocking — main agent continues other work
    toolsets=['terminal'],
)
# Result re-enters conversation when FastContext finishes
```

#### Mode 3: Parallel Fan-Out (Maximum Efficiency)
```python
# Dispatch multiple FastContext explorations in parallel for complex tasks
tasks = [
    {
        "goal": "FastContext: find auth logic in /repo. Run: fastcontext -q "
                "'Find authentication middleware and token validation' "
                "--citation --max-turns 4",
        "toolsets": ["terminal"],
    },
    {
        "goal": "FastContext: find database layer in /repo. Run: fastcontext -q "
                "'Find ORM models, migrations, and query builders' "
                "--citation --max-turns 4",
        "toolsets": ["terminal"],
    },
    {
        "goal": "FastContext: find API routes in /repo. Run: fastcontext -q "
                "'Find all HTTP route handlers and middleware chain' "
                "--citation --max-turns 4",
        "toolsets": ["terminal"],
    },
]
# All three run as background subagents simultaneously
# GB10 batches the requests via continuous batching in SGLang/vLLM
```

---

## 3. Implementation Plan (Phased)

### Phase 0: Project Setup & Scaffolding `[COMPLETE]`
- [x] Create project folder: `/home/r0b0tdgx/projects/fastcontext-hermes/`
- [x] Initialize git repo + remote `github.com/r0b0tlab/fastcontext-hermes`
- [x] Write project README with architecture diagram
- [x] Create vLLM/SGLang serving configs for GB10
- [x] Write async_explorer.py integration wrapper

### Phase 1: NVFP4 Quantization + SM121 Optimization `[NEXT — PRIORITY]`

The model is a standard Qwen3-4B dense transformer — no MoE, no multimodal. This is the simplest NVFP4 quantization target we've worked with. Goal: produce a publishable `r0b0tlab/FastContext-1.0-4B-RL-NVFP4` checkpoint optimized for SM121.

#### 1a: Download + Inspect Source Model
- [ ] Download `microsoft/FastContext-1.0-4B-RL` BF16 weights (~8 GB)
- [ ] Inspect safetensors header: confirm all-BF16, standard Qwen3 key structure
- [ ] Verify `config.json`: `architectures: ["Qwen3ForCausalLM"]`, `hidden_size`, `num_hidden_layers`
- [ ] Confirm tool-calling chat template exists in `tokenizer_config.json`

#### 1b: NVFP4 Quantization via ModelOpt
- [ ] Set up ModelOpt venv: `nvidia-modelopt[hf]>=0.44.0`, `torch+cu130`, `transformers>=5.4`
- [ ] Use `NVFP4_DEFAULT_CFG` (W4A4, group_size=16) — full quantization
  - Dense model, no MoE → no unfuse needed
  - Text-only → no multimodal exclusions needed
  - Standard exclusions (ModelOpt defaults): `*norm*`, `*lm_head*`, biases
  - Qwen3 `embed_tokens` is 152064 × 2560 = divisible by 16 → can quantize (unlike Gemma 4's 262144)
  - If embed_tokens causes issues: add `"*embed_tokens*": False` to exclusion list
- [ ] Calibration: `cnn_dailymail` 512 samples × 1024 tokens × batch 16
- [ ] **Critical**: Add `quant_method: "modelopt_fp4"` to `hf_quant_config.json`
- [ ] Export via `export_hf_checkpoint`, verify `hf_quant_config.json` present
- [ ] Copy from source: `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`, `generation_config.json`
- [ ] **Verify `lm_head.weight` in output** — if `tie_word_embeddings=false`, ModelOpt drops it; manually copy from BF16 source
- [ ] Expected output: ~2.4 GB NVFP4 checkpoint (3.3× compression from 8 GB BF16)

#### 1c: SM121 Serving Validation
- [ ] Serve NVFP4 model on GB10 via vLLM 0.23.0:
  ```bash
  vllm serve r0b0tlab/FastContext-1.0-4B-RL-NVFP4 \
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
      --port 30000
  ```
  **SM121-specific flags:**
  - `--attention-backend flashinfer` — native SM121 attention
  - `--kv-cache-dtype fp8` — FP8 KV cache (NVFP4 KV not available in public PyTorch)
  - `gpu_memory_utilization 0.40` — 4B NVFP4 is tiny (~2.4 GB), leave room for 131K context + batching
  - `--tool-call-parser hermes` — Qwen3 tool calls; verify parser compatibility
- [ ] Smoke test: send a test query with tool-call format
- [ ] Verify tool-calling works: READ/GLOB/GREP execute and return results
- [ ] Validate `<final_answer>` citation output format preserved
- [ ] Measure: VRAM usage, throughput (tok/s), power (W), latency (ms to first token)

#### 1d: BF16 Baseline Comparison
- [ ] Serve BF16 model on GB10 with identical config (minus `--quantization`)
- [ ] Run identical exploration queries on both
- [ ] Compare citation accuracy (file/line F1) — target: ≥95% of BF16
- [ ] Compare throughput — NVFP4 should be faster on SM121 FP4 tensor cores
- [ ] Compare VRAM/power — NVFP4 should use ~3× less VRAM, lower power

#### 1e: Publish NVFP4 Checkpoint
- [ ] Upload to `huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4`
- [ ] Professional model card: credits (Microsoft, NVIDIA ModelOpt), quantization recipe, `base_model: microsoft/FastContext-1.0-4B-RL`, before/after benchmark table
- [ ] No hardware lock-in language — generic "NVFP4-capable NVIDIA GPU"

### Phase 2: Hermes Async Subagent Integration `[AFTER NVFP4 VALIDATED]`
- [ ] Install FastContext CLI (`uv tool install git+https://github.com/microsoft/fastcontext.git`)
- [ ] Point FastContext at local GB10 NVFP4 endpoint
- [ ] Create Hermes skill: `fastcontext-explorer`
- [ ] Wire `async_explorer.py` to use `delegate_task(background=True)`
- [ ] Create `fastcontext` Hermes profile for dedicated exploration tasks
- [ ] Wire into existing skills: `subagent-driven-development`, `github-workflows`

### Phase 3: Benchmark & Validate `[AFTER INTEGRATION]`
- [ ] **Token comparison**: Run same coding task with/without FastContext
  - Measure main-agent token usage (API costs)
  - Measure FastContext local token usage (free, local compute)
  - Target: ≥40% API token reduction on real repos
- [ ] **Latency profile**: Time-to-citations vs manual exploration
  - NVFP4 on GB10 should be <3s per exploration turn (faster than BF16)
  - Parallel fan-out should complete in <10s for 3 queries
- [ ] **Accuracy**: Run on our actual repos (vllm-gb10, DSV4, diffusiongemma)
  - Compare FastContext citations vs manual grep findings
  - Measure recall: did it find all relevant files?
  - Measure precision: how many irrelevant files cited?
- [ ] **Concurrency**: Test parallel exploration requests
  - GB10 continuous batching should handle 8-16 concurrent FastContext queries
  - Measure throughput degradation curve
- [ ] **NVFP4 vs BF16 vs API cost**: Triple comparison report

### Phase 4: Publication `[AFTER ALL VALIDATED]`
- [ ] Final push to `github.com/r0b0tlab/fastcontext-hermes` with all benchmarks
- [ ] HTML report: NVFP4 quantization results + token savings + latency + accuracy
- [ ] X thread: async subagent + FastContext NVFP4 on GB10 in action

---

## 4. Workflow Integration Points

### Where FastContext Accelerates Our Workflows

| Workflow | Current Pain | FastContext Solution |
|----------|-------------|---------------------|
| **New model repo onboarding** | Manual grep/READ to find config, model arch, serving scripts | FC finds architecture files, config, entry points in <5s |
| **vLLM/GB10 kernel debugging** | Search across 1000s of vLLM source files for the right kernel | FC locates exact kernel files + line ranges |
| **Hermes skill development** | Read existing skills to match patterns | FC finds similar skills, patterns, pitfall sections |
| **GitHub PR review** | Read diff + surrounding context + tests | FC finds affected files + related tests + dependencies |
| **Benchmark prep** | Find benchmark scripts, config files, model paths | FC locates all benchmark infrastructure in one pass |

### Hybrid Strategy: API Intelligence + Local Exploration

```
EXPENSIVE (API)                    CHEAP (Local GB10)
─────────────                     ──────────────────
GPT-5.4 / GLM-5.1                 FastContext-1.0-4B-RL
  ↓                                 ↓
Complex reasoning                  Repository exploration
Code generation                    File location
Bug analysis                       Citation generation
Architecture decisions             Pattern matching
  ↓                                 ↓
Solves the task                    Feeds focused context
```

**Cost model:** A typical coding task costs $0.50–2.00 in API tokens for the main agent. Exploration accounts for 40–60% of those tokens. FastContext on GB10 costs $0.00 (local). Net savings: **40–60% API cost reduction** with **improved accuracy**.

---

## 5. Technical Details

### Serving Config

```yaml
# configs/fastcontext-sglang.yaml
model: microsoft/FastContext-1.0-4B-RL
backend: sglang
context_length: 262144
dtype: bfloat16
tp_size: 1
mem_fraction: 0.8
tool_call_parser: qwen
port: 30000
# GB10 specifics
gpu_memory_utilization: 0.85
enforce_eager: false  # Enable CUDA graphs for throughput
```

### Environment Variables

```bash
# FastContext endpoint (local GB10)
export FASTCONTEXT_BASE_URL="http://127.0.0.1:30000/v1"
export FASTCONTEXT_MODEL="FastContext-1.0-4B-RL"
export FASTCONTEXT_API_KEY="local"

# Main agent (API provider)
export MAIN_AGENT_PROVIDER="anthropic"  # or openrouter, etc.
export MAIN_AGENT_MODEL="claude-sonnet-4"
```

### Async Subagent Integration

The `delegate_task(background=True)` feature lets the main agent dispatch FastContext exploration without blocking:

1. **Dispatch:** Main agent fires off N background subagents, each running a FastContext query
2. **Continue:** Main agent keeps working on other aspects of the task
3. **Receive:** When subagents complete, their citations re-enter the main conversation
4. **Act:** Main agent uses the focused citations to make targeted edits

This is the optimal pattern because:
- Exploration (FastContext) and solving (main agent) run **truly concurrently**
- GB10's continuous batching handles multiple exploration requests efficiently
- The main agent's expensive API context stays clean (only receives compact citations)
- Multiple aspects of a repo can be explored in parallel (auth, DB, API routes, tests)

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| NVFP4 quantization degrades tool-call quality | Low | High | Benchmark F1 before/after; keep BF16 as fallback; 4B dense is quantization-friendly |
| Qwen3 tool-call parser incompatible with vLLM on SM121 | Medium | High | Test `hermes` and `qwen` parsers; SGLang fallback |
| ModelOpt export drops `lm_head.weight` (known pitfall) | Medium | Critical | Manually copy from BF16 source post-export; verify in index.json |
| FastContext trained on SWE-bench only, misses ML/infra patterns | Medium | Medium | Validate on our repos before relying on it |
| Background subagent latency > blocking call | Low | Low | GB10 NVFP4 4B is very fast; measure P50/P95 |
| FastContext returns too many/irrelevant citations | Medium | Medium | Tune `--max-turns` and use `--citation` mode |
| `embed_tokens` quantization causes quality regression | Low | Medium | NVFP4_DEFAULT_CFG quantizes it; if F1 drops, exclude `*embed_tokens*` |

---

## 7. Success Criteria

- [ ] NVFP4 checkpoint produced: `r0b0tlab/FastContext-1.0-4B-RL-NVFP4` (~2.4 GB)
- [ ] NVFP4 serves correctly on GB10 SM121 with working tool calls (READ/GLOB/GREP)
- [ ] NVFP4 citation accuracy ≥95% of BF16 baseline
- [ ] NVFP4 throughput ≥ BF16 baseline on SM121 (FP4 tensor core benefit)
- [ ] NVFP4 VRAM ≤3 GB (vs ~8 GB BF16) — massive headroom for context/batching
- [ ] Async subagent integration dispatches and receives citations
- [ ] ≥40% API token reduction on real coding tasks
- [ ] Exploration latency <3s per query on NVFP4 GB10
- [ ] Published repo + HF NVFP4 weights + HTML benchmark report
