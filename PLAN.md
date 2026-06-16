# FastContext-Hermes: Optimized Repo Exploration via Async Subagent Integration

> **Status:** Phase 0 complete — Phase 1 (NVFP4) planned, gaps closed
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
| Architecture | `Qwen3ForCausalLM` |
| Params | ~4.03B (36 layers, 32 attn heads, 8 KV heads, head_dim=128) |
| Hidden/Intermediate | 2560 / 9728 |
| Precision | BF16 |
| Context | 262K tokens (max_position_embeddings) |
| Vocab | 151936 |
| `tie_word_embeddings` | **True** — no separate lm_head.weight |
| Tools | READ, GLOB, GREP (read-only) |
| Training | SFT (exploration traces) → GRPO (file/line F1 reward) |
| Output format | `<final_answer>` block with `path/to/file.py:42-58` citations |
| Chat template | Qwen `<tool_call>` XML format, `Qwen2Tokenizer` |
| Serving | Any OpenAI-compatible endpoint (vLLM, SGLang) |
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

## 2. Architecture: Two Components, Not One

### Critical Distinction

FastContext is **two components** that must be deployed separately:

1. **Model server (vLLM/SGLang)** — serves the 4B model as an OpenAI-compatible chat completions endpoint with tool-call support. The model generates `<tool_call>` JSON.
2. **FastContext CLI (agent loop)** — Python CLI that calls the model server, parses tool-call responses, executes READ/GLOB/GREP on the filesystem, feeds results back to the model, and returns the final `<final_answer>` citations.

The CLI is the agent orchestrator; the model server is the brain. Both are required.

### Hybrid API + Local via Async Subagents

FastContext's paper architecture maps to Hermes's `delegate_task(background=true)` pattern. The main agent (expensive API model) stays focused on reasoning and solving; FastContext (cheap local model on GB10) handles exploration in the background.

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

The model is a standard Qwen3-4B dense transformer (`Qwen3ForCausalLM`) — no MoE, no multimodal, `tie_word_embeddings=True`. This is the simplest NVFP4 quantization target we've worked with. Goal: produce a publishable `r0b0tlab/FastContext-1.0-4B-RL-NVFP4` checkpoint.

**Key config.json findings (verified from HF):**
- `tie_word_embeddings: true` → **NO separate lm_head.weight**. The ModelOpt lm_head-drop pitfall does NOT apply. embed_tokens.weight IS lm_head.
- `vocab_size: 151936` (divisible by 16 ✓) → embed_tokens can be quantized
- `hidden_size: 2560`, `intermediate_size: 9728` (both divisible by 16 ✓)
- `architectures: ["Qwen3ForCausalLM"]` → standard vLLM registry, no custom loader
- Chat template uses `<tool_call>` XML format → vLLM `--tool-call-parser hermes`

**⚠️ Throughput caveat:** For small dimensions (2560×9728), BF16 tensor cores may outperform NVFP4 (HiDream-O1 precedent: 0.87× at 4096×12288). NVFP4 value here is primarily **memory compression (3×) and power savings**, not necessarily throughput. Matmul test (step 0) will determine the honest framing.

#### 1.0: Matmul Early Rejection Test `[BEFORE ANY QUANTIZATION]`
- [ ] Run `scripts/matmul_early_rejection.py` inside ComfyUI Docker container
  - Tests largest representative matmuls: mlp_down_proj [9728,2560], mlp_gate_proj [2560,9728], attn_q_proj [2560,4096], attn_o_proj [4096,2560]
  - Decision rule: if ALL layers < 1.0×, NVFP4 won't improve throughput → document as memory-only benefit
  - This prevents committing to a multi-hour quantization that produces a slower model
- [ ] If NVFP4 slower at matmul level: reassess plan — still proceed for memory savings (3× smaller, 2.4 GB vs 8 GB) but reframe success criteria honestly

#### 1a: Prior-Art Check + Download + Inspect Source Model
- [ ] Check HF for existing NVFP4 quantizations: `hf search models "FastContext NVFP4"`
- [ ] Check vLLM recipes: `curl recipes.vllm.ai/microsoft/FastContext-1.0-4B-RL`
- [ ] Download `microsoft/FastContext-1.0-4B-RL` BF16 weights (~8 GB) to `/home/r0b0tdgx/models/llm/bf16/microsoft/FastContext-1.0-4B-RL/`
- [ ] Inspect safetensors header: confirm all-BF16, verify key structure matches Qwen3
- [ ] Verify `config.json` properties against plan: `tie_word_embeddings=true`, `architectures`, dimensions

#### 1b: NVFP4 Quantization via ModelOpt
- [ ] Set up ModelOpt venv:
  ```bash
  uv venv ~/.venvs/modelopt --python 3.12
  source ~/.venvs/modelopt/bin/activate
  uv pip install "nvidia-modelopt[hf]>=0.44.0"
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  uv pip install "transformers>=5.4" safetensors accelerate datasets
  ```
- [ ] Run `scripts/quantize_fastcontext_nvfp4.py`:
  ```bash
  python scripts/quantize_fastcontext_nvfp4.py \
      --model-path microsoft/FastContext-1.0-4B-RL \
      --output-dir /home/r0b0tdgx/models/llm/nvfp4/r0b0tlab/FastContext-1.0-4B-RL-NVFP4
  ```
- [ ] Config: `NVFP4_DEFAULT_CFG` (W4A4, group_size=16)
  - Dense model, no MoE → no unfuse needed
  - Text-only → no multimodal exclusions needed
  - `tie_word_embeddings=true` → no lm_head fix needed
  - Standard exclusions (ModelOpt defaults): `*norm*`, biases
  - If embed_tokens quality regression: re-run with `--exclude-embed-tokens`
- [ ] Calibration: `cnn_dailymail` 512 samples × 1024 tokens × batch 16
- [ ] **Critical**: Verify `quant_method: "modelopt_fp4"` in `hf_quant_config.json` (script auto-adds)
- [ ] Copy from source: `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`, `generation_config.json` (script auto-copies)
- [ ] Expected output: ~2.4 GB NVFP4 checkpoint

#### 1c: SM121 Serving Validation
- [ ] Serve NVFP4 model on GB10 via vLLM 0.23.0 (bare metal or Docker):
  ```bash
  # Bare metal (pip wheel)
  vllm serve /path/to/FastContext-1.0-4B-RL-NVFP4 \
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

  # Docker alternative (our standard path):
  # docker run --gpus all --ipc=host -p 30000:30000 \
  #   -v /path/to/model:/mnt/model \
  #   --entrypoint python3 \
  #   vllm/vllm-openai:v0.23.0-aarch64-ubuntu2404 \
  #   -m vllm.entrypoints.openai.api_server \
  #   --model /mnt/model --quantization modelopt \
  #   --kv-cache-dtype fp8 --attention-backend flashinfer \
  #   --tool-call-parser hermes --enable-auto-tool-choice [...]
  ```
  **SM121-specific notes:**
  - `--tool-call-parser hermes` — handles `<tool_call>` XML format (verified from chat template)
  - `--kv-cache-dtype fp8` — NVFP4 KV not available in public PyTorch
  - `gpu_memory_utilization 0.40` — 4B NVFP4 is tiny (~2.4 GB), massive headroom
  - **Note:** `--entrypoint python3` is required for official vLLM Docker image (it has `vllm` as ENTRYPOINT)
- [ ] Verify `Qwen3ForCausalLM` is in vLLM model registry (startup log)
- [ ] Smoke test: send a tool-call chat request, verify model generates `<tool_call>` JSON
- [ ] Validate tool-calling works end-to-end via FastContext CLI (Phase 2 dependency, but verify model output format now)
- [ ] Measure: VRAM usage, throughput (tok/s), power (W), latency (ms to first token)

#### 1d: BF16 Baseline Comparison
- [ ] Serve BF16 model on GB10 with identical config (minus `--quantization`)
- [ ] Run identical queries on both
- [ ] Compare output quality — target: ≥95% citation match
- [ ] Compare throughput — NVFP4 may be equal or slower (small dimensions); document honestly
- [ ] Compare VRAM/power — NVFP4 should use ~3× less VRAM
- [ ] **Gate:** If NVFP4 quality < 90% of BF16, exclude embed_tokens and re-quantize

#### 1e: Publish NVFP4 Checkpoint
- [ ] Upload to `huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4`
- [ ] `base_model: microsoft/FastContext-1.0-4B-RL` in model card YAML
- [ ] Professional model card with honest throughput characterization (based on matmul test)
- [ ] No hardware lock-in language — generic "NVFP4-capable NVIDIA GPU"
- [ ] Credits: Microsoft (FastContext), Qwen (base model), NVIDIA (ModelOpt), CNN/DailyMail (calibration)

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

### Serving Config (NVFP4 — see configs/fastcontext-vllm.yaml)

```yaml
model: FastContext-1.0-4B-RL-NVFP4
quantization: modelopt
dtype: auto
attention_backend: flashinfer
kv_cache_dtype: fp8
gpu_memory_utilization: 0.40
max_model_len: 131072
tool_call_parser: hermes  # handles <tool_call> XML format
```

### BF16 Baseline Config (see configs/fastcontext-sglang.yaml)

Used for quality/throughput comparison. SGLang with Qwen3 tool-call parser, BF16 dtype.

### Environment Variables

```bash
# vLLM model server (component 1)
# No special env needed — vLLM reads from CLI args

# FastContext CLI (component 2)
# The CLI reads these bare names directly:
export BASE_URL="http://127.0.0.1:30000/v1"
export MODEL="FastContext-1.0-4B-RL-NVFP4"
export API_KEY="local"

# For benchmark configs, FastContext uses separate FASTCONTEXT_* vars
# (see benchmark/evaluation/configs/example.env in the repo)

# Main agent (API provider)
export MAIN_AGENT_PROVIDER="anthropic"  # or openrouter, etc.
export MAIN_AGENT_MODEL="claude-sonnet-4"
```

### FastContext CLI Requirements

- Python 3.12+ (system has 3.11.15 — need uv venv with 3.12)
- `uv` package manager
- OpenAI-compatible endpoint (our GB10 vLLM server)
- Installed via: `uv tool install git+https://github.com/microsoft/fastcontext.git`

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

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | **NVFP4 slower than BF16 for small dims** (2560×9728) | **High** | Medium | Matmul early rejection test (step 1.0) before quantizing. If slower, reframe as memory-only benefit (3× smaller). HiDream-O1 precedent: 0.87× at similar dims. |
| R2 | NVFP4 quantization degrades tool-call quality | Low | High | BF16 baseline comparison (step 1d). If <90%, exclude embed_tokens and re-quantize. 4B dense is quantization-friendly. |
| R3 | Qwen3 tool-call parser incompatible with vLLM SM121 | Medium | High | Verified: chat template uses `<tool_call>` XML → `--tool-call-parser hermes` is correct. Test early. Fallback: SGLang with `--tool-call-parser qwen`. |
| R4 | ModelOpt export drops weights | Low | Medium | `tie_word_embeddings=true` → NO separate lm_head.weight to drop. embed_tokens IS lm_head. Script verifies post-export. |
| R5 | FastContext CLI Python 3.12+ requirement blocks integration | Low | Medium | Use `uv tool install` with Python 3.12 venv. System has 3.11.15; uv can install 3.12 alongside. |
| R6 | FastContext trained on SWE-bench only, misses ML/infra patterns | Medium | Medium | Validate on our actual repos (vllm-gb10, DSV4, diffusiongemma) in Phase 3. |
| R7 | Background subagent latency > blocking call | Low | Low | GB10 NVFP4 4B is very fast (~2.4 GB); measure P50/P95. |
| R8 | FastContext returns too many/irrelevant citations | Medium | Medium | Tune `--max-turns` and use `--citation` mode. |
| R9 | `transformers_version` mismatch (4.51 vs >=5.4) | Low | Low | `Qwen3ForCausalLM` is stable across versions. ModelOpt uses its own model loading. |
| R10 | embed_tokens quantization causes quality regression | Medium | Medium | NVFP4_DEFAULT_CFG quantizes it (vocab 151936 div by 16). If F1 drops, `--exclude-embed-tokens`. With tied weights this is the only output projection. |

---

## 7. Success Criteria

### Phase 1 (NVFP4)
- [ ] Matmul early rejection test completed — honest throughput characterization documented
- [ ] NVFP4 checkpoint produced: `r0b0tlab/FastContext-1.0-4B-RL-NVFP4` (~2.4 GB)
- [ ] NVFP4 serves correctly on GB10 SM121 with working tool calls (`<tool_call>` XML)
- [ ] NVFP4 output quality ≥95% match with BF16 baseline (citation comparison)
- [ ] NVFP4 VRAM ≤3 GB (vs ~8 GB BF16) — 3× memory compression achieved
- [ ] NVFP4 throughput characterized honestly (may be equal/slower than BF16 at small dims)

### Phase 2–3 (Integration + Benchmark)
- [ ] FastContext CLI works with local NVFP4 endpoint (Python 3.12+ via uv)
- [ ] Async subagent integration dispatches and receives citations
- [ ] ≥40% API token reduction on real coding tasks
- [ ] Exploration latency characterized for NVFP4 GB10

### Phase 4 (Publication)
- [ ] Published repo + HF NVFP4 weights + HTML benchmark report
- [ ] Model card credits all parties; no hardware lock-in
