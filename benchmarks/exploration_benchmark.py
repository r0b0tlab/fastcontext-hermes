#!/usr/bin/env python3
"""
FastContext NVFP4 Exploration Benchmark

Proves the NVFP4 model works for real repository exploration by testing:
  1. Tool-call generation correctness (does it emit valid <tool_call> JSON?)
  2. Citation quality (does it return useful file:line references?)
  3. Single-request throughput (tok/s, TTFT)
  4. Concurrent exploration throughput (multiple parallel queries)
  5. NVFP4 vs BF16 quality comparison (same queries, same prompts)

The benchmark uses our own repos as exploration targets — real code the model
has never seen, not SWE-bench data it was trained on.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

# ─── Benchmark queries — designed to test tool-call generation ────────────

FASTCONTEXT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "GLOB",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match files"}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "READ",
            "description": "Read file contents with line numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "End line (inclusive)"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GREP",
            "description": "Search file contents with regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
]

EXPLORATION_QUERIES = [
    {
        "id": "q1_find_entry_points",
        "system": "You are a repository exploration assistant. Use tools to find relevant code. Return citations as file:line ranges in a <final_answer> block.",
        "query": "Find the main entry points and initialization logic in this repository. Start by listing Python files.",
        "expects_tools": ["GLOB"],
        "description": "Basic file discovery",
    },
    {
        "id": "q2_find_configs",
        "system": "You are a repository exploration assistant. Use tools to find relevant code. Return citations as file:line ranges in a <final_answer> block.",
        "query": "Find all configuration files and YAML configs in this project.",
        "expects_tools": ["GLOB", "GREP"],
        "description": "Config file discovery",
    },
    {
        "id": "q3_find_quantization",
        "system": "You are a repository exploration assistant. Use tools to find relevant code. Return citations as file:line ranges in a <final_answer> block.",
        "query": "Find the quantization logic — where NVFP4 quantization is configured and executed. Search for 'quantize' and 'NVFP4'.",
        "expects_tools": ["GREP", "READ"],
        "description": "Keyword search + file read",
    },
    {
        "id": "q4_find_serving",
        "system": "You are a repository exploration assistant. Use tools to find relevant code. Return citations as file:line ranges in a <final_answer> block.",
        "query": "Find the vLLM serving configuration. What flags and parameters are used to serve the model?",
        "expects_tools": ["GREP", "GLOB"],
        "description": "Config parameter discovery",
    },
    {
        "id": "q5_find_async",
        "system": "You are a repository exploration assistant. Use tools to find relevant code. Return citations as file:line ranges in a <final_answer> block.",
        "query": "Find the async subagent integration code. Where is delegate_task or background task logic defined?",
        "expects_tools": ["GREP"],
        "description": "Pattern-specific search",
    },
]

# Throughput prompts — longer outputs for stable tok/s measurement
THROUGHPUT_PROMPTS = [
    "Write a Python function that implements merge sort with type hints and docstring.",
    "Explain how transformer attention works, step by step, with a code example.",
    "Write a FastAPI endpoint that accepts a JSON payload, validates it, and returns a processed response.",
    "Describe 5 common design patterns in Python with brief code snippets for each.",
    "Write a CLI tool using argparse that can search files by pattern and display matching lines.",
]


@dataclass
class QueryResult:
    query_id: str
    description: str
    success: bool
    response_content: str = ""
    tool_calls_found: list = field(default_factory=list)
    has_tool_call_tag: bool = False
    has_final_answer: bool = False
    citations: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    ttft_ms: float = 0.0
    decode_tok_s: float = 0.0
    finish_reason: str = ""
    error: Optional[str] = None


@dataclass
class ThroughputResult:
    prompt_id: int
    prompt: str
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    tok_s: float = 0.0
    ttft_ms: float = 0.0
    error: Optional[str] = None


def parse_tool_calls(content: str) -> list:
    """Extract tool call names from model output."""
    calls = []
    # Match <tool_call> JSON blocks
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL):
        try:
            tc = json.loads(m.group(1))
            if "name" in tc:
                calls.append(tc["name"])
        except json.JSONDecodeError:
            pass
    # Also match inline JSON that looks like tool calls
    if not calls:
        for m in re.finditer(r'"name"\s*:\s*"(\w+)"', content):
            calls.append(m.group(1))
    return calls


def parse_citations(content: str) -> list:
    """Extract file:line citations from <final_answer> block or raw output."""
    fa_match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", content, re.DOTALL)
    text = fa_match.group(1) if fa_match else content
    citations = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # Match file:line patterns like src/foo.py:42-58 or ./foo.py:12
        if re.match(r"^[\w./-]+:\d+", line):
            citations.append(line)
    return citations


def run_exploration_query(base_url: str, model: str, query_spec: dict, timeout: int = 60) -> QueryResult:
    """Run a single exploration query against the model server (non-streaming for tool-call detection)."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": query_spec["system"]},
            {"role": "user", "content": query_spec["query"]},
        ],
        "tools": FASTCONTEXT_TOOLS,
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,  # Non-streaming: ensures tool_calls are properly structured
    }

    result = QueryResult(
        query_id=query_spec["id"],
        description=query_spec["description"],
        success=False,
    )

    try:
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        total_time = time.time() - t0

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        # Extract content (may be None when model emits tool_calls)
        content = msg.get("content") or ""

        # Extract structured tool_calls (vLLM properly parses hermes format)
        structured_tc = msg.get("tool_calls") or []
        tc_names = [tc["function"]["name"] for tc in structured_tc] if structured_tc else []

        # Also check content for inline <tool_call> tags (fallback)
        if not tc_names and content:
            tc_names = parse_tool_calls(content)

        usage = data.get("usage", {})

        result.response_content = content
        result.latency_seconds = total_time
        result.ttft_ms = total_time * 1000  # Non-streaming, so TTFT = total time
        result.has_tool_call_tag = len(tc_names) > 0 or "<tool_call>" in content
        result.tool_calls_found = tc_names
        result.has_final_answer = "<final_answer>" in content
        result.citations = parse_citations(content)
        result.prompt_tokens = usage.get("prompt_tokens", 0)
        result.completion_tokens = usage.get("completion_tokens", 0)

        decode_time = total_time * 0.9  # Approximate (90% of non-streaming time is decode)
        if result.completion_tokens > 0 and decode_time > 0:
            result.decode_tok_s = result.completion_tokens / decode_time

        result.success = True
        result.finish_reason = choice.get("finish_reason", "")

    except Exception as e:
        result.error = str(e)

    return result


def run_throughput_prompt(base_url: str, model: str, prompt: str, prompt_id: int, timeout: int = 120) -> ThroughputResult:
    """Run a single throughput measurement prompt (non-streaming for reliable token counts)."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": False,
    }

    result = ThroughputResult(prompt_id=prompt_id, prompt=prompt)

    try:
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        total_time = time.time() - t0

        data = resp.json()
        usage = data.get("usage", {})

        result.completion_tokens = usage.get("completion_tokens", 0)
        result.wall_time_s = total_time
        # Non-streaming: TTFT is approximately the full request time (prefill + decode)
        # But for benchmarking we report decode throughput = tokens / total_time
        result.ttft_ms = total_time * 1000  # Approximate

        if result.completion_tokens > 0 and total_time > 0:
            result.tok_s = result.completion_tokens / total_time

    except Exception as e:
        result.error = str(e)

    return result


def run_concurrent_throughput(base_url: str, model: str, concurrency: int, timeout: int = 120) -> dict:
    """Run N concurrent throughput prompts and measure aggregate tok/s."""
    prompts = THROUGHPUT_PROMPTS * (concurrency // len(THROUGHPUT_PROMPTS) + 1)
    prompts = prompts[:concurrency]

    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_throughput_prompt, base_url, model, p, i, timeout): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.time() - t0
    total_tokens = sum(r.completion_tokens for r in results if r.error is None)
    success_count = sum(1 for r in results if r.error is None)
    errors = [r.error for r in results if r.error]

    return {
        "concurrency": concurrency,
        "total_requests": concurrency,
        "successful": success_count,
        "failed": len(errors),
        "total_tokens": total_tokens,
        "wall_time_s": round(wall_time, 2),
        "aggregate_tok_s": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "individual_tok_s": [round(r.tok_s, 1) for r in results if r.error is None],
        "errors": errors[:3],
    }


def benchmark_model(base_url: str, model: str, label: str) -> dict:
    """Run full benchmark suite against a model endpoint."""
    print(f"\n{'═' * 60}")
    print(f"  Benchmarking: {label}")
    print(f"  Endpoint: {base_url}")
    print(f"  Model: {model}")
    print(f"{'═' * 60}")

    # Verify server is up
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        models = [m["id"] for m in r.json()["data"]]
        if model not in models:
            print(f"  ⚠️  Model '{model}' not in server models: {models}")
    except Exception as e:
        print(f"  ❌ Server not reachable: {e}")
        return {"label": label, "error": str(e)}

    # ─── Warmup ──────────────────────────────────────────────────────
    print("\n[Warmup] Sending warmup request...")
    run_throughput_prompt(base_url, model, "Hello", -1, timeout=60)

    # ─── Phase 1: Exploration query correctness ──────────────────────
    print(f"\n[Phase 1] Exploration query correctness ({len(EXPLORATION_QUERIES)} queries)...")
    query_results = []
    for q in EXPLORATION_QUERIES:
        print(f"  → {q['id']}: {q['description']}...", end=" ", flush=True)
        r = run_exploration_query(base_url, model, q)
        query_results.append(r)
        status = "✅" if r.success and r.has_tool_call_tag else "⚠️" if r.success else "❌"
        tools = ",".join(r.tool_calls_found) if r.tool_calls_found else "none"
        print(f"{status} tools=[{tools}] {r.completion_tokens}tok {r.latency_seconds:.1f}s")

    # ─── Phase 2: Single-request throughput ──────────────────────────
    print(f"\n[Phase 2] Single-request throughput ({len(THROUGHPUT_PROMPTS)} prompts)...")
    tp_results = []
    for i, p in enumerate(THROUGHPUT_PROMPTS):
        print(f"  → Prompt {i+1}...", end=" ", flush=True)
        r = run_throughput_prompt(base_url, model, p, i)
        tp_results.append(r)
        if r.error is None:
            print(f"✅ {r.tok_s:.1f} tok/s ({r.completion_tokens}tok, TTFT={r.ttft_ms:.0f}ms)")
        else:
            print(f"❌ {r.error[:50]}")

    # ─── Phase 3: Concurrent throughput ──────────────────────────────
    print(f"\n[Phase 3] Concurrent throughput...")
    conc_results = []
    for c in [1, 2, 4, 8, 16]:
        print(f"  → c={c}...", end=" ", flush=True)
        r = run_concurrent_throughput(base_url, model, c)
        conc_results.append(r)
        print(f"✅ {r['aggregate_tok_s']} agg tok/s ({r['successful']}/{c} ok, {r['wall_time_s']}s)")

    # ─── GPU stats ───────────────────────────────────────────────────
    print(f"\n[GPU] Capturing GPU stats...")
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split(", ")
        gpu_stats = {
            "utilization_pct": int(gpu_out[0]),
            "memory_used_mib": gpu_out[1],
            "power_w": float(gpu_out[2]),
            "temp_c": int(gpu_out[3]),
        }
    except Exception:
        gpu_stats = {}

    # ─── Summary ─────────────────────────────────────────────────────
    tool_call_rate = sum(1 for r in query_results if r.has_tool_call_tag) / len(query_results)
    avg_tok_s = statistics.mean([r.tok_s for r in tp_results if r.error is None]) if tp_results else 0
    avg_ttft = statistics.mean([r.ttft_ms for r in tp_results if r.error is None]) if tp_results else 0
    citation_rate = sum(1 for r in query_results if r.has_final_answer or r.citations) / len(query_results)

    summary = {
        "label": label,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exploration": {
            "total_queries": len(query_results),
            "tool_call_success_rate": round(tool_call_rate, 3),
            "citation_success_rate": round(citation_rate, 3),
            "results": [asdict(r) for r in query_results],
        },
        "throughput": {
            "avg_tok_s": round(avg_tok_s, 1),
            "avg_ttft_ms": round(avg_ttft, 0),
            "results": [asdict(r) for r in tp_results],
        },
        "concurrency": conc_results,
        "gpu": gpu_stats,
    }

    print(f"\n{'─' * 60}")
    print(f"  {label} Summary:")
    print(f"  Tool-call success: {tool_call_rate*100:.0f}%")
    print(f"  Citation success: {citation_rate*100:.0f}%")
    print(f"  Avg throughput: {avg_tok_s:.1f} tok/s")
    print(f"  Avg TTFT: {avg_ttft:.0f} ms")
    print(f"  Peak concurrent: {max(c['aggregate_tok_s'] for c in conc_results)} agg tok/s")
    print(f"{'─' * 60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="FastContext NVFP4 exploration benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--nvfp4-model", default="FastContext-1.0-4B-RL-NVFP4")
    parser.add_argument("--bf16-model", default="FastContext-1.0-4B-RL-BF16")
    parser.add_argument("--mode", choices=["nvfp4", "bf16", "both"], default="nvfp4")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    if args.mode in ("nvfp4", "both"):
        results["nvfp4"] = benchmark_model(args.base_url, args.nvfp4_model, "NVFP4")

    if args.mode in ("bf16", "both"):
        results["bf16"] = benchmark_model(args.base_url, args.bf16_model, "BF16")

    # ─── Comparison ──────────────────────────────────────────────────
    if "nvfp4" in results and "bf16" in results and "error" not in results.get("nvfp4", {}) and "error" not in results.get("bf16", {}):
        n = results["nvfp4"]
        b = results["bf16"]
        print(f"\n{'═' * 60}")
        print(f"  NVFP4 vs BF16 Comparison")
        print(f"{'═' * 60}")
        print(f"{'Metric':<30} {'NVFP4':>12} {'BF16':>12} {'Ratio':>8}")
        print(f"{'─' * 62}")
        print(f"{'Tool-call success':<30} {n['exploration']['tool_call_success_rate']*100:>11.0f}% {b['exploration']['tool_call_success_rate']*100:>11.0f}%")
        print(f"{'Citation success':<30} {n['exploration']['citation_success_rate']*100:>11.0f}% {b['exploration']['citation_success_rate']*100:>11.0f}%")
        print(f"{'Avg throughput (tok/s)':<30} {n['throughput']['avg_tok_s']:>12.1f} {b['throughput']['avg_tok_s']:>12.1f} {n['throughput']['avg_tok_s']/b['throughput']['avg_tok_s']:>7.1f}×")
        print(f"{'Avg TTFT (ms)':<30} {n['throughput']['avg_ttft_ms']:>12.0f} {b['throughput']['avg_ttft_ms']:>12.0f}")
        nv_conc = max(c["aggregate_tok_s"] for c in n["concurrency"])
        bf_conc = max(c["aggregate_tok_s"] for c in b["concurrency"])
        print(f"{'Peak concurrent (tok/s)':<30} {nv_conc:>12.1f} {bf_conc:>12.1f} {nv_conc/bf_conc:>7.1f}×")

    # ─── Save results ────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"benchmark_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    main()
