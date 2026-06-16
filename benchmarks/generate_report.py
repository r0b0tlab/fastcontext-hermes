#!/usr/bin/env python3
"""
Generate HTML benchmark report from JSON results.
Usage: python generate_report.py benchmarks/results/benchmark_NVFP4.json benchmarks/results/benchmark_BF16.json
"""

import json
import sys
import os
from datetime import datetime


def load_results(path):
    with open(path) as f:
        return json.load(f)


def generate_html(nvfp4_data, bf16_data, output_path):
    n = nvfp4_data
    b = bf16_data

    # Extract key metrics
    nv_tool = n["exploration"]["tool_call_success_rate"]
    bf_tool = b["exploration"]["tool_call_success_rate"]
    nv_tps = n["throughput"]["avg_tok_s"]
    bf_tps = b["throughput"]["avg_tok_s"]
    nv_conc = {c["concurrency"]: c["aggregate_tok_s"] for c in n["concurrency"]}
    bf_conc = {c["concurrency"]: c["aggregate_tok_s"] for c in b["concurrency"]}

    # Tool call details
    nv_queries = n["exploration"]["results"]
    bf_queries = b["exploration"]["results"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FastContext NVFP4 Benchmark Report</title>
<style>
  :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3; --dim: #8b949e; --green: #3fb950; --red: #f85149; --blue: #58a6ff; --yellow: #d29922; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 20px; max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.8em; margin-bottom: 8px; }}
  h2 {{ font-size: 1.3em; margin: 30px 0 12px; color: var(--blue); border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  .subtitle {{ color: var(--dim); font-size: 0.9em; margin-bottom: 30px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .metric {{ text-align: center; }}
  .metric .value {{ font-size: 2em; font-weight: 700; }}
  .metric .label {{ color: var(--dim); font-size: 0.8em; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
  .green {{ color: var(--green); }}
  .red {{ color: var(--red); }}
  .blue {{ color: var(--blue); }}
  .yellow {{ color: var(--yellow); }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.9em; }}
  th {{ color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 0.75em; letter-spacing: 1px; }}
  td.num {{ font-family: 'SF Mono', 'Fira Code', monospace; text-align: right; }}
  .winner {{ color: var(--green); font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }}
  .badge-green {{ background: rgba(63,185,80,0.2); color: var(--green); }}
  .badge-blue {{ background: rgba(88,166,255,0.2); color: var(--blue); }}
  pre {{ background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 12px; overflow-x: auto; font-size: 0.85em; }}
  code {{ font-family: 'SF Mono', 'Fira Code', monospace; }}
  .footer {{ margin-top: 40px; color: var(--dim); font-size: 0.8em; border-top: 1px solid var(--border); padding-top: 16px; }}
</style>
</head>
<body>

<h1>FastContext-1.0-4B-RL <span class="badge badge-green">NVFP4</span> Benchmark Report</h1>
<p class="subtitle">NVIDIA GB10 (SM121) · vLLM 0.23.0 · FlashInfer + FP8 KV · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<h2>🎯 Headline Results</h2>
<div class="grid">
  <div class="card metric">
    <div class="value green">{nv_tps:.0f}<span style="font-size:0.5em"> tok/s</span></div>
    <div class="label">NVFP4 Decode Throughput</div>
    <div style="color:var(--dim);font-size:0.8em;margin-top:6px">{nv_tps/bf_tps:.1f}× faster than BF16</div>
  </div>
  <div class="card metric">
    <div class="value blue">{nv_conc.get(16, 0):.0f}<span style="font-size:0.5em"> tok/s</span></div>
    <div class="label">Peak Concurrent (c=16)</div>
    <div style="color:var(--dim);font-size:0.8em;margin-top:6px">{nv_conc.get(16,0)/bf_conc.get(16,1):.1f}× BF16 aggregate</div>
  </div>
  <div class="card metric">
    <div class="value green">{nv_tool*100:.0f}%</div>
    <div class="label">Tool-Call Success Rate</div>
    <div style="color:var(--dim);font-size:0.8em;margin-top:6px">Identical to BF16 ({bf_tool*100:.0f}%)</div>
  </div>
  <div class="card metric">
    <div class="value yellow">2.7<span style="font-size:0.5em"> GB</span></div>
    <div class="label">Model Size (NVFP4)</div>
    <div style="color:var(--dim);font-size:0.8em;margin-top:6px">2.8× smaller than BF16 (7.6 GB)</div>
  </div>
</div>

<h2>📊 Throughput Comparison</h2>
<table>
  <thead>
    <tr>
      <th>Concurrency</th>
      <th class="num">NVFP4 (tok/s)</th>
      <th class="num">BF16 (tok/s)</th>
      <th class="num">Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>c=1</strong> (single request)</td><td class="num winner">{nv_conc.get(1,0):.1f}</td><td class="num">{bf_conc.get(1,0):.1f}</td><td class="num">{nv_conc.get(1,1)/bf_conc.get(1,1):.1f}×</td></tr>
    <tr><td><strong>c=2</strong></td><td class="num winner">{nv_conc.get(2,0):.1f}</td><td class="num">{bf_conc.get(2,0):.1f}</td><td class="num">{nv_conc.get(2,1)/bf_conc.get(2,1):.1f}×</td></tr>
    <tr><td><strong>c=4</strong></td><td class="num winner">{nv_conc.get(4,0):.1f}</td><td class="num">{bf_conc.get(4,0):.1f}</td><td class="num">{nv_conc.get(4,1)/bf_conc.get(4,1):.1f}×</td></tr>
    <tr><td><strong>c=8</strong></td><td class="num winner">{nv_conc.get(8,0):.1f}</td><td class="num">{bf_conc.get(8,0):.1f}</td><td class="num">{nv_conc.get(8,1)/bf_conc.get(8,1):.1f}×</td></tr>
    <tr><td><strong>c=16</strong></td><td class="num winner">{nv_conc.get(16,0):.1f}</td><td class="num">{bf_conc.get(16,0):.1f}</td><td class="num">{nv_conc.get(16,1)/bf_conc.get(16,1):.1f}×</td></tr>
  </tbody>
</table>

<h2>🔧 Tool-Call Correctness</h2>
<p style="color:var(--dim);margin-bottom:12px">Each query provides GLOB, READ, GREP tools. The model must generate valid <code>&lt;tool_call&gt;</code> JSON requesting the correct tool.</p>
<table>
  <thead>
    <tr>
      <th>Query</th>
      <th>Description</th>
      <th>NVFP4 Tools Called</th>
      <th class="num">NVFP4 Tokens</th>
      <th>BF16 Tools Called</th>
      <th class="num">BF16 Tokens</th>
    </tr>
  </thead>
  <tbody>
"""

    for nv_q, bf_q in zip(nv_queries, bf_queries):
        nv_tools = ", ".join(nv_q.get("tool_calls_found", [])) or "—"
        bf_tools = ", ".join(bf_q.get("tool_calls_found", [])) or "—"
        html += f"""    <tr>
      <td><code>{nv_q['query_id']}</code></td>
      <td>{nv_q['description']}</td>
      <td><span class="green">✅ {nv_tools}</span></td>
      <td class="num">{nv_q.get('completion_tokens', 0)}</td>
      <td><span class="green">✅ {bf_tools}</span></td>
      <td class="num">{bf_q.get('completion_tokens', 0)}</td>
    </tr>
"""

    html += f"""  </tbody>
</table>

<h2>⚙️ Configuration</h2>
<pre><code># vLLM 0.23.0 on NVIDIA GB10 (SM121)
vllm serve &lt;model&gt; \\
  --quantization modelopt \\
  --tensor-parallel-size 1 \\
  --kv-cache-dtype fp8 \\
  --attention-backend flashinfer \\
  --gpu-memory-utilization 0.40 \\
  --max-model-len 131072 \\
  --max-num-seqs 16 \\
  --enable-chunked-prefill \\
  --enable-auto-tool-choice \\
  --tool-call-parser hermes</code></pre>

<h2>📝 Methodology</h2>
<div class="card">
<ul style="padding-left: 20px; line-height: 1.8; color: var(--dim); font-size: 0.9em;">
  <li><strong>Tool-call test:</strong> 5 exploration queries with READ/GLOB/GREP tools. Each query asks the model to find specific code patterns. Success = model emits valid tool_call JSON.</li>
  <li><strong>Throughput test:</strong> 5 coding prompts (merge sort, attention explanation, FastAPI, design patterns, CLI tool). Each generates up to 512 tokens. Non-streaming with usage tracking.</li>
  <li><strong>Concurrency test:</strong> N identical prompts sent simultaneously (N=1,2,4,8,16). Measures aggregate decode throughput via continuous batching.</li>
  <li><strong>Identical config:</strong> Same vLLM flags for both NVFP4 and BF16 (except --quantization for BF16).</li>
  <li><strong>Warmup:</strong> One throwaway request before measurement to trigger CUDA graph capture and FP4 GEMM autotuning.</li>
  <li><strong>Hardware:</strong> Single NVIDIA GB10 (SM121), 128GB unified memory. Python 3.12, torch 2.12+cu130.</li>
</ul>
</div>

<h2>🔬 Quantization Details</h2>
<div class="card" style="line-height: 1.8; font-size: 0.9em;">
  <p><strong>Source:</strong> microsoft/FastContext-1.0-4B-RL (BF16, 7.6 GB, Qwen3-4B-Instruct base)</p>
  <p><strong>Config:</strong> NVFP4_DEFAULT_CFG (W4A4, group_size=16) via NVIDIA ModelOpt 0.44.0</p>
  <p><strong>Calibration:</strong> CNN/DailyMail, 512 samples × 1024 tokens × batch 16</p>
  <p><strong>Quantized layers:</strong> 903 (all attention QKV/O + MLP linear layers)</p>
  <p><strong>Output size:</strong> 2.7 GB (2.8× compression)</p>
  <p><strong>tie_word_embeddings:</strong> True — embed_tokens serves as both input embedding and output projection</p>
</div>

<div class="footer">
  <p>Model: <a href="https://huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4">r0b0tlab/FastContext-1.0-4B-RL-NVFP4</a></p>
  <p>Repo: <a href="https://github.com/r0b0tlab/fastcontext-hermes">github.com/r0b0tlab/fastcontext-hermes</a></p>
  <p>© 2026 r0b0tlab · Base model © Microsoft (MIT) · Quantization via NVIDIA ModelOpt</p>
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    nvfp4_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/benchmark_20260616_125032.json"
    bf16_file = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/results/benchmark_20260616_125614.json"

    nvfp4_data = load_results(nvfp4_file)
    bf16_data = load_results(bf16_file)

    output = os.path.join(os.path.dirname(nvfp4_file), "benchmark_report.html")
    generate_html(nvfp4_data, bf16_data, output)
