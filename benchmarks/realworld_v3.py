#!/usr/bin/env python3
"""
FastContext Real-World Benchmark v3
====================================
Uses the EXACT working tool schemas from debug_explore.py.
Properly handles finish_reason=stop as convergence (natural language answer).
"""
import json, os, re, time, subprocess, requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

REPO_PATH = "/home/r0b0tdgx/projects/hermes-concurrent-agents"
FASTCONTEXT_URL = "http://localhost:30000/v1/chat/completions"
SOLVER_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4") + "/chat/completions"
SOLVER_KEY = os.environ.get("GLM_API_KEY", "")
SOLVER_MODEL = "glm-5.2"
FASTCONTEXT_MODEL = "FastContext-1.0-4B-RL-NVFP4"
MAX_TURNS = 6
RESULTS_DIR = "/home/r0b0tdgx/projects/fastcontext-hermes/benchmarks/results"

# ─── Tools (proven working from debug) ───
TOOLS = [
    {"type": "function", "function": {
        "name": "Read",
        "description": "Read the contents of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "The absolute path of the file to read."},
            "offset": {"type": "integer", "description": "Line to start from (1-indexed)."},
            "limit": {"type": "integer", "description": "Number of lines to read."}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "Glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object", "properties": {
            "directory": {"type": "string", "description": "Directory to search in."},
            "pattern": {"type": "string", "description": "Glob pattern."}
        }, "required": ["pattern"]}
    }},
    {"type": "function", "function": {
        "name": "Grep",
        "description": "Search file contents with regex.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search."},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches"], "description": "Output mode."}
        }, "required": ["pattern"]}
    }},
]

def build_system_prompt(work_dir: str) -> str:
    work_dir_ls = ", ".join(sorted(os.listdir(work_dir)))
    return f"""You are a codebase exploration specialist. Find relevant files and return citations.

When done, output:
<final_answer>
/path/to/file.py:10-20
</final_answer>

Working directory: {work_dir}
Files in root: {work_dir_ls}
"""

QUESTIONS = [
    {"id": "q1_deploy_vllm", "question": "How do I deploy concurrent agents with a vLLM backend? What configuration is needed?",
     "ground_truth_files": ["config/vllm/docker-compose.yml", "docs/deployment-guide.md", "setup.sh", "scripts/spawn.sh"]},
    {"id": "q2_fault_injection", "question": "What is the fault injection testing mechanism and how does it work?",
     "ground_truth_files": ["scripts/fault-injection-test.sh", "docs/durability-tests.md"]},
    {"id": "q3_profile_architecture", "question": "How does the orchestrator profile differ from worker profiles?",
     "ground_truth_files": ["profiles/orchestrator/SOUL.md", "profiles/coder-worker/SOUL.md", "profiles/qa-worker/SOUL.md"]},
    {"id": "q4_benchmarking", "question": "What benchmarking methods are available and what metrics do they capture?",
     "ground_truth_files": ["scripts/benchmark.sh", "docs/benchmarking.md"]},
    {"id": "q5_ci_pipeline", "question": "How is the CI/CD pipeline configured? What checks run on push?",
     "ground_truth_files": [".github/workflows/ci.yml"]},
]

@dataclass
class FCResult:
    question_id: str
    success: bool = False
    turns: int = 0
    tool_call_log: list = field(default_factory=list)
    final_answer: str = ""
    fc_input_tokens: int = 0
    fc_output_tokens: int = 0
    local_tokens: int = 0
    citations: list = field(default_factory=list)
    elapsed: float = 0.0

@dataclass
class SolveResult:
    question_id: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed: float = 0.0
    mode: str = ""

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def execute_tool(name, args):
    """Simple tool execution matching debug_explore.py (proven working)."""
    if name == "Read":
        path = args.get("path", "")
        if not os.path.exists(path):
            return f"File not found: {path}"
        if os.path.isdir(path):
            files = os.listdir(path)
            return f"Directory listing ({path}):\n" + "\n".join(sorted(files)[:50])
        with open(path, errors="replace") as f:
            lines = f.readlines()
        offset = args.get("offset", 1) or 1
        limit = args.get("limit", 2000) or 2000
        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        numbered = "".join(f"{i+1}|{lines[i]}" for i in range(start, end))
        return f"```\n{numbered}\n```"
    elif name == "Glob":
        directory = args.get("directory", REPO_PATH)
        pattern = args.get("pattern", "*")
        try:
            result = subprocess.run(["rg", "--files", directory, "--glob", pattern],
                                  capture_output=True, text=True, timeout=10)
            out = result.stdout[:5000] if result.stdout else "No files"
        except:
            out = "Glob error"
        return out
    elif name == "Grep":
        pattern = args.get("pattern", "")
        path = args.get("path", REPO_PATH)
        mode = args.get("output_mode", "files_with_matches")
        cmd = ["rg"]
        if mode == "content":
            cmd.extend(["-n", "-C", "3"])
        else:
            cmd.append("--files-with-matches")
        cmd.extend(["--heading", "--color", "never", pattern, path])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            out = result.stdout[:5000] if result.stdout else "No matches"
        except:
            out = "Grep error"
        return out
    return f"Unknown: {name}"

def extract_citations_from_text(text: str) -> list:
    """Extract file paths from FastContext output text."""
    citations = []
    # Absolute paths with line ranges
    for m in re.finditer(r'(/[a-zA-Z0-9_./-]+):(\d+)[-–]?(\d+)?', text):
        fp = m.group(1)
        s = int(m.group(2))
        e = int(m.group(3)) if m.group(3) else s
        rel = os.path.relpath(fp, REPO_PATH) if fp.startswith(REPO_PATH) else fp
        citations.append({"file": rel, "start": s, "end": e})
    # Relative paths
    for m in re.finditer(r'(?:^|\s)((?:scripts|docs|config|profiles|\.github)/[a-zA-Z0-9_./-]+):(\d+)', text, re.MULTILINE):
        fp = m.group(1)
        s = int(m.group(2))
        if not any(c["file"] == fp for c in citations):
            citations.append({"file": fp, "start": s, "end": s})
    # Just filenames mentioned
    for m in re.finditer(r'`([a-zA-Z0-9_/-]+\.(?:sh|py|yml|yaml|md))`', text):
        fp = m.group(1)
        if not any(c["file"].endswith(fp) for c in citations):
            citations.append({"file": fp, "start": 0, "end": 0})
    return citations

def load_repo_files(repo_path):
    files = {}
    exts = {'.py', '.md', '.yaml', '.yml', '.sh', '.json', '.toml', '.txt'}
    for root, dirs, fns in os.walk(repo_path):
        if '.git' in root: continue
        for fn in fns:
            fp = os.path.join(root, fn)
            if os.path.splitext(fn)[1] in exts or fn in ('Makefile', 'Dockerfile'):
                rp = os.path.relpath(fp, repo_path)
                try:
                    with open(fp, errors='replace') as f:
                        files[rp] = f.read()
                except: pass
    return files

def format_repo_for_context(repo_files):
    return "\n".join(f"─── {fp} ───\n{repo_files[fp]}\n" for fp in sorted(repo_files))

def build_citation_context(citations, repo_files):
    if not citations:
        return "(no citations)"
    parts = []
    seen = set()
    for c in citations:
        fp = c["file"]
        # Try to find the file
        matched = None
        if fp in repo_files:
            matched = fp
        else:
            for key in repo_files:
                if key.endswith(fp) or fp.endswith(key):
                    matched = key
                    break
        if not matched or matched in seen:
            continue
        seen.add(matched)
        content = repo_files[matched]
        lines = content.split("\n")
        s = max(0, c.get("start", 0) - 1)
        e = c.get("end", 0)
        if s > 0 or e > 0:
            excerpt = "\n".join(f"{i+1:5d}|{lines[i]}" for i in range(s, min(len(lines), e)))
            parts.append(f"─── {matched}:{s+1}-{e} ───\n{excerpt}\n")
        else:
            numbered = "\n".join(f"{i+1:5d}|{line}" for i, line in enumerate(lines))
            parts.append(f"─── {matched} ───\n{numbered}\n")
    return "\n".join(parts) if parts else "(no extractable content)"

def run_fastcontext(question, question_id, system_prompt):
    """Run FastContext exploration (proven working approach from debug_explore.py)."""
    result = FCResult(question_id=question_id)
    start = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<query>{question}</query>"},
    ]

    for turn in range(1, MAX_TURNS + 1):
        try:
            payload = {"model": FASTCONTEXT_MODEL, "messages": messages, "tools": TOOLS,
                      "max_tokens": 4096, "temperature": 0.3}
            resp = requests.post(FASTCONTEXT_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.elapsed = time.time() - start
            result.final_answer = f"API error: {e}"
            return result

        choice = data["choices"][0]
        msg = choice["message"]
        fr = choice["finish_reason"]
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls") or []
        usage = data.get("usage", {})

        result.fc_input_tokens += usage.get("prompt_tokens", 0)
        result.fc_output_tokens += usage.get("completion_tokens", 0)

        # Convergence: model returns content (finish_reason=stop or no tool calls)
        if not tcs or fr == "stop":
            if content:
                result.final_answer = content
                result.success = True
                result.turns = turn
                break

        if tcs:
            messages.append(msg)
            for tc in tcs:
                func = tc["function"]
                tc_args = json.loads(func.get("arguments", "{}"))
                tool_result = execute_tool(func["name"], tc_args)
                result.local_tokens += estimate_tokens(tool_result)
                result.tool_call_log.append({"turn": turn, "name": func["name"], "args": tc_args,
                                           "result_tokens": estimate_tokens(tool_result)})
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": tool_result})
            result.turns = turn
        else:
            result.success = True
            result.turns = turn
            break
    else:
        # Max turns reached — try to use last content if any
        last_content = locals().get("content", "")
        if last_content:
            result.final_answer = last_content
            result.success = True
        else:
            result.final_answer = "Did not converge"

    result.citations = extract_citations_from_text(result.final_answer)
    result.elapsed = time.time() - start
    return result

def call_solver(question, context, mode, question_id):
    result = SolveResult(question_id=question_id, mode=mode)
    start = time.time()
    payload = {
        "model": SOLVER_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert software engineer. Answer based on the provided repository context. Cite specific files and line numbers."},
            {"role": "user", "content": f"Repository context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer citing specific files."},
        ],
        "max_tokens": 2048, "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {SOLVER_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(SOLVER_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        result.answer = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        result.input_tokens = u.get("prompt_tokens", 0)
        result.output_tokens = u.get("completion_tokens", 0)
    except Exception as e:
        result.answer = f"Error: {e}"
    result.elapsed = time.time() - start
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("FastContext Real-World Benchmark v3")
    print("=" * 70)

    repo_files = load_repo_files(REPO_PATH)
    full_context = format_repo_for_context(repo_files)
    full_tokens = estimate_tokens(full_context)
    system_prompt = build_system_prompt(REPO_PATH)
    print(f"Repo: {len(repo_files)} files, ~{full_tokens:,} tokens")
    print(f"Solver: {SOLVER_MODEL} | Explorer: {FASTCONTEXT_MODEL} | Max turns: {MAX_TURNS}\n")

    all_results = []

    for q in QUESTIONS:
        print(f"\n{'─'*60}")
        print(f"Q: {q['id']} — {q['question'][:70]}...")
        print(f"{'─'*60}")

        # Path A: Full repo
        print(f"  [A] Full repo → solver ({full_tokens:,} est tokens)...")
        solver_a = call_solver(q["question"], full_context, "full_repo", q["id"])
        print(f"      {solver_a.input_tokens:,} in + {solver_a.output_tokens:,} out = {solver_a.input_tokens + solver_a.output_tokens:,} ({solver_a.elapsed:.1f}s)")

        # Path B: FastContext
        print(f"  [B] FastContext NVFP4 → explore...")
        fc = run_fastcontext(q["question"], q["id"], system_prompt)

        if fc.success:
            print(f"      ✓ {fc.turns} turns, {len(fc.tool_call_log)} tool calls, {len(fc.citations)} citations ({fc.elapsed:.1f}s)")
            print(f"      FC: {fc.fc_input_tokens:,} in + {fc.fc_output_tokens:,} out = {fc.fc_input_tokens + fc.fc_output_tokens:,}")
            print(f"      Local: {fc.local_tokens:,} tokens of tool output")

            for tc in fc.tool_call_log:
                args_s = json.dumps(tc["args"])
                if len(args_s) > 70: args_s = args_s[:67] + "..."
                print(f"        T{tc['turn']} {tc['name']}({args_s})")

            if fc.citations:
                print(f"      Citations: {[c['file'] for c in fc.citations]}")

            citation_ctx = build_citation_context(fc.citations, repo_files)
            ctx_tokens = estimate_tokens(citation_ctx)
            print(f"      Citation context: {ctx_tokens:,} tokens")

            solver_b = call_solver(q["question"], citation_ctx, "fc_citations", q["id"])
            print(f"      Solver: {solver_b.input_tokens:,} in + {solver_b.output_tokens:,} out = {solver_b.input_tokens + solver_b.output_tokens:,} ({solver_b.elapsed:.1f}s)")
        else:
            print(f"      ✗ {fc.final_answer}")
            solver_b = SolveResult(question_id=q["id"], mode="failed")

        tokens_a = solver_a.input_tokens + solver_a.output_tokens
        tokens_b = solver_b.input_tokens + solver_b.output_tokens if solver_b.input_tokens > 0 else 0
        reduction = ((tokens_a - tokens_b) / tokens_a * 100) if tokens_a > 0 and tokens_b > 0 else 0
        print(f"\n  📊 Reduction: {reduction:.1f}% ({tokens_a:,} → {tokens_b:,})")

        all_results.append({
            "question_id": q["id"], "question": q["question"],
            "ground_truth": q["ground_truth_files"],
            "without_fc": {"input": solver_a.input_tokens, "output": solver_a.output_tokens,
                          "total": tokens_a, "answer": solver_a.answer[:800], "elapsed": solver_a.elapsed},
            "with_fc": {
                "fc_success": fc.success, "fc_turns": fc.turns,
                "fc_tool_calls": len(fc.tool_call_log),
                "fc_input": fc.fc_input_tokens, "fc_output": fc.fc_output_tokens,
                "fc_local_tokens": fc.local_tokens,
                "fc_citations": fc.citations,
                "fc_tool_log": fc.tool_call_log,
                "fc_final_answer": fc.final_answer[:1000],
                "fc_elapsed": fc.elapsed,
                "solver_input": solver_b.input_tokens, "solver_output": solver_b.output_tokens,
                "solver_total": tokens_b, "solver_answer": solver_b.answer[:800] if solver_b.answer else "",
                "solver_elapsed": solver_b.elapsed,
            },
            "reduction_pct": reduction,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Question':<25} {'Path A':>12} {'Path B':>12} {'Reduce':>8} {'Turns':>6} {'Tools':>6} {'OK':>4}")
    print("─" * 75)
    ta, tb = 0, 0
    for r in all_results:
        a = r["without_fc"]["total"]
        b = r["with_fc"]["solver_total"]
        ta += a; tb += b
        ok = "✓" if r["with_fc"]["fc_success"] else "✗"
        print(f"{r['question_id']:<25} {a:>12,} {b:>12,} {r['reduction_pct']:>7.1f}% {r['with_fc']['fc_turns']:>6} {r['with_fc']['fc_tool_calls']:>6} {ok:>4}")
    print("─" * 75)
    overall = ((ta - tb) / ta * 100) if ta > 0 else 0
    print(f"{'TOTAL':<25} {ta:>12,} {tb:>12,} {overall:>7.1f}%")

    out = {"config": {"repo": REPO_PATH, "files": len(repo_files), "tokens": full_tokens,
                       "solver": SOLVER_MODEL, "explorer": FASTCONTEXT_MODEL, "max_turns": MAX_TURNS},
           "results": all_results,
           "summary": {"total_a": ta, "total_b": tb, "reduction_pct": overall}}
    with open(os.path.join(RESULTS_DIR, "realworld_v3.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/realworld_v3.json")


if __name__ == "__main__":
    main()
