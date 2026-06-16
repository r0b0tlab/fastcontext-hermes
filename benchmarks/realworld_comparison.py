#!/usr/bin/env python3
"""
FastContext Real-World Benchmark v2
====================================
Tests the model card's core claims using the EXACT tool schemas and system prompt
from microsoft/fastcontext source code.

Path A: Stuff full repo → API solver
Path B: FastContext NVFP4 explores → citations → targeted context → API solver
"""

import json, os, re, time, glob as globmod, subprocess, requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import platform

# ─── Configuration ───
REPO_PATH = "/home/r0b0tdgx/projects/hermes-concurrent-agents"
FASTCONTEXT_URL = "http://localhost:30000/v1/chat/completions"
SOLVER_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4") + "/chat/completions"
SOLVER_KEY = os.environ.get("GLM_API_KEY", "")
SOLVER_MODEL = "glm-5.2"
FASTCONTEXT_MODEL = "FastContext-1.0-4B-RL-NVFP4"
MAX_TURNS = 6  # matches fastcontext CLI default
RESULTS_DIR = "/home/r0b0tdgx/projects/fastcontext-hermes/benchmarks/results"

# ─── Exact Tool Schemas from microsoft/fastcontext source ───
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the contents of a file. Supports reading specific line ranges via offset and limit. Returns file contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute path of the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from. Positive values are 1-indexed from the start of the file. Negative values count backwards from the end (e.g. -1 is the last line). Only provide if the file is too large to read at once.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                    },
                },
                "required": ["path"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern. Use this to discover files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The absolute path of the directory to search in. If not provided, the current working directory will be used.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files or directories.",
                    },
                },
                "required": ["pattern"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents using ripgrep. Supports regex patterns, file type filters, and context lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regular expression pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search. Defaults to current working directory.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob filter (e.g., '*.py', '*.{ts,tsx}').",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode. Default: files_with_matches.",
                    },
                    "-C": {
                        "type": "number",
                        "description": "Lines of context before and after match (content mode). Default: 3.",
                    },
                    "-n": {
                        "type": "boolean",
                        "description": "Show line numbers (content mode). Default: true.",
                    },
                    "-i": {
                        "type": "boolean",
                        "description": "Case-insensitive search. Default: false.",
                    },
                    "head_limit": {
                        "type": "number",
                        "description": "Limit output to first N entries.",
                    },
                },
                "required": ["pattern"],
            }
        }
    },
]


def build_system_prompt(work_dir: str) -> str:
    """Build system prompt matching microsoft/fastcontext system.md template."""
    work_dir_ls = "\n".join(sorted(os.listdir(work_dir)))
    os_kind = platform.system()
    shell_name = os.getenv("SHELL", "bash")

    return f"""You are a codebase exploration specialist focused exclusively on searching and analyzing existing code.
Your main goal is to explore the codebase based on a query, which are denoted by the <query> tag.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use Read when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

## Required Output

End your response with an optional brief explanation of your findings (no more than 50 words), followed by a `<final_answer>` tag containing the relevant file paths and line ranges.

<final_answer>
/absolute/path/to/file_1.py:10-15 (Optional Brief Reason: e.g., "Core logic to modify")
/absolute/path/to/file_2.js:102-123
</final_answer>

## Working Environment

OS Version: {os_kind}

Shell: {shell_name}

Workspace Path: {work_dir}

The directory listing of the workspace is:
```
{work_dir_ls}
```

Now, complete the user's search request efficiently and report your findings clearly."""


# ─── 5 Real Exploration Questions ───
QUESTIONS = [
    {
        "id": "q1_deploy_vllm",
        "question": "How do I deploy concurrent agents with a vLLM backend? What configuration is needed?",
        "ground_truth_files": [
            "config/vllm/docker-compose.yml",
            "docs/deployment-guide.md",
            "setup.sh",
            "scripts/spawn.sh",
        ],
    },
    {
        "id": "q2_fault_injection",
        "question": "What is the fault injection testing mechanism and how does it work?",
        "ground_truth_files": [
            "scripts/fault-injection-test.sh",
            "docs/durability-tests.md",
        ],
    },
    {
        "id": "q3_profile_architecture",
        "question": "How does the orchestrator profile differ from worker profiles (coder, creative, qa, research)?",
        "ground_truth_files": [
            "profiles/orchestrator/SOUL.md",
            "profiles/coder-worker/SOUL.md",
            "profiles/qa-worker/SOUL.md",
        ],
    },
    {
        "id": "q4_benchmarking",
        "question": "What benchmarking methods are available and what metrics do they capture?",
        "ground_truth_files": [
            "scripts/benchmark.sh",
            "docs/benchmarking.md",
        ],
    },
    {
        "id": "q5_ci_pipeline",
        "question": "How is the CI/CD pipeline configured? What checks run on push?",
        "ground_truth_files": [
            ".github/workflows/ci.yml",
        ],
    },
]


# ─── Data Structures ───
@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result_preview: str = ""
    result_tokens: int = 0

@dataclass
class FastContextResult:
    question_id: str
    success: bool = False
    turns: int = 0
    tool_calls: list = field(default_factory=list)
    final_answer: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_local_tokens: int = 0
    citations: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str = ""

@dataclass
class SolverResult:
    question_id: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    context_type: str = ""

@dataclass
class ComparisonResult:
    question_id: str
    question: str
    solver_full: Optional[SolverResult] = None
    fc_exploration: Optional[FastContextResult] = None
    solver_fc: Optional[SolverResult] = None
    token_reduction_pct: float = 0.0
    ground_truth_files: list = field(default_factory=list)


# ─── Tool Execution (matching microsoft/fastcontext exactly) ───
MAX_LINE = 2000
MAX_LINE_LENGTH = 2000

def execute_read(path: str, offset: int = None, limit: int = None) -> str:
    """Execute Read tool matching microsoft/fastcontext read.py."""
    if not os.path.exists(path):
        return f"Read Tool: file {path} does not exist."

    with open(path, "r", errors="replace") as f:
        raw_lines = f.readlines()

    if not raw_lines:
        return "File is empty."

    end_line = -1
    if offset is None or offset < 0:
        offset = 1
    if limit is not None:
        end_line = offset + limit - 1
    if end_line == -1 or end_line > len(raw_lines):
        end_line = len(raw_lines)

    lines = []
    total_read_lines = end_line - offset + 1
    if total_read_lines > MAX_LINE:
        end_line = offset + MAX_LINE - 1
    for i in range(offset - 1, end_line):
        raw_line = raw_lines[i]
        if len(raw_line) > MAX_LINE_LENGTH:
            line = raw_line[:MAX_LINE_LENGTH] + "...\n"
        else:
            line = raw_line
        prefixed = f"{i+1}|{line}"
        lines.append(prefixed)
    if total_read_lines > MAX_LINE:
        lines.append("...")
    content = "".join(lines)
    return f"```{path}:{offset}-{end_line}\n{content}\n```"

def execute_glob(directory: str, pattern: str, cwd: str = REPO_PATH) -> str:
    """Execute Glob tool matching microsoft/fastcontext glob.py."""
    if not os.path.isdir(directory):
        return f"The directory `{directory}` does not exist or is not a directory."

    try:
        cmd = ["rg", "--files", directory, "--glob", pattern]
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
        output = result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Glob error: {e}"

    if not output:
        return "No files found"

    matched = output.splitlines()
    limit_n = 100
    if len(matched) > limit_n:
        matched = matched[:limit_n]
        matched.append(f"Results are truncated: showing first {limit_n} results.")

    return "\n".join(matched)

def execute_grep(pattern: str, path: str = None, glob: str = None,
                 output_mode: str = "files_with_matches", context: int = 3,
                 ignore_case: bool = False, head_limit: int = None,
                 cwd: str = REPO_PATH) -> str:
    """Execute Grep tool matching microsoft/fastcontext grep.py."""
    search_path = path or cwd

    cmd = ["rg", "--heading", "--color", "never"]

    if glob:
        cmd.extend(["--glob", glob])
    if ignore_case:
        cmd.append("--ignore-case")

    if output_mode == "content":
        cmd.extend(["-C", str(context), "-n"])
    elif output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count-matches")

    cmd.append(pattern)
    cmd.append(search_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Grep error: {e}"

    if not output:
        return "No matches found"

    lines = output.splitlines()
    limit_n = head_limit if head_limit and 0 < head_limit < 100 else 100
    if len(lines) > limit_n:
        lines = lines[:limit_n]
        lines.append(f"Results truncated to first {limit_n} lines")

    return "\n".join(lines)

def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch tool call to correct executor."""
    if name == "Read":
        return execute_read(
            arguments.get("path", ""),
            arguments.get("offset"),
            arguments.get("limit")
        )
    elif name == "Glob":
        directory = arguments.get("directory", REPO_PATH)
        return execute_glob(directory, arguments.get("pattern", "*"))
    elif name == "Grep":
        return execute_grep(
            arguments.get("pattern", ""),
            arguments.get("path"),
            arguments.get("glob"),
            arguments.get("output_mode", "files_with_matches"),
            arguments.get("-C", 3),
            arguments.get("-i", False),
            arguments.get("head_limit"),
        )
    return f"Unknown tool: {name}"

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def extract_final_answer(text: str) -> str:
    """Extract <final_answer> block matching microsoft/fastcontext utils.py."""
    m = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    if m is None:
        return text
    return m.group(0)

def extract_citations(text: str, repo_path: str = REPO_PATH) -> list:
    """Extract file:line citations from final_answer text."""
    citations = []
    # Match absolute paths with line ranges
    for match in re.finditer(r'(/[a-zA-Z0-9_./-]+):(\d+)[-–]?(\d+)?', text):
        filepath = match.group(1)
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line
        # Convert to relative
        if filepath.startswith(repo_path):
            relpath = os.path.relpath(filepath, repo_path)
        else:
            relpath = filepath
        citations.append({"file": relpath, "start_line": start_line, "end_line": end_line})

    # Also match relative paths
    if not citations:
        for match in re.finditer(r'([a-zA-Z0-9_./-]+/[a-zA-Z0-9_./-]+):(\d+)[-–]?(\d+)?', text):
            filepath = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3)) if match.group(3) else start_line
            citations.append({"file": filepath, "start_line": start_line, "end_line": end_line})

    return citations


# ─── Repo Utilities ───
def load_repo_files(repo_path: str) -> dict:
    files = {}
    extensions = {'.py', '.md', '.yaml', '.yml', '.sh', '.json', '.toml', '.txt'}
    for root, dirs, filenames in os.walk(repo_path):
        if '.git' in root:
            continue
        for fname in filenames:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1]
            if ext in extensions or fname in ('Makefile', 'Dockerfile'):
                relpath = os.path.relpath(fpath, repo_path)
                try:
                    with open(fpath, 'r', errors='replace') as f:
                        files[relpath] = f.read()
                except:
                    pass
    return files

def format_repo_for_context(repo_files: dict) -> str:
    parts = []
    for fpath in sorted(repo_files.keys()):
        parts.append(f"─── {fpath} ───\n{repo_files[fpath]}\n")
    return "\n".join(parts)

def build_citation_context(citations: list, repo_files: dict, repo_path: str) -> str:
    """Build focused context from FastContext citations."""
    if not citations:
        # Fall back to final answer text itself
        return "(no specific citations provided)"

    parts = []
    seen_files = set()
    for cite in citations:
        filepath = cite["file"]
        if filepath in seen_files:
            continue
        seen_files.add(filepath)

        if filepath in repo_files:
            content = repo_files[filepath]
            start = cite.get("start_line", 0)
            end = cite.get("end_line", 0)

            if start and end and start > 0:
                lines = content.split("\n")
                start_idx = max(0, start - 1)
                end_idx = min(len(lines), end)
                excerpt = "\n".join(f"{i+1:5d}|{lines[i]}" for i in range(start_idx, end_idx))
                parts.append(f"─── {filepath}:{start}-{end} ───\n{excerpt}\n")
            else:
                numbered = "\n".join(f"{i+1:5d}|{line}" for i, line in enumerate(content.split("\n")))
                parts.append(f"─── {filepath} ───\n{numbered}\n")
        else:
            parts.append(f"(file not found: {filepath})\n")

    return "\n".join(parts) if parts else "(no extractable citations)"


# ─── FastContext Exploration Loop ───
def run_fastcontext_exploration(question: str, question_id: str, system_prompt: str) -> FastContextResult:
    """Run FastContext exploration loop matching microsoft/fastcontext agent.py."""
    result = FastContextResult(question_id=question_id)
    start_time = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<query>{question}</query>"},
    ]

    for turn in range(1, MAX_TURNS + 1):
        try:
            payload = {
                "model": FASTCONTEXT_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "max_tokens": 4096,
                "temperature": 0.3,
            }
            resp = requests.post(FASTCONTEXT_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            result.error = f"Turn {turn} API error: {e}"
            result.elapsed_seconds = time.time() - start_time
            return result

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")
        usage = data.get("usage", {})

        result.total_input_tokens += usage.get("prompt_tokens", 0)
        result.total_output_tokens += usage.get("completion_tokens", 0)

        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""

        # If the model returns content (not tool calls), check for final_answer
        if not tool_calls or finish_reason == "stop":
            if content:
                final = extract_final_answer(content)
                if "<final_answer>" in final or finish_reason == "stop":
                    result.final_answer = content
                    result.success = True
                    result.turns = turn
                    break

        if tool_calls:
            # Add assistant message with tool calls
            messages.append(message)

            for tc in tool_calls:
                func = tc.get("function", {})
                tc_name = func.get("name", "")
                try:
                    tc_args = json.loads(func.get("arguments", "{}"))
                except:
                    tc_args = {}

                tool_result = execute_tool(tc_name, tc_args)
                tool_tokens = estimate_tokens(tool_result)

                result.tool_calls.append(ToolCallRecord(
                    name=tc_name, arguments=tc_args,
                    result_preview=tool_result[:300],
                    result_tokens=tool_tokens
                ))
                result.total_local_tokens += tool_tokens

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result,
                })

            result.turns = turn
        else:
            # No tool calls and no final answer — model stopped
            result.final_answer = content
            result.success = True
            result.turns = turn
            break
    else:
        # Force-extract final answer from last content
        result.final_answer = "Did not converge in max turns"
        result.error = f"Exceeded {MAX_TURNS} turns"

    # Extract citations
    if result.final_answer and "<final_answer>" in result.final_answer:
        result.citations = extract_citations(result.final_answer)

    result.elapsed_seconds = time.time() - start_time
    return result


# ─── Solver Model ───
def call_solver(question: str, context: str, context_type: str, question_id: str) -> SolverResult:
    result = SolverResult(question_id=question_id, answer="", context_type=context_type)
    start_time = time.time()

    system_prompt = "You are an expert software engineer. Based on the provided repository context, answer the user's question accurately and concisely. Cite specific files and line numbers when relevant."

    user_prompt = f"""Repository context:
{context}

Question: {question}

Please provide a detailed answer based on the repository context above. Cite specific files."""

    payload = {
        "model": SOLVER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    headers = {"Authorization": f"Bearer {SOLVER_KEY}", "Content-Type": "application/json"}

    try:
        resp = requests.post(SOLVER_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        result.answer = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        result.input_tokens = usage.get("prompt_tokens", 0)
        result.output_tokens = usage.get("completion_tokens", 0)
    except Exception as e:
        result.answer = f"Error: {e}"

    result.elapsed_seconds = time.time() - start_time
    return result


# ─── Main ───
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("FastContext Real-World Benchmark v2")
    print("=" * 70)
    print(f"Repo: {REPO_PATH}")
    print(f"Solver: {SOLVER_MODEL} via z.ai API")
    print(f"Explorer: {FASTCONTEXT_MODEL} (NVFP4) via localhost:30000")
    print(f"Max turns: {MAX_TURNS}")
    print()

    repo_files = load_repo_files(REPO_PATH)
    full_repo_context = format_repo_for_context(repo_files)
    full_repo_tokens = estimate_tokens(full_repo_context)
    print(f"Repo: {len(repo_files)} files, ~{full_repo_tokens:,} estimated tokens")

    system_prompt = build_system_prompt(REPO_PATH)
    print(f"System prompt: {estimate_tokens(system_prompt):,} tokens")
    print()

    all_results = []

    for q in QUESTIONS:
        print(f"\n{'─' * 60}")
        print(f"Q: {q['id']} — {q['question'][:70]}...")
        print(f"{'─' * 60}")

        comparison = ComparisonResult(
            question_id=q["id"], question=q["question"],
            ground_truth_files=q["ground_truth_files"],
        )

        # ═══ Path A: Full Repo ═══
        print(f"\n  [A] WITHOUT FastContext — full repo ({full_repo_tokens:,} est tokens)...")
        solver_full = call_solver(q["question"], full_repo_context, "full_repo", q["id"])
        comparison.solver_full = solver_full
        print(f"      → {solver_full.input_tokens:,} in + {solver_full.output_tokens:,} out = {solver_full.input_tokens + solver_full.output_tokens:,} total ({solver_full.elapsed_seconds:.1f}s)")

        # ═══ Path B: FastContext ═══
        print(f"\n  [B] WITH FastContext NVFP4 — exploring...")
        fc_result = run_fastcontext_exploration(q["question"], q["id"], system_prompt)
        comparison.fc_exploration = fc_result

        if fc_result.success:
            print(f"      ✓ Converged in {fc_result.turns} turns, {len(fc_result.tool_calls)} tool calls")
            print(f"      FC: {fc_result.total_input_tokens:,} in + {fc_result.total_output_tokens:,} out")
            print(f"      Local: {fc_result.total_local_tokens:,} tokens of tool output")
            print(f"      Citations: {len(fc_result.citations)} files")
            print(f"      Time: {fc_result.elapsed_seconds:.1f}s")

            for tc in fc_result.tool_calls:
                args_str = json.dumps(tc.arguments)
                if len(args_str) > 80:
                    args_str = args_str[:77] + "..."
                print(f"        {tc.name}({args_str})")

            citation_context = build_citation_context(fc_result.citations, repo_files, REPO_PATH)
            citation_tokens = estimate_tokens(citation_context)
            print(f"      Citation context: {citation_tokens:,} tokens")

            solver_fc = call_solver(q["question"], citation_context, "fastcontext_citations", q["id"])
            comparison.solver_fc = solver_fc
            print(f"      Solver: {solver_fc.input_tokens:,} in + {solver_fc.output_tokens:,} out = {solver_fc.input_tokens + solver_fc.output_tokens:,} ({solver_fc.elapsed_seconds:.1f}s)")
        else:
            print(f"      ✗ {fc_result.error}")
            comparison.solver_fc = SolverResult(question_id=q["id"], answer="FC failed", context_type="failed")

        # Token reduction
        tokens_a = comparison.solver_full.input_tokens + comparison.solver_full.output_tokens
        if comparison.solver_fc and comparison.solver_fc.input_tokens > 0:
            tokens_b = comparison.solver_fc.input_tokens + comparison.solver_fc.output_tokens
            comparison.token_reduction_pct = ((tokens_a - tokens_b) / tokens_a * 100) if tokens_a > 0 else 0
            print(f"\n  📊 Solver token reduction: {comparison.token_reduction_pct:.1f}% ({tokens_a:,} → {tokens_b:,})")

        all_results.append(comparison)

    # ═══ Summary ═══
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Question':<25} {'Path A':>12} {'Path B':>12} {'Reduction':>10} {'Turns':>6} {'Tools':>6}")
    print("─" * 75)

    total_a, total_b = 0, 0
    for c in all_results:
        a = (c.solver_full.input_tokens + c.solver_full.output_tokens) if c.solver_full else 0
        b = (c.solver_fc.input_tokens + c.solver_fc.output_tokens) if c.solver_fc and c.solver_fc.input_tokens > 0 else 0
        turns = c.fc_exploration.turns if c.fc_exploration else 0
        ntools = len(c.fc_exploration.tool_calls) if c.fc_exploration else 0
        total_a += a
        total_b += b
        status = "✓" if c.fc_exploration and c.fc_exploration.success else "✗"
        print(f"{c.question_id:<25} {a:>12,} {b:>12,} {c.token_reduction_pct:>9.1f}% {turns:>6} {ntools:>6} {status}")

    print("─" * 75)
    overall = ((total_a - total_b) / total_a * 100) if total_a > 0 else 0
    print(f"{'TOTAL':<25} {total_a:>12,} {total_b:>12,} {overall:>9.1f}%")

    # Save
    output = {
        "config": {"repo": REPO_PATH, "repo_files": len(repo_files), "repo_tokens": full_repo_tokens,
                    "solver": SOLVER_MODEL, "explorer": FASTCONTEXT_MODEL, "max_turns": MAX_TURNS},
        "results": [],
        "summary": {"total_without": total_a, "total_with": total_b, "reduction_pct": overall},
    }

    for c in all_results:
        output["results"].append({
            "question_id": c.question_id,
            "question": c.question,
            "without_fc": {
                "tokens": (c.solver_full.input_tokens + c.solver_full.output_tokens) if c.solver_full else 0,
                "answer": (c.solver_full.answer[:500] if c.solver_full else ""),
            },
            "with_fc": {
                "success": c.fc_exploration.success if c.fc_exploration else False,
                "turns": c.fc_exploration.turns if c.fc_exploration else 0,
                "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in c.fc_exploration.tool_calls] if c.fc_exploration else [],
                "citations": c.fc_exploration.citations if c.fc_exploration else [],
                "fc_tokens": (c.fc_exploration.total_input_tokens + c.fc_exploration.total_output_tokens) if c.fc_exploration else 0,
                "solver_tokens": (c.solver_fc.input_tokens + c.solver_fc.output_tokens) if c.solver_fc else 0,
                "answer": (c.solver_fc.answer[:500] if c.solver_fc else ""),
            },
            "ground_truth": c.ground_truth_files,
            "reduction_pct": c.token_reduction_pct,
        })

    out_path = os.path.join(RESULTS_DIR, "realworld_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
