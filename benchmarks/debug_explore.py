#!/usr/bin/env python3
"""Debug FastContext exploration loop - trace single query."""
import json, requests, os, subprocess

FASTCONTEXT_URL = "http://localhost:30000/v1/chat/completions"
MODEL = "FastContext-1.0-4B-RL-NVFP4"
REPO = "/home/r0b0tdgx/projects/hermes-concurrent-agents"
MAX_TURNS = 6

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
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches"]}
        }, "required": ["pattern"]}
    }},
]

sys_prompt = f"""You are a codebase exploration specialist. Find relevant files and return citations.

When done, output:
<final_answer>
/path/to/file.py:10-20
</final_answer>

Working directory: {REPO}
Files in root: {', '.join(sorted(os.listdir(REPO)))}
"""

messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "<query>What is the fault injection testing mechanism and how does it work?</query>"},
]

def execute_tool(name, args):
    if name == "Read":
        path = args.get("path", "")
        if not os.path.exists(path):
            return f"File not found: {path}"
        with open(path, errors="replace") as f:
            lines = f.readlines()
        numbered = "".join(f"{i+1}|{l}" for i, l in enumerate(lines[:2000]))
        return f"```\n{numbered}\n```"
    elif name == "Glob":
        directory = args.get("directory", REPO)
        pattern = args.get("pattern", "*")
        result = subprocess.run(["rg", "--files", directory, "--glob", pattern],
                              capture_output=True, text=True, timeout=10)
        return result.stdout[:5000] if result.stdout else "No files"
    elif name == "Grep":
        pattern = args.get("pattern", "")
        path = args.get("path", REPO)
        result = subprocess.run(["rg", "--files-with-matches", pattern, path],
                              capture_output=True, text=True, timeout=10)
        return result.stdout[:5000] if result.stdout else "No matches"

for turn in range(1, MAX_TURNS + 1):
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 4096, "temperature": 0.3}
    resp = requests.post(FASTCONTEXT_URL, json=payload, timeout=120)
    data = resp.json()

    choice = data["choices"][0]
    msg = choice["message"]
    fr = choice["finish_reason"]
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    usage = data.get("usage", {})

    print(f"\n{'='*50}")
    print(f"TURN {turn} | finish_reason={fr} | content_len={len(content)} | tool_calls={len(tcs)}")
    print(f"usage: in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")

    if content:
        print(f"CONTENT: {repr(content[:300])}")

    for tc in tcs:
        args = tc["function"]["arguments"][:100]
        print(f"  TOOL: {tc['function']['name']}({args})")

    # Check if model returned final answer
    if "<final_answer>" in content.lower() or (not tcs and content):
        print(f"\n>>> FINAL ANSWER DETECTED at turn {turn}")
        print(content[:500])
        break

    if not tcs:
        print(f"\n>>> STOPPED without tool calls or final_answer")
        print(f"content: {repr(content[:300])}")
        break

    # Execute tools and continue
    messages.append(msg)
    for tc in tcs:
        func = tc["function"]
        tc_args = json.loads(func.get("arguments", "{}"))
        result = execute_tool(func["name"], tc_args)
        messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})

    print(f"  → executed {len(tcs)} tools, continuing...")
else:
    print(f"\n>>> DID NOT CONVERGE after {MAX_TURNS} turns")
