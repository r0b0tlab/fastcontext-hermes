#!/usr/bin/env python3
"""
Async FastContext Explorer — Hermes Integration

Dispatches FastContext as a background subagent via delegate_task(background=True).
Multiple exploration queries run in parallel against the local GB10 FastContext endpoint.
Results (file:line citations) are collected and deduplicated before returning to the
main agent.

Usage from Hermes controller:
    from async_explorer import AsyncExplorer

    explorer = AsyncExplorer(work_dir="/path/to/repo")
    citations = await explorer.explore([
        "Find authentication middleware and token validation",
        "Find database models and migration scripts",
        "Find API route handlers and middleware chain",
    ])
    # citations = {
    #     "auth": "src/auth/middleware.py:42-78\nsrc/auth/tokens.py:15-30",
    #     "database": "src/models/user.py:1-45\nsrc/migrations/001.py:1-20",
    #     "api": "src/routes/api.py:10-50\nsrc/middleware/chain.py:5-25",
    # }
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExplorationResult:
    """Result of a single FastContext exploration."""
    query_label: str
    query: str
    citations: list[str] = field(default_factory=list)
    raw_output: str = ""
    trajectory_path: Optional[str] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.citations) > 0

    def to_dict(self) -> dict:
        return {
            "query_label": self.query_label,
            "query": self.query,
            "citations": self.citations,
            "latency_seconds": self.latency_seconds,
            "success": self.success,
            "error": self.error,
        }


def parse_citations(raw_output: str) -> list[str]:
    """Extract file:line citations from FastContext <final_answer> block."""
    # Extract <final_answer> block
    match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", raw_output, re.DOTALL)
    if not match:
        # Fallback: try to find file:line patterns directly
        lines = raw_output.strip().split("\n")
        return [l.strip() for l in lines if re.match(r"^[\w/.-]+:\d+", l.strip())]

    content = match.group(1)
    citations = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if line and (":" in line or "/" in line):
            citations.append(line)
    return citations


async def run_fastcontext(
    query: str,
    work_dir: str,
    label: str = "query",
    max_turns: int = 6,
    citation_only: bool = True,
    timeout: int = 60,
) -> ExplorationResult:
    """
    Run a single FastContext exploration against the local GB10 endpoint.

    This is designed to be called via delegate_task(background=True) from the
    Hermes controller. Each call is independent and can run in parallel.
    """
    import time

    start = time.time()
    traj_dir = Path("/tmp/fastcontext-trajs")
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = str(traj_dir / f"{label}.jsonl")

    cmd = [
        "fastcontext",
        "--query", query,
        "--max-turns", str(max_turns),
        "--traj", traj_path,
    ]
    if citation_only:
        cmd.append("--citation")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            # FastContext CLI reads BASE_URL, MODEL, API_KEY directly.
            # The benchmark configs use FASTCONTEXT_* as separate credentials,
            # but the CLI itself uses the bare names.
            env={
                **os.environ,
                "BASE_URL": os.environ.get(
                    "FASTCONTEXT_BASE_URL",
                    os.environ.get("BASE_URL", "http://127.0.0.1:30000/v1"),
                ),
                "MODEL": os.environ.get(
                    "FASTCONTEXT_MODEL",
                    os.environ.get("MODEL", "FastContext-1.0-4B-RL-NVFP4"),
                ),
                "API_KEY": os.environ.get(
                    "FASTCONTEXT_API_KEY",
                    os.environ.get("API_KEY", "local"),
                ),
            },
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        elapsed = time.time() - start

        output = stdout.decode("utf-8", errors="replace")
        citations = parse_citations(output)

        if not citations and proc.returncode != 0:
            return ExplorationResult(
                query_label=label,
                query=query,
                raw_output=output + stderr.decode("utf-8", errors="replace"),
                trajectory_path=traj_path,
                latency_seconds=elapsed,
                error=f"FastContext exited with code {proc.returncode}",
            )

        return ExplorationResult(
            query_label=label,
            query=query,
            citations=citations,
            raw_output=output,
            trajectory_path=traj_path,
            latency_seconds=elapsed,
        )

    except asyncio.TimeoutError:
        return ExplorationResult(
            query_label=label,
            query=query,
            latency_seconds=time.time() - start,
            error=f"Timed out after {timeout}s",
        )
    except FileNotFoundError:
        return ExplorationResult(
            query_label=label,
            query=query,
            latency_seconds=time.time() - start,
            error="fastcontext CLI not found. Install: uv tool install git+https://github.com/microsoft/fastcontext.git",
        )
    except Exception as e:
        return ExplorationResult(
            query_label=label,
            query=query,
            latency_seconds=time.time() - start,
            error=str(e),
        )


class AsyncExplorer:
    """
    Orchestrates parallel FastContext explorations via async subagents.

    Designed for Hermes delegate_task(background=True) integration:
    1. Dispatch N exploration queries in parallel
    2. Each query runs FastContext CLI against local GB10 endpoint
    3. GB10 continuous batching handles concurrent requests efficiently
    4. Collect and deduplicate citations
    """

    def __init__(self, work_dir: str, default_max_turns: int = 6):
        self.work_dir = work_dir
        self.default_max_turns = default_max_turns

    async def explore(
        self,
        queries: dict[str, str] | list[str],
        max_turns: int | None = None,
        timeout: int = 60,
    ) -> dict[str, ExplorationResult]:
        """
        Run multiple FastContext explorations in parallel.

        Args:
            queries: Either a dict {label: query} or list of query strings.
            max_turns: Max exploration turns per query.
            timeout: Per-query timeout in seconds.

        Returns:
            Dict mapping label → ExplorationResult with citations.
        """
        turns = max_turns or self.default_max_turns

        if isinstance(queries, list):
            queries = {f"query_{i}": q for i, q in enumerate(queries)}

        tasks = [
            run_fastcontext(
                query=q,
                work_dir=self.work_dir,
                label=label,
                max_turns=turns,
                timeout=timeout,
            )
            for label, q in queries.items()
        ]

        results = await asyncio.gather(*tasks)

        return dict(zip(queries.keys(), results))

    def format_for_agent(self, results: dict[str, ExplorationResult]) -> str:
        """
        Format exploration results as compact context for the main agent.
        This is what gets injected into the solver's context.
        """
        lines = ["## Repository Context (via FastContext)\n"]
        all_citations = set()

        for label, result in results.items():
            if result.success:
                lines.append(f"### {label}")
                for cite in result.citations:
                    all_citations.add(cite)
                    lines.append(f"- `{cite}`")
                lines.append(f"_(explored in {result.latency_seconds:.1f}s)_\n")
            else:
                lines.append(f"### {label}")
                lines.append(f"_Exploration failed: {result.error}_\n")

        lines.append(f"\n**Total unique files/lines cited: {len(all_citations)}**")
        return "\n".join(lines)


# ─── CLI for standalone testing ─────────────────────────────────────────

async def main():
    """Standalone test: explore current directory."""
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    queries = {
        "entry_points": "Find the main entry points and initialization logic",
        "config": "Find configuration files and environment variable usage",
        "tests": "Find test files and test fixtures",
    }

    print(f"Exploring: {work_dir}")
    print(f"Queries: {list(queries.keys())}\n")

    explorer = AsyncExplorer(work_dir=work_dir)
    results = await explorer.explore(queries)

    print(explorer.format_for_agent(results))

    # Save results
    output_path = Path(work_dir) / ".fastcontext" / "exploration_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {k: v.to_dict() for k, v in results.items()},
        indent=2,
    ))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
