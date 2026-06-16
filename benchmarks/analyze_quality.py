#!/usr/bin/env python3
"""Analyze retrieval quality: which ground truth files did FastContext actually interact with?"""
import json, os

REPO = "/home/r0b0tdgx/projects/hermes-concurrent-agents"

with open("/home/r0b0tdgx/projects/fastcontext-hermes/benchmarks/results/realworld_v3.json") as f:
    data = json.load(f)

print("=" * 70)
print("RETRIEVAL QUALITY ANALYSIS")
print("=" * 70)

total_truth = 0
total_found = 0
total_read = 0

for r in data["results"]:
    qid = r["question_id"]
    truth = set(r["ground_truth"])
    fc = r["with_fc"]

    # Files the model actually READ (from tool call log)
    files_read = set()
    files_grepped = set()
    for tc in fc["fc_tool_log"]:
        if tc["name"] == "Read":
            path = tc["args"].get("path", "")
            if path.startswith(REPO):
                relpath = os.path.relpath(path, REPO)
                files_read.add(relpath)
        elif tc["name"] == "Grep":
            # Check if grep targeted specific ground truth files
            path = tc["args"].get("path", "")
            if path.startswith(REPO):
                relpath = os.path.relpath(path, REPO)
                files_grepped.add(relpath)

    # Also check cited files
    cited = set(c["file"] for c in fc["fc_citations"])
    cited_clean = set()
    for c in cited:
        # Clean up paths
        if c.startswith("//"): continue
        cited_clean.add(c)

    # Interactions with ground truth
    read_truth = truth & files_read
    grep_truth = truth & files_grepped
    cited_truth = truth & cited_clean
    all_interacted = truth & (files_read | files_grepped | cited_clean)

    total_truth += len(truth)
    total_found += len(all_interacted)
    total_read += len(files_read)

    print(f"\n{'─'*60}")
    print(f"Q: {qid}")
    print(f"  Ground truth: {sorted(truth)}")
    print(f"  Files READ:   {sorted(files_read)[:10]}")
    print(f"  READ matches: {sorted(read_truth) if read_truth else 'NONE'}")
    print(f"  GREP matches: {sorted(grep_truth) if grep_truth else 'NONE'}")
    print(f"  Cited matches: {sorted(cited_truth) if cited_truth else 'NONE'}")
    print(f"  TOTAL interacted with truth: {len(all_interacted)}/{len(truth)} ({len(all_interacted)/len(truth)*100:.0f}%)")
    print(f"  Token reduction: {r['reduction_pct']:.1f}%")

recall = total_found / total_truth * 100 if total_truth > 0 else 0
print(f"\n{'='*70}")
print(f"OVERALL RETRIEVAL RECALL: {total_found}/{total_truth} = {recall:.0f}%")
print(f"TOTAL FILES READ BY FC: {total_read}")
print(f"OVERALL TOKEN REDUCTION: {data['summary']['reduction_pct']:.1f}%")
