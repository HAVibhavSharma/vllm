# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Harvest replayable non-determinism cases from trace-analyser reports.

Scans TRACE_ANALYSIS_DIR/trace_analysis_*.json for LLM events where the
input was identical across runs but the output diverged (the
non-determinism sources), re-fetches the exact request payload from
LangSmith, and writes one case file per event into replay/cases/.

Qualifying event (per PLAN.md):
  * kind LLM, within a structural cluster of >= 2 runs
  * input_identical (input_divergence <= noise) AND output_divergence > noise
  * non-zero prompt tokens on the source run (a real LLM call)

Source LangSmith projects may be expired: those are skipped and reported
in cases/coverage_summary.json, never fatal.

Usage:  python3 extract_cases.py [--refresh]
        (--refresh re-extracts cases that already exist)
"""
import glob
import json
import os
import sys

import replay_lib as rl


def qualifying_positions(report):
    """Yield (cluster_id, member_runs, position_record) for every
    identical-input/divergent-output LLM position in the report."""
    noise = report.get("noise_threshold", 0.02)
    clusters = report.get("cluster_analyses") or []
    if not clusters and report.get("content_analysis_cluster"):
        clusters = [{"cluster_id": 0,
                     "runs": report["content_analysis_cluster"],
                     "content_analysis": report.get("content_analysis")}]
    for c in clusters:
        ca = c.get("content_analysis") or {}
        for p in ca.get("per_position") or []:
            if p.get("input_identical") and \
                    p.get("output_divergence", 0) > noise:
                yield c["cluster_id"], c.get("runs") or [], p


def case_id(source_hash, cluster_id, pos):
    node = (pos.get("node") or "llm").replace("/", "_").replace(" ", "_")
    return "%s_c%d_p%d_%s" % (source_hash, cluster_id, pos["position"], node)


def extract_file(path, refresh, coverage):
    with open(path) as f:
        report = json.load(f)
    source_hash = os.path.basename(path)[len("trace_analysis_"):-len(".json")]
    entry = {"source": os.path.basename(path),
             "project_id": report.get("project_id"),
             "cases_written": 0, "cases_cached": 0, "skipped": []}
    coverage.append(entry)

    wanted = list(qualifying_positions(report))
    if not wanted:
        entry["skipped"].append("no identical-input/divergent-output "
                                "LLM events in report")
        return

    todo = []
    for cluster_id, members, pos in wanted:
        cid = case_id(source_hash, cluster_id, pos)
        out_path = os.path.join(rl.CASES_DIR, cid + ".json")
        if os.path.exists(out_path) and not refresh:
            entry["cases_cached"] += 1
            continue
        todo.append((cluster_id, members, pos, cid, out_path))
    if not todo:
        return

    project = report.get("project_id")
    try:
        roots = rl.list_root_runs(project)
    except rl.ReplayError as e:
        entry["skipped"].append("LangSmith project %s unreachable "
                                "(expired?): %s" % (project, e))
        return
    roots_by_name = {r["name"]: r for r in roots}

    trees = {}  # run name -> llm sequence (fetch each reference run once)
    for cluster_id, members, pos, cid, out_path in todo:
        ref = next((m for m in members if m in roots_by_name), None)
        if ref is None:
            entry["skipped"].append("%s: no cluster member run found in "
                                    "project" % cid)
            continue
        if ref not in trees:
            print("  pulling tree for run %r ..." % ref[:40],
                  file=sys.stderr)
            try:
                trees[ref] = rl.llm_sequence(
                    rl.pull_tree(roots_by_name[ref]["id"]))
            except rl.ReplayError as e:
                trees[ref] = None
                entry["skipped"].append("%s: tree fetch failed: %s"
                                        % (cid, e))
        seq = trees.get(ref)
        if not seq:
            continue
        if pos["position"] >= len(seq):
            entry["skipped"].append("%s: position %d out of range (%d llm "
                                    "calls in run)" % (cid, pos["position"],
                                                       len(seq)))
            continue
        run = seq[pos["position"]]
        ptok = rl.prompt_tokens(run)
        if ptok == 0:
            entry["skipped"].append("%s: zero prompt tokens — not a real "
                                    "LLM call" % cid)
            continue
        try:
            params = rl.request_params(run)
            rl.to_openai_messages(run.get("inputs") or {})  # validate early
        except rl.ReplayError as e:
            entry["skipped"].append("%s: payload not replayable: %s"
                                    % (cid, e))
            continue
        case = {
            "case_id": cid,
            "source_file": os.path.basename(path),
            "project_id": project,
            "cluster_id": cluster_id,
            "position": pos["position"],
            "node": pos.get("node"),
            "source_run": {"name": ref, "id": run["id"]},
            "prompt_tokens": ptok,
            "original": {"input_divergence": pos.get("input_divergence"),
                         "output_divergence": pos.get("output_divergence"),
                         "config": pos.get("config")},
            "request": {"params": params},
            "raw_inputs": run.get("inputs") or {},
            # rendered exactly like replay outputs (content + tool calls),
            # for the byte-to-byte match test
            "original_output": rl.trace_analyser().out_text(run),
        }
        os.makedirs(rl.CASES_DIR, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(case, f, indent=2)
        entry["cases_written"] += 1
        print("  wrote %s (orig out_div=%.1f%%, %s prompt tokens)"
              % (cid, (pos.get("output_divergence") or 0) * 100,
                 ptok if ptok is not None else "unknown"), file=sys.stderr)


def main():
    refresh = "--refresh" in sys.argv
    pattern = os.path.join(rl.TRACE_ANALYSIS_DIR, "trace_analysis_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No trace_analysis_*.json under %s" % rl.TRACE_ANALYSIS_DIR,
              file=sys.stderr)
        sys.exit(1)
    coverage = []
    for path in files:
        print("%s:" % os.path.basename(path), file=sys.stderr)
        extract_file(path, refresh, coverage)
    os.makedirs(rl.CASES_DIR, exist_ok=True)
    summary_path = os.path.join(rl.CASES_DIR, "coverage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(coverage, f, indent=2)
    written = sum(e["cases_written"] for e in coverage)
    cached = sum(e["cases_cached"] for e in coverage)
    skipped = sum(len(e["skipped"]) for e in coverage)
    print("\n%d case(s) written, %d already cached, %d skipped "
          "(details: %s)" % (written, cached, skipped, summary_path),
          file=sys.stderr)


if __name__ == "__main__":
    main()
