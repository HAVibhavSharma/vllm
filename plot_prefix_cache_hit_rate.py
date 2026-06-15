import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._")
    return sanitized or "unknown_langgraph_node"


def _load_job_records(file_path: str, target_job_id: str) -> list[dict]:
    records: list[dict] = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if str(data.get("job_id")) == str(target_job_id):
                records.append(data)

    return records


def _plot_hit_rates(
    hit_rates: list[float],
    prefill_times: list[float],
    title: str,
    output_path: str,
) -> None:
    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    avg_prefill_time = sum(prefill_times) / len(prefill_times) if prefill_times else 0.0

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel("Request Number")
    ax1.set_ylabel("Prefix Cache Hit Rate", color=color1)
    ax1.plot(range(1, len(hit_rates) + 1), hit_rates, marker="o", linestyle="-", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("TTFT (Prefill Time, s)", color=color2)
    ax2.plot(range(1, len(prefill_times) + 1), prefill_times, marker="s", linestyle="-", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Offset vertical limits for single-datapoint scenarios to prevent overlapping
    if len(hit_rates) == 1:
        hr = hit_rates[0]
        pt = prefill_times[0]
        ax1.set_ylim(hr - 0.2, hr + 0.2)
        ax2.set_ylim(pt - 0.3, pt + 0.1)

    plt.title(title)
    plt.text(
        0.95,
        0.95,
        f"Avg Hit Rate: {avg_hit_rate:.4f}\nAvg TTFT: {avg_prefill_time:.4f}s",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_hit_rate(file_path: str, target_job_id: str, output_dir: str) -> None:
    records = _load_job_records(file_path, target_job_id)
    if not records:
        print(f"No matching records found for job_id={target_job_id}.")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_hit_rates = [float(record.get("prefix_cache_hit_rate", 0.0)) for record in records]
    all_prefill_times = [float(record.get("prefill_time", 0.0)) for record in records]
    overall_output_path = os.path.join(
        output_dir,
        f"prefix_cache_hit_rate_job_{_sanitize_filename(str(target_job_id))}.png",
    )
    _plot_hit_rates(
        all_hit_rates,
        all_prefill_times,
        f"Prefix Cache Hit Rate and TTFT for job_id: {target_job_id}",
        overall_output_path,
    )

    node_to_hit_rates: dict[str, list[float]] = defaultdict(list)
    node_to_prefill_times: dict[str, list[float]] = defaultdict(list)
    for record in records:
        langgraph_node = record.get("langgraph_node")
        node_key = (
            str(langgraph_node)
            if langgraph_node not in (None, "")
            else "unknown_langgraph_node"
        )
        node_to_hit_rates[node_key].append(
            float(record.get("prefix_cache_hit_rate", 0.0))
        )
        node_to_prefill_times[node_key].append(
            float(record.get("prefill_time", 0.0))
        )

    for langgraph_node in sorted(node_to_hit_rates.keys()):
        hit_rates = node_to_hit_rates[langgraph_node]
        prefill_times = node_to_prefill_times[langgraph_node]
        output_path = os.path.join(
            output_dir,
            f"{_sanitize_filename(langgraph_node)}.png",
        )
        _plot_hit_rates(
            hit_rates,
            prefill_times,
            (
                "Hit Rate & TTFT for "
                f"job_id: {target_job_id}, node: {langgraph_node}"
            ),
            output_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot prefix cache hit rate from vLLM request stats JSONL."
    )
    parser.add_argument("file_path", help="Path to the JSONL file")
    parser.add_argument("job_id", help="Filter by specific job_id")
    parser.add_argument("output_dir", help="Output directory for the plots")

    args = parser.parse_args()
    plot_hit_rate(args.file_path, args.job_id, args.output_dir)
