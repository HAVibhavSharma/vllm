import argparse
import json
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot prefix cache hit rate from vLLM request stats JSONL.")
    parser.add_argument("input_file", help="Path to the JSONL file containing request stats.")
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Cannot find input file: {input_file}")

    hit_rates = []
    
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            hit_rate = data.get("prefix_cache_hit_rate")
            if hit_rate is not None:
                hit_rates.append(float(hit_rate))
            else:
                hit_rates.append(0.0)

    # Plot per request by index
    indices = list(range(len(hit_rates)))

    plt.figure(figsize=(10, 6))
    plt.plot(indices, hit_rates, marker='o', linestyle='-', markersize=3)
    plt.title("Prefix Cache Hit Rate Per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Prefix Cache Hit Rate")
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    
    # Save PNG in the same directory as the input file
    output_png = os.path.splitext(input_file)[0] + ".png"
    plt.savefig(output_png, bbox_inches='tight')
    print(f"Saved plot to {output_png}")

if __name__ == "__main__":
    main()