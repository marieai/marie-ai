import re
import sys
from collections import defaultdict

import numpy as np


def parse_log_file(file_path):
    """Parses the log file to extract performance metrics."""
    metrics = defaultdict(list)
    # Regex to capture all metrics from the new structured log format
    log_pattern = re.compile(
        r"cb_total=([\d.]+)s \(etcd=([\d.]+)s \[lease=([\d.]+)s, put=([\d.]+)s\], signal=([\d.]+)s\)"
    )

    with open(file_path, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                metrics['cb_total'].append(float(match.group(1)))
                metrics['etcd'].append(float(match.group(2)))
                metrics['lease'].append(float(match.group(3)))
                metrics['put'].append(float(match.group(4)))
                metrics['signal'].append(float(match.group(5)))

    return metrics


def print_histogram(title, data, bins=10, width=50):
    """Calculates and prints statistics and a text-based histogram."""
    if not data:
        print(f"\n--- {title} ---")
        print("No data found.")
        return

    data_np = np.array(data) * 1000  # Convert to milliseconds

    print(f"\n--- {title} (in ms) ---")
    print(f"Count    : {len(data_np):,}")
    print(f"Min      : {np.min(data_np):.2f} ms")
    print(f"Max      : {np.max(data_np):.2f} ms")
    print(f"Mean     : {np.mean(data_np):.2f} ms")
    print(f"Std Dev  : {np.std(data_np):.2f} ms")
    print("\nPercentiles:")
    print(f"  50th (Median): {np.percentile(data_np, 50):.2f} ms")
    print(f"  90th         : {np.percentile(data_np, 90):.2f} ms")
    print(f"  95th         : {np.percentile(data_np, 95):.2f} ms")
    print(f"  99th         : {np.percentile(data_np, 99):.2f} ms")

    # Calculate histogram
    counts, bin_edges = np.histogram(data_np, bins=bins)
    max_count = np.max(counts) if len(counts) > 0 else 0

    print("\nDistribution Histogram:")
    for i in range(len(counts)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        count = counts[i]

        # Calculate bar length
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = '#' * bar_len

        # Percentage of total
        percentage = (count / len(data_np)) * 100 if len(data_np) > 0 else 0

        print(
            f"{bin_start:8.2f} - {bin_end:8.2f} ms | {count:8,d} | {bar} ({percentage:.1f}%)"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/parse_stress_log.py <path_to_log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    try:
        all_metrics = parse_log_file(log_file)
        print_histogram("Total Callback Time (cb_total)", all_metrics['cb_total'])
        print_histogram("ETCD Time (lease + put)", all_metrics['etcd'])
        print_histogram("ETCD Lease Time", all_metrics['lease'])
        print_histogram("ETCD Put Time", all_metrics['put'])
        print_histogram("Signal Time", all_metrics['signal'])

    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
