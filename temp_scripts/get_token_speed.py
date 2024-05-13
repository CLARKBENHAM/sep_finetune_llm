import scipy.stats as stats
import re
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
import numpy as np


def extract_data_from_log(log_file_path):
    """Extracts data from log file, grouping by nvidia-smi output."""

    data_chunks = []
    current_chunk = {
        "timestamps": [],
        "token_sizes": [],
        "memory_used": [],
        "memory_after_empty": [],
    }

    with open(log_file_path, "r") as file:
        for line in file:
            # Check for nvidia-smi header
            if re.search(r"NVIDIA-SMI", line):
                if current_chunk["timestamps"]:  # Add previous chunk if exists
                    data_chunks.append(current_chunk)
                current_chunk = {
                    "timestamps": [],
                    "token_sizes": [],
                    "memory_used": [],
                    "memory_after_empty": [],
                }

            # Extract timestamp
            time_pattern = r"(\w+ \w+ \d+ \d+:\d+:\d+ \d+)"

            # time_match = re.search(r"(?<=^)\w{3}\s+\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\d{4}", line)
            time_match = re.search(time_pattern, line)
            if time_match:
                match = time_match.group(0)
                current_chunk["timestamps"].append(datetime.strptime(match, "%a %b %d %H:%M:%S %Y"))

            # Extract token size from list
            size_match = re.search(r"\[(\d{4})(?:,\s*\d{4})*\]", line)
            if size_match:
                current_chunk["token_sizes"] += [
                    [int(x) for x in size_match.group(0)[1:-1].split(",")]
                ]

            # Extract memory usage
            memory_match = re.search(r"(\d+\.\d+)\s*MB used$", line)
            if memory_match:
                current_chunk["memory_used"].append(float(memory_match.group(1)))
            memory_match = re.search(r"(\d+\.\d+)\s*MB used after empty", line)
            if memory_match:
                current_chunk["memory_after_empty"].append(float(memory_match.group(1)))

        if current_chunk["timestamps"]:  # Append the last chunk
            data_chunks.append(current_chunk)

    return data_chunks


def reg_plot(x1, y1, xlabel, ylabel, title=None, transformed_x1=None):
    if title is None:
        title = f"{ylabel} vs {xlabel}"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(
        x=x1, y=y1, scatter=True, ci=95, line_kws={"color": "red"}, scatter_kws={"s": 2}, ax=ax
    )
    ax.set_title(title, fontsize=25)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    corr, p = stats.pearsonr(x1, y1)
    ax.text(
        0.05,
        0.95,
        f"corr: {corr:.2f} p: {p:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=20,
    )

    # Calculate the slope and intercept of the line
    slope, intercept = np.polyfit(x1, y1, 1)
    # Add the equation to the plot
    ax.text(
        0.05,
        0.9,
        f"y = {slope:.2e}x + {intercept:.2e}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=20,
    )

    # Plot the transformed data on the second x-axis
    # Replace 'transformed_x1' with your transformed data
    if transformed_x1 is not None:
        # Create a second x-axis
        ax2 = ax.twiny()
        if False:
            sns.regplot(
                x=transformed_x1,
                y=y1,
                scatter=True,
                ci=95,
                line_kws={"color": "red"},
                scatter_kws={"s": 2},
                ax=ax2,
            )
        ax2.plot(transformed_x1, y1, "none")  # This 'plots' an invisible line to adjust the axis

        # Set the labels for both x-axes
        ax.set_xlabel(xlabel)
        ax2.set_xlabel("Transformed " + xlabel)  # Replace with your label for the transformed data
    plt.tight_layout()
    plt.show()


def analyze_data_chunks(data_chunks):
    """Calculates batch durations and plots data."""
    batch_durations = []
    all_token_sizes = []
    all_memory_used = []
    avg_token_sizes = []

    for i in range(1, len(data_chunks)):
        # Duration calculation (adjust as needed for your timestamp format)
        duration = data_chunks[i]["timestamps"][0] - data_chunks[i - 1]["timestamps"][0]
        batch_durations.append(duration.total_seconds())

        sq_token_sizes = sum([len(l) * (l[0] ** 2) for l in data_chunks[i - 1]["token_sizes"]])
        # print(len(data_chunks[i - 1]["memory_used"]), len(sq_token_sizes))
        all_token_sizes.extend([sq_token_sizes])
        avg_token_sizes += sum([len(l) * l[0] for l in data_chunks[i - 1]["token_sizes"]]) / 50
        mem = data_chunks[i - 1]["memory_used"]
        all_memory_used.extend([sum(mem) / len(mem)])

    reg_plot(
        all_token_sizes,
        batch_durations,
        "Total Token Cost of Batch (max in each batch squared)",
        "Batch Duration (seconds)",
        transformed_x1=avg_token_sizes,
    )

    reg_plot(all_memory_used, batch_durations, "Memory Used (MB)", "Batch Duration (seconds)")

    reg_plot(all_token_sizes, all_memory_used, "Total Token Cost of Batch", "Memory Used (MB)")
    if False:
        plt.figure(figsize=(10, 6))
        # plt.scatter(all_token_sizes, batch_durations)
        ax = sns.regplot(
            x=all_token_sizes,
            y=batch_durations,
            scatter=True,
            ci=95,
            line_kws={"color": "red"},
            scatter_kws={"s": 2},
        )

        plt.xlabel("Total Token's in Batch")
        plt.ylabel("Batch Duration (seconds)")
        plt.title("Batch Duration vs. Total Token's in Batch")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(all_memory_used, batch_durations)
        plt.xlabel("Memory Used (MB)")
        plt.ylabel("Batch Duration (seconds)")
        plt.title("Batch Duration vs. Memory Used")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(all_token_sizes, all_memory_used)
        plt.ylabel("Memory Used (MB)")
        plt.xlabel("Total Token's in Batch")
        plt.title("Total Token's in Batch vs. Memory Used")
        plt.grid(True)
        plt.show()


file_path = "/Users/clarkbenham/del/run1_log.txt"
data_chunks = extract_data_from_log(file_path)
analyze_data_chunks(data_chunks)
