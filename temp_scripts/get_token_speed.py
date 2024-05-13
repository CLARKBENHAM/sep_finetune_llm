import re
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


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


def analyze_data_chunks(data_chunks):
    """Calculates batch durations and plots data."""
    batch_durations = []
    all_token_sizes = []
    all_memory_used = []

    for i in range(1, len(data_chunks)):
        # Duration calculation (adjust as needed for your timestamp format)
        duration = data_chunks[i]["timestamps"][0] - data_chunks[i - 1]["timestamps"][0]
        batch_durations.append(duration.total_seconds())

        sq_token_sizes = sum([len(l) * (l[0] ** 2) for l in data_chunks[i - 1]["token_sizes"]])
        # print(len(data_chunks[i - 1]["memory_used"]), len(sq_token_sizes))
        all_token_sizes.extend([sq_token_sizes])
        mem = data_chunks[i - 1]["memory_used"]
        all_memory_used.extend([sum(mem) / len(mem)])

    plt.figure(figsize=(10, 6))
    plt.scatter(all_token_sizes, batch_durations)
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
