"""
simulator.py
Generates a realistic filesystem snapshot timeline.
Each snapshot represents a backup taken at a point in time.
Snapshots are saved as CSVs in dataset/snapshots/.
"""

import os
import random
import math
import csv
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────

NUM_SNAPSHOTS = 50          # total snapshots in timeline
NUM_BASE_FILES = 500        # files in the initial filesystem
SNAPSHOT_INTERVAL_HOURS = 6 # time between snapshots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "snapshots")

EXTENSIONS = [
    ".docx", ".pdf", ".xlsx", ".txt", ".pptx",
    ".jpg", ".png", ".mp4", ".db", ".sql",
    ".py", ".js", ".json", ".xml", ".csv"
]

DIRECTORIES = [
    "/data/finance",
    "/data/hr",
    "/data/engineering",
    "/data/marketing",
    "/data/legal",
    "/data/backups",
    "/home/user1",
    "/home/user2",
    "/var/logs",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_entropy(data_bytes: bytes) -> float:
    """Shannon entropy of a byte sequence, scaled to 0-8 bits."""
    if not data_bytes:
        return 0.0
    freq = {}
    for b in data_bytes:
        freq[b] = freq.get(b, 0) + 1
    n = len(data_bytes)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        entropy -= p * math.log2(p)
    return round(entropy, 4)

def simulate_file_entropy(is_encrypted: bool = False) -> float:
    """
    Simulate realistic entropy values.
    Normal files: 3.0 - 6.5 (text, structured data)
    Encrypted files: 7.5 - 8.0 (high entropy, near random)
    """
    if is_encrypted:
        return round(random.uniform(7.5, 8.0), 4)
    else:
        return round(random.uniform(3.0, 6.5), 4)

def random_filename(ext: str = None) -> str:
    words = ["report", "backup", "data", "config", "log", "archive",
             "invoice", "contract", "notes", "summary", "export", "file"]
    name = random.choice(words) + "_" + str(random.randint(100, 9999))
    ext = ext or random.choice(EXTENSIONS)
    return name + ext

def random_path() -> str:
    directory = random.choice(DIRECTORIES)
    return os.path.join(directory, random_filename())

def make_file_entry(path: str, modified_time: datetime,
                    is_encrypted: bool = False) -> dict:
    """Create a single file metadata entry."""
    ext = os.path.splitext(path)[1]
    size = random.randint(1024, 10 * 1024 * 1024)  # 1KB to 10MB
    if is_encrypted:
        size = int(size * random.uniform(0.98, 1.02))  # size barely changes

    return {
        "path": path,
        "extension": ext,
        "size_bytes": size,
        "entropy": simulate_file_entropy(is_encrypted),
        "acl": random.choice(["rw-r--r--", "rwxr-xr-x", "rw-rw-r--"]),
        "uid": random.randint(1000, 1005),
        "gid": random.randint(100, 105),
        "modified_time": modified_time.isoformat(),
        "is_encrypted": is_encrypted,
    }

# ── Core Snapshot Builder ─────────────────────────────────────────────────────

def build_initial_filesystem(base_time: datetime) -> dict:
    """
    Build the initial filesystem state (snapshot 0).
    Returns a dict: path -> file_entry
    """
    filesystem = {}
    for _ in range(NUM_BASE_FILES):
        path = random_path()
        # avoid duplicate paths
        while path in filesystem:
            path = random_path()
        filesystem[path] = make_file_entry(path, base_time)
    return filesystem

def apply_normal_changes(filesystem: dict, snapshot_time: datetime) -> dict:
    """
    Apply realistic day-to-day changes:
    - Add a few files
    - Delete a few files
    - Modify some files
    """
    files = list(filesystem.keys())

    # Add 5-20 new files
    for _ in range(random.randint(5, 20)):
        path = random_path()
        while path in filesystem:
            path = random_path()
        filesystem[path] = make_file_entry(path, snapshot_time)

    # Delete 3-15 files
    to_delete = random.sample(files, min(random.randint(3, 15), len(files)))
    for path in to_delete:
        del filesystem[path]

    # Modify 10-30 files
    remaining = list(filesystem.keys())
    to_modify = random.sample(remaining, min(random.randint(10, 30), len(remaining)))
    for path in to_modify:
        filesystem[path] = make_file_entry(path, snapshot_time)

    return filesystem

def save_snapshot(filesystem: dict, snapshot_id: int, snapshot_time: datetime,
                  event_type: str):
    """Save a snapshot as a CSV file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"snapshot_{snapshot_id:03d}.csv")

    fieldnames = ["path", "extension", "size_bytes", "entropy",
                  "acl", "uid", "gid", "modified_time", "is_encrypted",
                  "snapshot_id", "snapshot_time", "event_type"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in filesystem.values():
            row = dict(entry)
            row["snapshot_id"] = snapshot_id
            row["snapshot_time"] = snapshot_time.isoformat()
            row["event_type"] = event_type
            writer.writerow(row)

    return filepath

# ── Main Simulation ───────────────────────────────────────────────────────────

def run_simulation():
    print(f"Starting simulation: {NUM_SNAPSHOTS} snapshots, {NUM_BASE_FILES} base files")
    print(f"Output: {OUTPUT_DIR}\n")

    base_time = datetime(2024, 1, 1, 0, 0, 0)
    filesystem = build_initial_filesystem(base_time)

    # Save snapshot 0 (initial clean state)
    save_snapshot(filesystem, 0, base_time, "clean")
    print(f"[000] clean — {len(filesystem)} files")

    for i in range(1, NUM_SNAPSHOTS):
        snapshot_time = base_time + timedelta(hours=i * SNAPSHOT_INTERVAL_HOURS)

        # Apply normal changes every snapshot
        filesystem = apply_normal_changes(filesystem, snapshot_time)

        save_snapshot(filesystem, i, snapshot_time, "clean")
        print(f"[{i:03d}] clean — {len(filesystem)} files")

    print(f"\nDone. {NUM_SNAPSHOTS} snapshots saved to {OUTPUT_DIR}")
    return OUTPUT_DIR

if __name__ == "__main__":
    run_simulation()