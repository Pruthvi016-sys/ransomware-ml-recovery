"""
fmd_generator.py
Generates Filesystem Metadata Diff (FMD) files by comparing consecutive snapshots.
Each FMD captures what changed between snapshot T-1 and snapshot T.
This mirrors Rubrik CDM's FMD file generation exactly.

Output: dataset/fmd/fmd_XXX.csv  (one per snapshot, starting from snapshot 1)
Also outputs: dataset/fmd_summary.csv (one row per snapshot, aggregated features)
"""

import os
import csv
import math
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "dataset", "snapshots")
FMD_DIR       = os.path.join(os.path.dirname(__file__), "dataset", "fmd")
SUMMARY_PATH  = os.path.join(os.path.dirname(__file__), "dataset", "fmd_summary.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_snapshot(snapshot_id: int) -> dict:
    """Load snapshot CSV as dict: path -> row."""
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing snapshot: {path}")
    files = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            files[row["path"]] = row
    return files

def get_snapshot_meta(snapshot_id: int) -> tuple:
    """Return (snapshot_time, event_type) from first row of snapshot."""
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return row["snapshot_time"], row["event_type"]

def shannon_entropy(value: float) -> float:
    """Already computed per file — just return it."""
    return value

# ── FMD Core ──────────────────────────────────────────────────────────────────

def compute_fmd(prev: dict, curr: dict, snapshot_id: int) -> tuple[list, dict]:
    """
    Diff prev snapshot against curr snapshot.
    Returns:
      fmd_rows  — list of per-file change entries
      summary   — aggregated stats for this diff
    """
    prev_paths = set(prev.keys())
    curr_paths = set(curr.keys())

    added_paths   = curr_paths - prev_paths
    deleted_paths = prev_paths - curr_paths
    common_paths  = prev_paths & curr_paths

    # Find modified files (same path but content changed)
    modified_paths = set()
    for path in common_paths:
        if (prev[path]["size_bytes"]   != curr[path]["size_bytes"] or
            prev[path]["entropy"]      != curr[path]["entropy"]    or
            prev[path]["modified_time"] != curr[path]["modified_time"]):
            modified_paths.add(path)

    fmd_rows = []

    # Added files
    for path in added_paths:
        row = curr[path]
        fmd_rows.append({
            "snapshot_id":  snapshot_id,
            "change_type":  "added",
            "path":         path,
            "extension":    row["extension"],
            "size_bytes":   row["size_bytes"],
            "entropy":      row["entropy"],
            "acl":          row["acl"],
            "uid":          row["uid"],
            "gid":          row["gid"],
            "is_encrypted": row["is_encrypted"],
            "event_type":   row["event_type"],
        })

    # Deleted files
    for path in deleted_paths:
        row = prev[path]
        fmd_rows.append({
            "snapshot_id":  snapshot_id,
            "change_type":  "deleted",
            "path":         path,
            "extension":    row["extension"],
            "size_bytes":   row["size_bytes"],
            "entropy":      row["entropy"],
            "acl":          row["acl"],
            "uid":          row["uid"],
            "gid":          row["gid"],
            "is_encrypted": row.get("is_encrypted", "False"),
            "event_type":   row["event_type"],
        })

    # Modified files
    for path in modified_paths:
        row = curr[path]
        prev_row = prev[path]
        entropy_delta = float(row["entropy"]) - float(prev_row["entropy"])
        fmd_rows.append({
            "snapshot_id":  snapshot_id,
            "change_type":  "modified",
            "path":         path,
            "extension":    row["extension"],
            "size_bytes":   row["size_bytes"],
            "entropy":      row["entropy"],
            "acl":          row["acl"],
            "uid":          row["uid"],
            "gid":          row["gid"],
            "is_encrypted": row["is_encrypted"],
            "event_type":   row["event_type"],
        })

    # ── Aggregate summary stats ───────────────────────────────────────────────

    all_entropies = [float(r["entropy"]) for r in fmd_rows if r["entropy"]]
    modified_entropies = [
        float(curr[p]["entropy"]) for p in modified_paths
    ]
    prev_entropies = [
        float(prev[p]["entropy"]) for p in modified_paths if p in prev
    ]

    avg_entropy       = sum(all_entropies) / len(all_entropies) if all_entropies else 0.0
    avg_mod_entropy   = sum(modified_entropies) / len(modified_entropies) if modified_entropies else 0.0
    avg_prev_entropy  = sum(prev_entropies) / len(prev_entropies) if prev_entropies else 0.0
    entropy_delta     = avg_mod_entropy - avg_prev_entropy

    # Entropy variance across modified files
    if modified_entropies:
        mean = avg_mod_entropy
        variance = sum((e - mean) ** 2 for e in modified_entropies) / len(modified_entropies)
    else:
        variance = 0.0

    # High entropy file count (>7.0 signals encryption)
    high_entropy_count = sum(1 for e in all_entropies if e > 7.0)

    # Extension change ratio
    ext_changes = sum(
        1 for p in common_paths
        if prev[p]["extension"] != curr[p]["extension"]
    )
    ext_change_ratio = ext_changes / len(common_paths) if common_paths else 0.0

    # Bulk rename flag: lots of files with .locked extension added
    locked_count = sum(1 for r in fmd_rows if r["extension"] == ".locked")
    bulk_rename_flag = 1 if locked_count > 10 else 0

    # Size change ratio across modified files
    size_deltas = []
    for p in modified_paths:
        if p in prev:
            delta = abs(int(curr[p]["size_bytes"]) - int(prev[p]["size_bytes"]))
            size_deltas.append(delta)
    avg_size_delta = sum(size_deltas) / len(size_deltas) if size_deltas else 0.0

    # Delete/add ratio (key for mass move detection)
    n_added   = len(added_paths)
    n_deleted = len(deleted_paths)
    delete_add_ratio = (
        n_deleted / n_added if n_added > 0 else
        (10.0 if n_deleted > 0 else 0.0)
    )

    snapshot_time, event_type = get_snapshot_meta(snapshot_id)

    # Label: 0=clean, 1=ransomware, 2=mass_move
    label_map = {"clean": 0, "ransomware": 1, "mass_move": 2}
    label = label_map.get(event_type, 0)

    summary = {
        "snapshot_id":        snapshot_id,
        "snapshot_time":      snapshot_time,
        "event_type":         event_type,
        "label":              label,
        "total_files_prev":   len(prev),
        "total_files_curr":   len(curr),
        "files_added":        n_added,
        "files_deleted":      n_deleted,
        "files_modified":     len(modified_paths),
        "files_unchanged":    len(common_paths) - len(modified_paths),
        "delete_add_ratio":   round(delete_add_ratio, 4),
        "ext_change_ratio":   round(ext_change_ratio, 4),
        "bulk_rename_flag":   bulk_rename_flag,
        "locked_file_count":  locked_count,
        "avg_entropy":        round(avg_entropy, 4),
        "entropy_delta":      round(entropy_delta, 4),
        "entropy_variance":   round(variance, 4),
        "high_entropy_count": high_entropy_count,
        "avg_size_delta":     round(avg_size_delta, 2),
    }

    return fmd_rows, summary

def save_fmd(fmd_rows: list, snapshot_id: int):
    """Save per-file FMD entries to CSV."""
    os.makedirs(FMD_DIR, exist_ok=True)
    path = os.path.join(FMD_DIR, f"fmd_{snapshot_id:03d}.csv")
    if not fmd_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fmd_rows[0].keys())
        writer.writeheader()
        writer.writerows(fmd_rows)

def save_summary(summaries: list):
    """Save aggregated FMD summary to a single CSV."""
    if not summaries:
        return
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nFMD summary saved → {SUMMARY_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_fmd_generation():
    print("="*55)
    print("RANSOMWARE RECOVERY — FMD GENERATOR")
    print("="*55)

    # Count available snapshots
    snapshots = sorted([
        f for f in os.listdir(SNAPSHOTS_DIR) if f.endswith(".csv")
    ])
    n = len(snapshots)
    print(f"Found {n} snapshots in {SNAPSHOTS_DIR}\n")

    summaries = []
    prev = load_snapshot(0)

    for sid in range(1, n):
        curr = load_snapshot(sid)
        fmd_rows, summary = compute_fmd(prev, curr, sid)
        save_fmd(fmd_rows, sid)
        summaries.append(summary)

        tag = f"[{summary['event_type'].upper():<11}]"
        print(
            f"  [{sid:03d}] {tag} "
            f"+{summary['files_added']:3d} "
            f"-{summary['files_deleted']:3d} "
            f"~{summary['files_modified']:3d}  "
            f"entropy={summary['avg_entropy']:.3f}  "
            f"high_entropy={summary['high_entropy_count']:4d}  "
            f"locked={summary['locked_file_count']:4d}"
        )

        prev = curr

    save_summary(summaries)

    # Quick sanity check
    ransomware_rows = [s for s in summaries if s["event_type"] == "ransomware"]
    mass_move_rows  = [s for s in summaries if s["event_type"] == "mass_move"]
    clean_rows      = [s for s in summaries if s["event_type"] == "clean"]

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"  Clean snapshots     : {len(clean_rows)}")
    print(f"  Mass move snapshots : {len(mass_move_rows)}")
    print(f"  Ransomware snapshots: {len(ransomware_rows)}")
    if ransomware_rows:
        r = ransomware_rows[0]
        print(f"\n  First ransomware snapshot [{r['snapshot_id']:03d}]:")
        print(f"    avg_entropy     = {r['avg_entropy']}")
        print(f"    high_entropy    = {r['high_entropy_count']}")
        print(f"    locked_files    = {r['locked_file_count']}")
        print(f"    bulk_rename     = {r['bulk_rename_flag']}")
        print(f"    entropy_delta   = {r['entropy_delta']}")
    if mass_move_rows:
        m = mass_move_rows[0]
        print(f"\n  Mass move snapshot [{m['snapshot_id']:03d}]:")
        print(f"    delete_add_ratio = {m['delete_add_ratio']}")
        print(f"    avg_entropy      = {m['avg_entropy']}")
        print(f"    high_entropy     = {m['high_entropy_count']}")
    print(f"{'='*55}")
    print(f"\nFMD files saved → {FMD_DIR}")

if __name__ == "__main__":
    run_fmd_generation()
