"""
blast_radius.py
Stage 3A: Blast Radius Analysis.

After ransomware is confirmed, identifies:
- Which files were encrypted
- Which directories were hit
- What file types were affected
- Severity score (low / medium / critical)
- Propagation pattern across directory tree

Output:
  data/dataset/blast_radius.csv     → per-directory impact summary
  data/dataset/affected_files.csv   → full list of affected files
"""

import os
import csv
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR            = os.path.dirname(os.path.dirname(__file__))
SNAPSHOTS_DIR       = os.path.join(BASE_DIR, "data", "dataset", "snapshots")
STAGE2B_RESULTS     = os.path.join(BASE_DIR, "data", "dataset", "stage2b_results.csv")
BLAST_RADIUS_OUT    = os.path.join(BASE_DIR, "data", "dataset", "blast_radius.csv")
AFFECTED_FILES_OUT  = os.path.join(BASE_DIR, "data", "dataset", "affected_files.csv")

RANSOMWARE_SNAPSHOT = 35   # first confirmed attack snapshot

# Severity thresholds (% of total files encrypted)
SEVERITY_LOW        = 0.20   # < 20%  → low
SEVERITY_MEDIUM     = 0.50   # 20-50% → medium
                             # > 50%  → critical

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_snapshot(snapshot_id: int) -> list[dict]:
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def get_top_directory(path: str) -> str:
    """Extract top-level directory from file path."""
    parts = path.strip("/").split("/")
    if len(parts) >= 2:
        return "/" + "/".join(parts[:2])
    return "/" + parts[0]

def get_severity(encrypted_ratio: float) -> str:
    if encrypted_ratio < SEVERITY_LOW:
        return "LOW"
    elif encrypted_ratio < SEVERITY_MEDIUM:
        return "MEDIUM"
    else:
        return "CRITICAL"

def load_stage2b_results() -> dict:
    """Return dict: snapshot_id -> encryption_prob."""
    results = {}
    with open(STAGE2B_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            results[int(row["snapshot_id"])] = {
                "encryption_prob":      float(row["encryption_prob"]),
                "ransomware_confirmed": int(row["ransomware_confirmed"]),
            }
    return results

# ── Core Analysis ─────────────────────────────────────────────────────────────

def analyze_blast_radius(attack_snapshot_id: int) -> tuple[list, list, dict]:
    """
    Compare the attack snapshot against the previous clean snapshot
    to determine blast radius.

    Returns:
      directory_summary  — per-directory impact stats
      affected_files     — list of all affected file entries
      overall_stats      — top-level summary dict
    """
    print(f"\nLoading attack snapshot [{attack_snapshot_id:03d}]...")
    attack_snap  = load_snapshot(attack_snapshot_id)

    print(f"Loading pre-attack snapshot [{attack_snapshot_id - 1:03d}]...")
    clean_snap   = load_snapshot(attack_snapshot_id - 1)

    total_files  = len(clean_snap)

    # Build lookup: path → row for clean snapshot
    clean_lookup = {row["path"]: row for row in clean_snap}

    # ── Identify affected files ───────────────────────────────────────────────
    affected_files = []

    # Files that are now encrypted (.locked extension or high entropy)
    attack_lookup  = {row["path"]: row for row in attack_snap}

    encrypted_count    = 0
    deleted_count      = 0
    ransom_note_count  = 0

    for row in attack_snap:
        is_encrypted = row.get("is_encrypted", "False") == "True"
        is_locked    = row["extension"] == ".locked"
        is_ransom    = "RESTORE_FILES" in row["path"]

        if is_ransom:
            ransom_note_count += 1
            continue

        if is_encrypted or is_locked:
            encrypted_count += 1
            # Try to find original filename (before .locked rename)
            original_path = row["path"].replace(".locked", "")
            affected_files.append({
                "file_path":         row["path"],
                "original_path":     original_path,
                "directory":         get_top_directory(row["path"]),
                "extension":         row["extension"],
                "size_bytes":        row["size_bytes"],
                "entropy":           row["entropy"],
                "change_type":       "encrypted",
                "attack_snapshot":   attack_snapshot_id,
            })

    # Files that were deleted (in clean but not in attack)
    for path, row in clean_lookup.items():
        if path not in attack_lookup:
            original_ext = row["extension"]
            if original_ext != ".locked":
                deleted_count += 1
                affected_files.append({
                    "file_path":       path,
                    "original_path":   path,
                    "directory":       get_top_directory(path),
                    "extension":       original_ext,
                    "size_bytes":      row["size_bytes"],
                    "entropy":         row["entropy"],
                    "change_type":     "deleted",
                    "attack_snapshot": attack_snapshot_id,
                })

    # ── Per-directory breakdown ───────────────────────────────────────────────
    dir_stats = defaultdict(lambda: {
        "encrypted": 0, "deleted": 0, "total_clean": 0
    })

    for row in clean_snap:
        d = get_top_directory(row["path"])
        dir_stats[d]["total_clean"] += 1

    for f in affected_files:
        d = f["directory"]
        if f["change_type"] == "encrypted":
            dir_stats[d]["encrypted"] += 1
        elif f["change_type"] == "deleted":
            dir_stats[d]["deleted"] += 1

    directory_summary = []
    for directory, stats in sorted(dir_stats.items()):
        total_d   = stats["total_clean"]
        affected  = stats["encrypted"] + stats["deleted"]
        ratio     = affected / total_d if total_d > 0 else 0.0
        directory_summary.append({
            "directory":        directory,
            "total_files":      total_d,
            "encrypted":        stats["encrypted"],
            "deleted":          stats["deleted"],
            "total_affected":   affected,
            "impact_ratio":     round(ratio, 4),
            "severity":         get_severity(ratio),
        })

    # ── File type breakdown ───────────────────────────────────────────────────
    ext_counts = defaultdict(int)
    for f in affected_files:
        ext = f["extension"] if f["extension"] != ".locked" else ".locked"
        ext_counts[ext] += 1

    # ── Overall stats ─────────────────────────────────────────────────────────
    total_affected    = encrypted_count + deleted_count
    encrypted_ratio   = encrypted_count / total_files if total_files > 0 else 0.0
    overall_severity  = get_severity(encrypted_ratio)

    overall_stats = {
        "attack_snapshot_id":   attack_snapshot_id,
        "total_files_before":   total_files,
        "total_files_after":    len(attack_snap),
        "encrypted_count":      encrypted_count,
        "deleted_count":        deleted_count,
        "ransom_notes":         ransom_note_count,
        "total_affected":       total_affected,
        "encrypted_ratio":      round(encrypted_ratio, 4),
        "severity":             overall_severity,
        "directories_hit":      sum(1 for d in directory_summary if d["total_affected"] > 0),
        "total_directories":    len(directory_summary),
        "top_ext_hit":          max(ext_counts, key=ext_counts.get) if ext_counts else "N/A",
        "ext_breakdown":        dict(sorted(ext_counts.items(),
                                           key=lambda x: -x[1])[:5]),
    }

    return directory_summary, affected_files, overall_stats

# ── Save ──────────────────────────────────────────────────────────────────────

def save_outputs(directory_summary, affected_files):
    if directory_summary:
        with open(BLAST_RADIUS_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=directory_summary[0].keys())
            writer.writeheader()
            writer.writerows(directory_summary)
        print(f"Blast radius saved → {BLAST_RADIUS_OUT}")

    if affected_files:
        with open(AFFECTED_FILES_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=affected_files[0].keys())
            writer.writeheader()
            writer.writerows(affected_files)
        print(f"Affected files saved → {AFFECTED_FILES_OUT}")

# ── Print Report ──────────────────────────────────────────────────────────────

def print_blast_report(directory_summary, overall_stats):
    print(f"\n{'='*55}")
    print("BLAST RADIUS REPORT")
    print(f"{'='*55}")
    print(f"  Attack Snapshot   : [{overall_stats['attack_snapshot_id']:03d}]")
    print(f"  Severity          : *** {overall_stats['severity']} ***")
    print(f"  Files Before      : {overall_stats['total_files_before']}")
    print(f"  Files After       : {overall_stats['total_files_after']}")
    print(f"  Encrypted         : {overall_stats['encrypted_count']} "
          f"({overall_stats['encrypted_ratio']*100:.1f}%)")
    print(f"  Deleted           : {overall_stats['deleted_count']}")
    print(f"  Ransom Notes      : {overall_stats['ransom_notes']}")
    print(f"  Directories Hit   : {overall_stats['directories_hit']} / "
          f"{overall_stats['total_directories']}")
    print(f"  Top Extension Hit : {overall_stats['top_ext_hit']}")
    print(f"\n  Extension Breakdown:")
    for ext, count in overall_stats["ext_breakdown"].items():
        print(f"    {ext:<12} {count} files")

    print(f"\n  Directory Impact:")
    print(f"  {'Directory':<30} {'Files':<8} {'Enc':<6} {'Del':<6} {'Severity'}")
    print(f"  {'-'*60}")
    for d in sorted(directory_summary, key=lambda x: -x["impact_ratio"]):
        if d["total_affected"] > 0:
            print(
                f"  {d['directory']:<30} "
                f"{d['total_files']:<8} "
                f"{d['encrypted']:<6} "
                f"{d['deleted']:<6} "
                f"{d['severity']}"
            )
    print(f"{'='*55}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_blast_radius(attack_snapshot_id: int = RANSOMWARE_SNAPSHOT):
    print("="*55)
    print("STAGE 3A — BLAST RADIUS ANALYSIS")
    print("="*55)

    directory_summary, affected_files, overall_stats = analyze_blast_radius(
        attack_snapshot_id
    )
    print_blast_report(directory_summary, overall_stats)
    save_outputs(directory_summary, affected_files)

    return directory_summary, affected_files, overall_stats

if __name__ == "__main__":
    run_blast_radius()