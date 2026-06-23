"""
event_injector.py
Injects ransomware and mass move events into the clean snapshot timeline.
Modifies specific snapshot CSVs in-place and updates their event_type labels.

Events injected:
  - RANSOMWARE at snapshot RANSOMWARE_SNAPSHOT_ID
  - MASS_MOVE   at snapshot MASS_MOVE_SNAPSHOT_ID
"""

import os
import csv
import random
import math
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "dataset", "snapshots")

RANSOMWARE_SNAPSHOT_ID = 35   # inject ransomware at snapshot 35
MASS_MOVE_SNAPSHOT_ID  = 20   # inject mass move at snapshot 20

# Ransomware behaviour knobs
RANSOMWARE_ENCRYPT_RATIO  = 0.75   # 75% of files get encrypted
RANSOMWARE_DELETE_RATIO   = 0.10   # 10% of files deleted outright
RANSOMWARE_LOCKED_EXT     = ".locked"

# Mass move behaviour knobs
MASS_MOVE_RATIO           = 0.40   # 40% of files moved to new directory
MASS_MOVE_TARGET_DIR      = "/data/archive"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_snapshot(snapshot_id: int) -> list[dict]:
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def save_snapshot(rows: list[dict], snapshot_id: int):
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def high_entropy() -> str:
    """Return a high entropy value string (7.5-8.0) simulating encrypted file."""
    return str(round(random.uniform(7.5, 8.0), 4))

def rename_to_locked(path: str) -> str:
    """Rename file path to .locked extension."""
    base = os.path.splitext(path)[0]
    return base + RANSOMWARE_LOCKED_EXT

def move_to_archive(path: str) -> str:
    """Move file path to archive directory."""
    filename = os.path.basename(path)
    return os.path.join(MASS_MOVE_TARGET_DIR, filename)

# ── Ransomware Injection ──────────────────────────────────────────────────────

def inject_ransomware(snapshot_id: int):
    """
    Simulates a ransomware attack on a snapshot:
    1. Encrypts 75% of files → rename to .locked, spike entropy to 7.5-8.0
    2. Deletes 10% of files outright
    3. Adds a few ransom note files
    4. Labels snapshot as 'ransomware'
    """
    print(f"\n[INJECTOR] Injecting RANSOMWARE into snapshot {snapshot_id:03d}...")
    rows = load_snapshot(snapshot_id)
    total = len(rows)

    random.shuffle(rows)

    n_encrypt = int(total * RANSOMWARE_ENCRYPT_RATIO)
    n_delete  = int(total * RANSOMWARE_DELETE_RATIO)

    encrypted_rows = []
    deleted_count  = 0
    kept_rows      = []

    for i, row in enumerate(rows):
        if i < n_encrypt:
            # Encrypt: rename, spike entropy, flag
            row["path"]         = rename_to_locked(row["path"])
            row["extension"]    = RANSOMWARE_LOCKED_EXT
            row["entropy"]      = high_entropy()
            row["is_encrypted"] = "True"
            row["event_type"]   = "ransomware"
            encrypted_rows.append(row)
        elif i < n_encrypt + n_delete:
            # Delete: just drop the row
            deleted_count += 1
        else:
            row["event_type"] = "ransomware"
            kept_rows.append(row)

    # Add ransom note files
    ransom_notes = []
    for j in range(5):
        note_path = f"/data/RESTORE_FILES_{j}.txt"
        ransom_notes.append({
            "path":          note_path,
            "extension":     ".txt",
            "size_bytes":    random.randint(512, 2048),
            "entropy":       str(round(random.uniform(3.5, 4.5), 4)),
            "acl":           "rw-r--r--",
            "uid":           "0",
            "gid":           "0",
            "modified_time": rows[0]["snapshot_time"],
            "is_encrypted":  "False",
            "snapshot_id":   str(snapshot_id),
            "snapshot_time": rows[0]["snapshot_time"],
            "event_type":    "ransomware",
        })

    final_rows = encrypted_rows + kept_rows + ransom_notes

    save_snapshot(final_rows, snapshot_id)

    print(f"  Total files before : {total}")
    print(f"  Encrypted          : {len(encrypted_rows)} ({RANSOMWARE_ENCRYPT_RATIO*100:.0f}%)")
    print(f"  Deleted            : {deleted_count} ({RANSOMWARE_DELETE_RATIO*100:.0f}%)")
    print(f"  Ransom notes added : {len(ransom_notes)}")
    print(f"  Total files after  : {len(final_rows)}")
    print(f"  Label              : ransomware")

# ── Mass Move Injection ───────────────────────────────────────────────────────

def inject_mass_move(snapshot_id: int):
    """
    Simulates a large file move operation (NOT ransomware — false positive case):
    1. Takes 40% of files, deletes from original path, re-adds at new path
    2. Entropy stays normal (files not encrypted)
    3. Labels snapshot as 'mass_move'

    This looks like ransomware (bulk deletes + adds) but entropy stays low.
    The Stage 2A false positive filter should catch this.
    """
    print(f"\n[INJECTOR] Injecting MASS MOVE into snapshot {snapshot_id:03d}...")
    rows = load_snapshot(snapshot_id)
    total = len(rows)

    random.shuffle(rows)
    n_move = int(total * MASS_MOVE_RATIO)

    moved_rows  = []
    static_rows = []

    for i, row in enumerate(rows):
        if i < n_move:
            # Move: change path to archive dir, keep entropy same
            old_path         = row["path"]
            row["path"]      = move_to_archive(old_path)
            row["event_type"] = "mass_move"
            moved_rows.append(row)
        else:
            row["event_type"] = "mass_move"
            static_rows.append(row)

    final_rows = moved_rows + static_rows
    save_snapshot(final_rows, snapshot_id)

    print(f"  Total files        : {total}")
    print(f"  Files moved        : {n_move} ({MASS_MOVE_RATIO*100:.0f}%)")
    print(f"  Destination        : {MASS_MOVE_TARGET_DIR}")
    print(f"  Label              : mass_move")

# ── Summary Printer ───────────────────────────────────────────────────────────

def print_event_summary():
    print("\n" + "="*55)
    print("EVENT INJECTION SUMMARY")
    print("="*55)
    print(f"  Snapshots 00–{RANSOMWARE_SNAPSHOT_ID-1:02d}  → clean")
    print(f"  Snapshot  {MASS_MOVE_SNAPSHOT_ID:02d}      → mass_move (false positive)")
    print(f"  Snapshot  {RANSOMWARE_SNAPSHOT_ID:02d}      → ransomware (attack)")
    print(f"  Snapshots {RANSOMWARE_SNAPSHOT_ID+1:02d}–49  → ransomware (post-attack)")
    print("="*55)
    print("\nExpected pipeline behaviour:")
    print(f"  Stage 1 fires      : snapshot {RANSOMWARE_SNAPSHOT_ID} (and {MASS_MOVE_SNAPSHOT_ID})")
    print(f"  Stage 2A filters   : snapshot {MASS_MOVE_SNAPSHOT_ID} (mass move, not ransomware)")
    print(f"  Stage 2B confirms  : snapshot {RANSOMWARE_SNAPSHOT_ID}")
    print(f"  Recovery point     : snapshot {RANSOMWARE_SNAPSHOT_ID - 1:03d}")

# ── Post-attack snapshots ─────────────────────────────────────────────────────

def label_post_attack_snapshots():
    """
    Snapshots after the ransomware event remain in infected state.
    Re-label them as ransomware so the pipeline knows.
    """
    for sid in range(RANSOMWARE_SNAPSHOT_ID + 1, 50):
        rows = load_snapshot(sid)
        for row in rows:
            row["event_type"] = "ransomware"
        save_snapshot(rows, sid)
    print(f"\n[INJECTOR] Labelled snapshots {RANSOMWARE_SNAPSHOT_ID+1}–49 as post-attack ransomware")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_injection():
    print("="*55)
    print("RANSOMWARE RECOVERY — EVENT INJECTOR")
    print("="*55)

    inject_mass_move(MASS_MOVE_SNAPSHOT_ID)
    inject_ransomware(RANSOMWARE_SNAPSHOT_ID)
    label_post_attack_snapshots()
    print_event_summary()

if __name__ == "__main__":
    run_injection()