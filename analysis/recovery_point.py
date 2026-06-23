"""
recovery_point.py
Stage 3B: Recovery Point Identification.

Walks back through snapshot timeline to find the last clean snapshot
before the ransomware attack. Also computes:
- Data loss window (time between clean point and attack)
- MTTR estimate
- Files safe to restore
- Diff between clean point and attack point

Output:
  data/dataset/recovery_report.csv  → recovery recommendation
  data/dataset/recovery_diff.csv    → file-level diff clean vs attack
"""

import os
import csv
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR            = os.path.dirname(os.path.dirname(__file__))
SNAPSHOTS_DIR       = os.path.join(BASE_DIR, "data", "dataset", "snapshots")
STAGE1_RESULTS      = os.path.join(BASE_DIR, "data", "dataset", "stage1_results.csv")
STAGE2B_RESULTS     = os.path.join(BASE_DIR, "data", "dataset", "stage2b_results.csv")
RECOVERY_REPORT_OUT = os.path.join(BASE_DIR, "data", "dataset", "recovery_report.csv")
RECOVERY_DIFF_OUT   = os.path.join(BASE_DIR, "data", "dataset", "recovery_diff.csv")

# Thresholds
ANOMALY_SCORE_THRESHOLD  = 0.0    # IF score: above this = normal
ENCRYPTION_PROB_THRESHOLD = 0.5   # DNN prob: below this = clean

# Average time between snapshots (hours) — matches simulator config
SNAPSHOT_INTERVAL_HOURS  = 6

# MTTR baseline without ML (manual investigation estimate in hours)
MANUAL_MTTR_HOURS        = 72

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_stage1_results() -> dict:
    results = {}
    with open(STAGE1_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            results[int(row["snapshot_id"])] = {
                "anomaly_score": float(row["anomaly_score"]),
                "anomaly_flag":  int(row["anomaly_flag"]),
                "event_type":    row["event_type"],
            }
    return results

def load_stage2b_results() -> dict:
    results = {}
    with open(STAGE2B_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            results[int(row["snapshot_id"])] = {
                "encryption_prob":      float(row["encryption_prob"]),
                "ransomware_confirmed": int(row["ransomware_confirmed"]),
            }
    return results

def get_snapshot_time(snapshot_id: int) -> datetime:
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return datetime.fromisoformat(row["snapshot_time"])

def get_snapshot_file_count(snapshot_id: int) -> int:
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))

def load_snapshot_files(snapshot_id: int) -> dict:
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{snapshot_id:03d}.csv")
    files = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            files[row["path"]] = row
    return files

# ── Recovery Point Search ─────────────────────────────────────────────────────

def find_recovery_point(stage1: dict, stage2b: dict,
                        attack_snapshot_id: int) -> int:
    """
    Walk backwards from attack snapshot.
    Return the last snapshot ID that is both:
      - Not anomalous (Stage 1 says normal)
      - Not encrypted (Stage 2B prob < threshold)
    """
    sid = attack_snapshot_id - 1

    while sid >= 0:
        s1  = stage1.get(sid, {})
        s2b = stage2b.get(sid, {})

        anomaly_flag  = s1.get("anomaly_flag", 0)
        enc_prob      = s2b.get("encryption_prob", 0.0)
        confirmed     = s2b.get("ransomware_confirmed", 0)

        is_clean = (anomaly_flag == 0 and
                    enc_prob < ENCRYPTION_PROB_THRESHOLD and
                    confirmed == 0)

        if is_clean:
            return sid

        sid -= 1

    return 0   # fallback to snapshot 0

# ── Recovery Diff ─────────────────────────────────────────────────────────────

def compute_recovery_diff(clean_id: int,
                          attack_id: int) -> tuple[list, dict]:
    """
    Diff between clean recovery point and attack snapshot.
    Shows exactly what data was lost / corrupted.
    """
    clean_files  = load_snapshot_files(clean_id)
    attack_files = load_snapshot_files(attack_id)

    diff_rows = []

    # Files that were encrypted (exist in attack as .locked)
    attack_paths = set(attack_files.keys())
    clean_paths  = set(clean_files.keys())

    # Map original → locked path
    encrypted_originals = set()
    for path in attack_paths:
        if path.endswith(".locked"):
            original = path.replace(".locked", "")
            encrypted_originals.add(original)

    for path in clean_paths:
        if path in encrypted_originals:
            diff_rows.append({
                "original_path":  path,
                "status":         "encrypted",
                "extension":      clean_files[path]["extension"],
                "size_bytes":     clean_files[path]["size_bytes"],
                "clean_entropy":  clean_files[path]["entropy"],
                "attack_entropy": attack_files.get(
                    path.replace(clean_files[path]["extension"], ".locked"),
                    {}
                ).get("entropy", "N/A"),
                "recoverable":    "YES — restore from clean snapshot",
            })
        elif path not in attack_paths:
            diff_rows.append({
                "original_path":  path,
                "status":         "deleted",
                "extension":      clean_files[path]["extension"],
                "size_bytes":     clean_files[path]["size_bytes"],
                "clean_entropy":  clean_files[path]["entropy"],
                "attack_entropy": "N/A",
                "recoverable":    "YES — restore from clean snapshot",
            })

    # Ransom notes added (new files in attack not in clean)
    for path in attack_paths:
        if path not in clean_paths and "RESTORE_FILES" in path:
            diff_rows.append({
                "original_path":  path,
                "status":         "ransom_note_added",
                "extension":      ".txt",
                "size_bytes":     attack_files[path]["size_bytes"],
                "clean_entropy":  "N/A",
                "attack_entropy": attack_files[path]["entropy"],
                "recoverable":    "DELETE — do not restore",
            })

    diff_summary = {
        "encrypted":     sum(1 for r in diff_rows if r["status"] == "encrypted"),
        "deleted":       sum(1 for r in diff_rows if r["status"] == "deleted"),
        "ransom_notes":  sum(1 for r in diff_rows if r["status"] == "ransom_note_added"),
        "total_affected":len(diff_rows),
        "recoverable":   sum(1 for r in diff_rows if "YES" in r["recoverable"]),
    }

    return diff_rows, diff_summary

# ── MTTR Calculation ──────────────────────────────────────────────────────────

def estimate_mttr(clean_id: int, attack_id: int,
                  detection_id: int) -> dict:
    """
    Estimate Mean Time To Recover with and without ML.

    Without ML: manual log analysis + investigation (MANUAL_MTTR_HOURS)
    With ML:    data loss window + small recovery overhead
    """
    clean_time     = get_snapshot_time(clean_id)
    attack_time    = get_snapshot_time(attack_id)
    detection_time = get_snapshot_time(detection_id)

    data_loss_hours = (attack_time - clean_time).total_seconds() / 3600
    detection_delay = (detection_time - attack_time).total_seconds() / 3600

    # Recovery time with ML: data loss window + 2hr recovery overhead
    ml_mttr_hours = data_loss_hours + 2.0
    time_saved    = max(0, MANUAL_MTTR_HOURS - ml_mttr_hours)
    reduction_pct = (time_saved / MANUAL_MTTR_HOURS * 100
                     if MANUAL_MTTR_HOURS > 0 else 0)

    return {
        "clean_snapshot_time":   clean_time.isoformat(),
        "attack_snapshot_time":  attack_time.isoformat(),
        "detection_snapshot_id": detection_id,
        "data_loss_hours":       round(data_loss_hours, 2),
        "detection_delay_hours": round(detection_delay, 2),
        "ml_mttr_hours":         round(ml_mttr_hours, 2),
        "manual_mttr_hours":     MANUAL_MTTR_HOURS,
        "hours_saved":           round(time_saved, 2),
        "mttr_reduction_pct":    round(reduction_pct, 1),
    }

# ── Save ──────────────────────────────────────────────────────────────────────

def save_report(report: dict):
    with open(RECOVERY_REPORT_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report.keys())
        writer.writeheader()
        writer.writerow(report)
    print(f"Recovery report saved → {RECOVERY_REPORT_OUT}")

def save_diff(diff_rows: list):
    if not diff_rows:
        return
    with open(RECOVERY_DIFF_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diff_rows[0].keys())
        writer.writeheader()
        writer.writerows(diff_rows)
    print(f"Recovery diff saved  → {RECOVERY_DIFF_OUT}")

# ── Print Report ──────────────────────────────────────────────────────────────

def print_recovery_report(clean_id, attack_id, mttr, diff_summary):
    print(f"\n{'='*55}")
    print("RECOVERY RECOMMENDATION")
    print(f"{'='*55}")
    print(f"  Attack Snapshot       : [{attack_id:03d}]")
    print(f"  ✓ Clean Recovery Point: [{clean_id:03d}]  ← RESTORE FROM THIS")
    print(f"\n  Timeline:")
    print(f"    Last clean snapshot : {mttr['clean_snapshot_time']}")
    print(f"    Attack occurred     : {mttr['attack_snapshot_time']}")
    print(f"    Data loss window    : {mttr['data_loss_hours']} hours")
    print(f"\n  Recovery Estimate:")
    print(f"    MTTR with ML        : {mttr['ml_mttr_hours']} hours")
    print(f"    MTTR without ML     : {mttr['manual_mttr_hours']} hours")
    print(f"    Time saved          : {mttr['hours_saved']} hours "
          f"({mttr['mttr_reduction_pct']}% reduction)")
    print(f"\n  Data Loss Summary:")
    print(f"    Encrypted files     : {diff_summary['encrypted']}")
    print(f"    Deleted files       : {diff_summary['deleted']}")
    print(f"    Ransom notes        : {diff_summary['ransom_notes']}")
    print(f"    Recoverable files   : {diff_summary['recoverable']}")
    print(f"\n  Action:")
    print(f"    → Restore all files from snapshot [{clean_id:03d}]")
    print(f"    → Delete ransom notes ({diff_summary['ransom_notes']} files)")
    print(f"    → Investigate infection vector before reconnecting to network")
    print(f"{'='*55}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_recovery_point(attack_snapshot_id: int = 35):
    print("="*55)
    print("STAGE 3B — RECOVERY POINT IDENTIFICATION")
    print("="*55)

    stage1  = load_stage1_results()
    stage2b = load_stage2b_results()

    print(f"\nSearching for clean recovery point before snapshot [{attack_snapshot_id:03d}]...")
    clean_id = find_recovery_point(stage1, stage2b, attack_snapshot_id)
    print(f"Found: snapshot [{clean_id:03d}]")

    print(f"\nComputing recovery diff [{clean_id:03d}] → [{attack_snapshot_id:03d}]...")
    diff_rows, diff_summary = compute_recovery_diff(clean_id, attack_snapshot_id)

    # Detection = first snapshot flagged by Stage 1
    detection_id = min(
        (sid for sid, r in stage1.items() if r["anomaly_flag"] == 1),
        default=attack_snapshot_id
    )

    mttr = estimate_mttr(clean_id, attack_snapshot_id, detection_id)
    print_recovery_report(clean_id, attack_snapshot_id, mttr, diff_summary)

    # Save
    report = {
        "clean_recovery_snapshot":  clean_id,
        "attack_snapshot":          attack_snapshot_id,
        **mttr,
        **{f"diff_{k}": v for k, v in diff_summary.items()},
    }
    save_report(report)
    save_diff(diff_rows)

    return clean_id, mttr, diff_summary

if __name__ == "__main__":
    run_recovery_point()