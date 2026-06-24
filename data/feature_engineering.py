"""
feature_engineering.py
Reads fmd_summary.csv and engineers features for ML models.

Produces two output CSVs:
  dataset/features_stage1.csv  → for Stage 1 anomaly detection (behavior features)
  dataset/features_stage2.csv  → for Stage 2 encryption detection (entropy features)

Also adds rolling/temporal features using a sliding window over snapshot history.
"""

import os
import csv
import math

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
SUMMARY_PATH    = os.path.join(BASE_DIR, "data", "dataset", "fmd_summary.csv")
STAGE1_OUT      = os.path.join(BASE_DIR, "data", "dataset", "features_stage1.csv")
STAGE2_OUT      = os.path.join(BASE_DIR, "data", "dataset", "features_stage2.csv")

ROLLING_WINDOW  = 5   # snapshots to look back for baseline

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_summary() -> list[dict]:
    with open(SUMMARY_PATH, newline="") as f:
        return list(csv.DictReader(f))

def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def rolling_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def rolling_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = rolling_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build Stage 1 and Stage 2 feature sets from raw FMD summary rows.

    Stage 1 features — filesystem BEHAVIOR (feeds Isolation Forest):
      Captures unusual activity patterns regardless of encryption.

    Stage 2 features — ENTROPY signals (feeds DNN):
      Captures encryption-specific signals, only meaningful after Stage 1 fires.
    """

    stage1_rows = []
    stage2_rows = []

    # History buffers for rolling features
    entropy_history      = []
    files_added_history  = []
    files_deleted_history= []
    anomaly_score_history= []   # placeholder, filled by Stage 1 model later

    for i, row in enumerate(rows):
        sid          = safe_int(row["snapshot_id"])
        label        = safe_int(row["label"])
        event_type   = row["event_type"]

        # Raw counts
        files_added    = safe_int(row["files_added"])
        files_deleted  = safe_int(row["files_deleted"])
        files_modified = safe_int(row["files_modified"])
        files_unchanged= safe_int(row["files_unchanged"])
        total_curr     = safe_int(row["total_files_curr"])
        total_prev     = safe_int(row["total_files_prev"])

        # Entropy
        avg_entropy        = safe_float(row["avg_entropy"])
        entropy_delta      = safe_float(row["entropy_delta"])
        entropy_variance   = safe_float(row["entropy_variance"])
        high_entropy_count = safe_int(row["high_entropy_count"])

        # Ratios
        delete_add_ratio   = safe_float(row["delete_add_ratio"])
        ext_change_ratio   = safe_float(row["ext_change_ratio"])
        bulk_rename_flag   = safe_int(row["bulk_rename_flag"])
        locked_file_count  = safe_int(row["locked_file_count"])
        avg_size_delta     = safe_float(row["avg_size_delta"])

        # ── Rolling / temporal features ───────────────────────────────────────

        # Use last ROLLING_WINDOW clean-ish snapshots for baseline
        window_entropy   = entropy_history[-ROLLING_WINDOW:]
        window_added     = files_added_history[-ROLLING_WINDOW:]
        window_deleted   = files_deleted_history[-ROLLING_WINDOW:]

        rolling_avg_entropy  = rolling_mean(window_entropy)
        rolling_std_entropy  = rolling_std(window_entropy)
        entropy_zscore       = (
            (avg_entropy - rolling_avg_entropy) / rolling_std_entropy
            if rolling_std_entropy > 0 else 0.0
        )

        rolling_avg_added    = rolling_mean(window_added)
        rolling_avg_deleted  = rolling_mean(window_deleted)

        # How much does this snapshot deviate from recent normal?
        added_spike   = files_added   - rolling_avg_added
        deleted_spike = files_deleted - rolling_avg_deleted

        # Total churn rate: what % of files changed this snapshot
        total_changed = files_added + files_deleted + files_modified
        churn_rate    = total_changed / total_curr if total_curr > 0 else 0.0

        # File count delta from previous snapshot
        file_count_delta = total_curr - total_prev
        file_count_delta_ratio = (
            file_count_delta / total_prev if total_prev > 0 else 0.0
        )

        # High entropy ratio: what fraction of changed files are high entropy
        high_entropy_ratio = (
            high_entropy_count / total_changed if total_changed > 0 else 0.0
        )

        # ── Stage 1 feature row (behavior) ────────────────────────────────────
        s1 = {
            "snapshot_id":            sid,
            "event_type":             event_type,
            "label":                  label,
            # Raw behavior
            "files_added":            files_added,
            "files_deleted":          files_deleted,
            "files_modified":         files_modified,
            "files_unchanged":        files_unchanged,
            "file_count_delta":       file_count_delta,
            "file_count_delta_ratio": round(file_count_delta_ratio, 6),
            "churn_rate":             round(churn_rate, 6),
            # Ratio features
            "delete_add_ratio":       delete_add_ratio,
            "ext_change_ratio":       ext_change_ratio,
            "bulk_rename_flag":       bulk_rename_flag,
            "locked_file_count":      locked_file_count,
            "avg_size_delta":         avg_size_delta,
            # Rolling / temporal
            "rolling_avg_entropy":    round(rolling_avg_entropy, 4),
            "rolling_std_entropy":    round(rolling_std_entropy, 4),
            "entropy_zscore":         round(entropy_zscore, 4),
            "rolling_avg_added":      round(rolling_avg_added, 4),
            "rolling_avg_deleted":    round(rolling_avg_deleted, 4),
            "added_spike":            round(added_spike, 4),
            "deleted_spike":          round(deleted_spike, 4),
            # Light entropy signal (behavior-level)
            "avg_entropy":            avg_entropy,
            "high_entropy_count":     high_entropy_count,
            "high_entropy_ratio":     round(high_entropy_ratio, 6),
        }

        # ── Stage 2 feature row (encryption signals) ──────────────────────────
        s2 = {
            "snapshot_id":         sid,
            "event_type":          event_type,
            "label":               label,
            # Core encryption signals
            "avg_entropy":         avg_entropy,
            "entropy_delta":       entropy_delta,
            "entropy_variance":    entropy_variance,
            "high_entropy_count":  high_entropy_count,
            "high_entropy_ratio":  round(high_entropy_ratio, 6),
            "locked_file_count":   locked_file_count,
            "bulk_rename_flag":    bulk_rename_flag,
            "ext_change_ratio":    ext_change_ratio,
            # Entropy vs baseline
            "rolling_avg_entropy": round(rolling_avg_entropy, 4),
            "rolling_std_entropy": round(rolling_std_entropy, 4),
            "entropy_zscore":      round(entropy_zscore, 4),
            # Supporting signals
            "avg_size_delta":      avg_size_delta,
            "delete_add_ratio":    delete_add_ratio,
            "churn_rate":          round(churn_rate, 6),
        }

        stage1_rows.append(s1)
        stage2_rows.append(s2)

        # Update history buffers
        entropy_history.append(avg_entropy)
        files_added_history.append(files_added)
        files_deleted_history.append(files_deleted)

    return stage1_rows, stage2_rows

def save_features(rows: list[dict], path: str):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def run_feature_engineering():
    print("="*55)
    print("RANSOMWARE RECOVERY — FEATURE ENGINEERING")
    print("="*55)

    rows = load_summary()
    print(f"Loaded {len(rows)} rows from fmd_summary.csv")

    stage1_rows, stage2_rows = engineer_features(rows)

    save_features(stage1_rows, STAGE1_OUT)
    save_features(stage2_rows, STAGE2_OUT)

    print(f"\nStage 1 features : {len(stage1_rows[0]) - 3} features → {STAGE1_OUT}")
    print(f"Stage 2 features : {len(stage2_rows[0]) - 3} features → {STAGE2_OUT}")

    # Print a few key rows to verify signals
    print(f"\n{'='*55}")
    print("SIGNAL CHECK")
    print(f"{'='*55}")
    print(f"{'Snapshot':<10} {'Event':<12} {'entropy_z':<12} {'high_ent':<10} {'locked':<8} {'bulk':<6}")
    print("-"*55)
    for r in stage1_rows:
        if r["event_type"] != "clean" or r["snapshot_id"] in [34, 1]:
            print(
                f"  [{r['snapshot_id']:03d}]   "
                f"{r['event_type']:<12} "
                f"{r['entropy_zscore']:<12} "
                f"{r['high_entropy_count']:<10} "
                f"{r['locked_file_count']:<8} "
                f"{r['bulk_rename_flag']:<6}"
            )
    print(f"{'='*55}")

if __name__ == "__main__":
    run_feature_engineering()
