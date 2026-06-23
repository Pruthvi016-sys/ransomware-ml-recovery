"""
stage2a_mass_move.py
Stage 2A: Mass Move False Positive Filter.

Only runs on snapshots flagged by Stage 1.
Determines whether the anomaly is a benign mass file move
(bulk delete + re-add at new path) rather than ransomware.

Uses a rule-based + scoring approach:
  - delete_add_ratio close to 1.0  → files moved, not deleted
  - avg_entropy stays low          → files not encrypted
  - high_entropy_count == 0        → no encrypted files detected
  - bulk_rename_flag == 0          → no .locked extensions

If a snapshot passes the mass move filter → NOT ransomware, skip Stage 2B.
If it fails the filter → forward to Stage 2B encryption detection.

Output:
  data/dataset/stage2a_results.csv  → filter decisions for Stage 1 flagged snapshots
"""

import os
import csv

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(os.path.dirname(__file__))
FEATURES_PATH    = os.path.join(BASE_DIR, "data", "dataset", "features_stage1.csv")
STAGE1_RESULTS   = os.path.join(BASE_DIR, "data", "dataset", "stage1_results.csv")
RESULTS_PATH     = os.path.join(BASE_DIR, "data", "dataset", "stage2a_results.csv")

# ── Thresholds for mass move detection ───────────────────────────────────────
# A snapshot looks like a mass move if ALL of:
#   1. delete/add ratio is close to 1 (files moved = deleted + re-added)
#   2. average entropy stays in normal range
#   3. no high entropy files detected
#   4. no bulk rename to .locked

RATIO_LOW        = 0.70   # delete_add_ratio lower bound
RATIO_HIGH       = 1.40   # delete_add_ratio upper bound
MAX_AVG_ENTROPY  = 6.50   # encrypted files push avg above this
MAX_HIGH_ENTROPY = 5      # allow tiny number of naturally high-entropy files
MIN_MOVED_FILES  = 50     # must involve significant number of files to be a "mass" move

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_stage1_flags() -> set:
    """Return set of snapshot_ids flagged by Stage 1."""
    flagged = set()
    with open(STAGE1_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["stage2_trigger"]) == 1:
                flagged.add(int(row["snapshot_id"]))
    return flagged

def load_features() -> dict:
    """Return dict: snapshot_id -> feature row."""
    features = {}
    with open(FEATURES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            features[int(row["snapshot_id"])] = row
    return features

def safe_float(val, default=0.0) -> float:
    try:    return float(val)
    except: return default

def safe_int(val, default=0) -> int:
    try:    return int(val)
    except: return default

# ── Mass Move Scoring ─────────────────────────────────────────────────────────

def compute_mass_move_score(row: dict) -> tuple[float, dict]:
    """
    Score how likely this snapshot is a mass move (not ransomware).
    Returns (score 0-1, breakdown dict).
    Score > 0.5 → classify as mass move (false positive).
    Score < 0.5 → forward to Stage 2B.
    """
    delete_add_ratio   = safe_float(row["delete_add_ratio"])
    avg_entropy        = safe_float(row["avg_entropy"])
    high_entropy_count = safe_int(row["high_entropy_count"])
    bulk_rename_flag   = safe_int(row["bulk_rename_flag"])
    locked_file_count  = safe_int(row["locked_file_count"])
    files_added        = safe_int(row["files_added"])
    files_deleted      = safe_int(row["files_deleted"])

    scores = {}

    # Signal 1: delete/add ratio close to 1.0
    # Perfect move: ratio = 1.0 exactly
    # Ransomware: many deletes, many new .locked adds → ratio varies but entropy spikes
    ratio_dist = abs(delete_add_ratio - 1.0)
    scores["ratio_score"] = max(0.0, 1.0 - ratio_dist * 2)

    # Signal 2: entropy stays normal
    # Normal files: avg entropy 3.5 - 6.5
    # Encrypted:    avg entropy 7.5 - 8.0
    if avg_entropy < 6.5:
        scores["entropy_score"] = 1.0
    elif avg_entropy < 7.0:
        scores["entropy_score"] = 0.5
    else:
        scores["entropy_score"] = 0.0

    # Signal 3: no high entropy files
    if high_entropy_count == 0:
        scores["high_entropy_score"] = 1.0
    elif high_entropy_count <= MAX_HIGH_ENTROPY:
        scores["high_entropy_score"] = 0.5
    else:
        scores["high_entropy_score"] = 0.0

    # Signal 4: no .locked renames
    if bulk_rename_flag == 0 and locked_file_count == 0:
        scores["rename_score"] = 1.0
    elif locked_file_count < 10:
        scores["rename_score"] = 0.5
    else:
        scores["rename_score"] = 0.0

    # Signal 5: significant number of files involved (must be "mass")
    n_moved = min(files_added, files_deleted)
    scores["mass_score"] = 1.0 if n_moved >= MIN_MOVED_FILES else 0.3

    # Weighted final score
    weights = {
        "ratio_score":        0.25,
        "entropy_score":      0.30,
        "high_entropy_score": 0.25,
        "rename_score":       0.15,
        "mass_score":         0.05,
    }
    final_score = sum(scores[k] * weights[k] for k in weights)

    return round(final_score, 4), scores

# ── Main Filter ───────────────────────────────────────────────────────────────

def run_stage2a(flagged_ids: set = None) -> list:
    print("="*55)
    print("STAGE 2A — MASS MOVE FALSE POSITIVE FILTER")
    print("="*55)

    if flagged_ids is None:
        flagged_ids = load_stage1_flags()

    features = load_features()

    print(f"Snapshots flagged by Stage 1 : {len(flagged_ids)}")
    print(f"Snapshot IDs                 : {sorted(flagged_ids)}\n")

    results = []

    for sid in sorted(flagged_ids):
        if sid not in features:
            continue

        row = features[sid]
        score, breakdown = compute_mass_move_score(row)

        is_mass_move   = score > 0.5
        forward_to_2b  = not is_mass_move
        decision       = "MASS_MOVE (skip 2B)" if is_mass_move else "SUSPICIOUS → forward to 2B"

        result = {
            "snapshot_id":       sid,
            "event_type":        row["event_type"],
            "label":             row["label"],
            "mass_move_score":   score,
            "is_mass_move":      int(is_mass_move),
            "forward_to_2b":     int(forward_to_2b),
            "ratio_score":       round(breakdown["ratio_score"], 4),
            "entropy_score":     round(breakdown["entropy_score"], 4),
            "high_entropy_score":round(breakdown["high_entropy_score"], 4),
            "rename_score":      round(breakdown["rename_score"], 4),
            "delete_add_ratio":  row["delete_add_ratio"],
            "avg_entropy":       row["avg_entropy"],
            "high_entropy_count":row["high_entropy_count"],
            "bulk_rename_flag":  row["bulk_rename_flag"],
        }
        results.append(result)

        print(f"  Snapshot [{sid:03d}] | event={row['event_type']:<12} | "
              f"mass_move_score={score:.3f} | {decision}")
        print(f"    ratio={breakdown['ratio_score']:.2f}  "
              f"entropy={breakdown['entropy_score']:.2f}  "
              f"high_ent={breakdown['high_entropy_score']:.2f}  "
              f"rename={breakdown['rename_score']:.2f}")

    # Evaluation
    print(f"\n{'='*55}")
    print("STAGE 2A — EVALUATION")
    print(f"{'='*55}")

    mass_move_detected = [r for r in results if r["is_mass_move"]]
    forwarded          = [r for r in results if r["forward_to_2b"]]

    # Correct if: actual mass_move → detected as mass_move
    #             actual ransomware → forwarded to 2B
    correct = sum(
        1 for r in results
        if (r["event_type"] == "mass_move" and r["is_mass_move"]) or
           (r["event_type"] != "mass_move" and r["forward_to_2b"])
    )

    print(f"  Total flagged       : {len(results)}")
    print(f"  Classified mass_move: {len(mass_move_detected)}")
    print(f"  Forwarded to 2B     : {len(forwarded)}")
    print(f"  Correct decisions   : {correct}/{len(results)}")
    print(f"{'='*55}")

    # Save
    if results:
        with open(RESULTS_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved → {RESULTS_PATH}")

    return results

if __name__ == "__main__":
    # Run Stage 1 first to get flags, then run 2A
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, "models"))
    from stage1_anomaly import run_stage1

    stage1_results = run_stage1()
    flagged = {r["snapshot_id"] for r in stage1_results if r["stage2_trigger"]}
    run_stage2a(flagged)