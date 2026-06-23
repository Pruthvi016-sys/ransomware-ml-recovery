"""
stage1_anomaly.py
Stage 1: Anomaly Detection using Isolation Forest.

Trained ONLY on clean snapshots — learns what normal looks like.
Flags any snapshot that deviates significantly as SUSPICIOUS.
Suspicious snapshots are forwarded to Stage 2A (mass move filter).

Output:
  models/saved/stage1_model.pkl       → trained Isolation Forest
  data/dataset/stage1_results.csv     → anomaly scores + predictions for all snapshots
"""

import os
import csv
import pickle
import math

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
FEATURES_PATH   = os.path.join(BASE_DIR, "data", "dataset", "features_stage1.csv")
RESULTS_PATH    = os.path.join(BASE_DIR, "data", "dataset", "stage1_results.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "saved", "stage1_model.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "saved", "stage1_scaler.pkl")

# Features fed into Isolation Forest
# Deliberately excludes entropy-heavy features — those are Stage 2's job
STAGE1_FEATURES = [
    "files_added",
    "files_deleted",
    "files_modified",
    "file_count_delta",
    "file_count_delta_ratio",
    "churn_rate",
    "delete_add_ratio",
    "ext_change_ratio",
    "bulk_rename_flag",
    "locked_file_count",
    "avg_size_delta",
    "rolling_avg_added",
    "rolling_avg_deleted",
    "added_spike",
    "deleted_spike",
    "high_entropy_count",
    "high_entropy_ratio",
    "avg_entropy",
    "entropy_zscore",
]

# Isolation Forest config
CONTAMINATION   = 0.05   # expect ~5% anomalies in training data
N_ESTIMATORS    = 200
RANDOM_STATE    = 42

# Anomaly threshold: IF returns -1 (anomaly) or 1 (normal)
# We also use raw scores for ranking

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_features() -> tuple[list, list, list, list]:
    """
    Returns:
      X_clean  — feature matrix for clean snapshots only (training)
      X_all    — feature matrix for all snapshots (inference)
      meta_all — list of (snapshot_id, event_type, label) for all
      feature_names
    """
    rows = []
    with open(FEATURES_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    X_clean = []
    X_all   = []
    meta    = []

    for row in rows:
        vec = [float(row[f]) for f in STAGE1_FEATURES]
        meta.append({
            "snapshot_id": int(row["snapshot_id"]),
            "event_type":  row["event_type"],
            "label":       int(row["label"]),
        })
        X_all.append(vec)
        if row["event_type"] == "clean":
            X_clean.append(vec)

    return X_clean, X_all, meta, STAGE1_FEATURES

# ── Training ──────────────────────────────────────────────────────────────────

def train(X_clean: list) -> tuple:
    """Train Isolation Forest on clean snapshots only."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler

# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, scaler, X_all: list, meta: list) -> list:
    """
    Run all snapshots through trained model.
    Returns list of result dicts.

    anomaly_score: raw IF score (more negative = more anomalous)
    anomaly_flag:  1 if anomalous, 0 if normal
    stage2_trigger: 1 if this snapshot should go to Stage 2
    """
    X_scaled = scaler.transform(X_all)

    # predict returns: 1 (normal) or -1 (anomaly)
    preds  = model.predict(X_scaled)

    # decision_function: higher = more normal, lower = more anomalous
    scores = model.decision_function(X_scaled)

    results = []
    for i, (pred, score) in enumerate(zip(preds, scores)):
        anomaly_flag   = 1 if pred == -1 else 0
        stage2_trigger = anomaly_flag  # pass suspicious ones to Stage 2

        results.append({
            "snapshot_id":    meta[i]["snapshot_id"],
            "event_type":     meta[i]["event_type"],
            "label":          meta[i]["label"],
            "anomaly_score":  round(float(score), 6),
            "anomaly_flag":   anomaly_flag,
            "stage2_trigger": stage2_trigger,
            "correct":        int(anomaly_flag == int(meta[i]["label"] != 0)),
        })

    return results

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results: list):
    total       = len(results)
    flagged     = sum(r["anomaly_flag"] for r in results)
    correct     = sum(r["correct"] for r in results)

    # True positives: ransomware/mass_move correctly flagged
    tp = sum(1 for r in results if r["anomaly_flag"] == 1 and r["label"] != 0)
    fp = sum(1 for r in results if r["anomaly_flag"] == 1 and r["label"] == 0)
    fn = sum(1 for r in results if r["anomaly_flag"] == 0 and r["label"] != 0)
    tn = sum(1 for r in results if r["anomaly_flag"] == 0 and r["label"] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'='*55}")
    print("STAGE 1 — EVALUATION")
    print(f"{'='*55}")
    print(f"  Total snapshots   : {total}")
    print(f"  Flagged anomalies : {flagged}")
    print(f"  True Positives    : {tp}  (ransomware/mass_move correctly flagged)")
    print(f"  False Positives   : {fp}  (clean incorrectly flagged)")
    print(f"  False Negatives   : {fn}  (attacks missed)")
    print(f"  True Negatives    : {tn}  (clean correctly passed)")
    print(f"  Precision         : {precision:.3f}")
    print(f"  Recall            : {recall:.3f}")
    print(f"  F1 Score          : {f1:.3f}")
    print(f"{'='*55}")

    print(f"\n{'Snapshot':<10} {'Event':<12} {'Score':<12} {'Flag':<6} {'Correct'}")
    print("-"*50)
    for r in results:
        marker = "✓" if r["correct"] else "✗"
        print(
            f"  [{r['snapshot_id']:03d}]  "
            f"{r['event_type']:<12} "
            f"{r['anomaly_score']:<12.4f} "
            f"{'ANOMALY' if r['anomaly_flag'] else 'normal':<9} "
            f"{marker}"
        )

# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(results: list):
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {RESULTS_PATH}")

def save_model(model, scaler):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"Model  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_stage1():
    print("="*55)
    print("STAGE 1 — ANOMALY DETECTION (Isolation Forest)")
    print("="*55)

    X_clean, X_all, meta, feature_names = load_features()
    print(f"Training on {len(X_clean)} clean snapshots")
    print(f"Inference on {len(X_all)} total snapshots")
    print(f"Features: {len(feature_names)}")

    print("\nTraining Isolation Forest...")
    model, scaler = train(X_clean)
    print("Done.")

    results = predict(model, scaler, X_all, meta)
    evaluate(results)
    save_results(results)
    save_model(model, scaler)

    # Return for pipeline chaining
    return results

if __name__ == "__main__":
    run_stage1()