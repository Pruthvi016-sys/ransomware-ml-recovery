"""
evaluation/eval.py
Final evaluation script for the Ransomware Recovery ML pipeline.

Computes Accuracy / Precision / Recall / F1 / ROC-AUC for:
  Stage 1   — Isolation Forest anomaly detector
  Stage 2A  — Mass-move false-positive filter
  Stage 2B  — DNN encryption confirmation
  End-to-End — final pipeline verdict (Stage1 -> Stage2A -> Stage2B)

Also pulls in blast_radius.csv and recovery_report.csv to print the
business-impact summary (MTTR reduction, blast radius) alongside the
model metrics, then saves:
  data/dataset/evaluation_report.json
  data/dataset/evaluation_scorecard.csv

Run with:
    python evaluation/eval.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score
)

# ── Path Setup ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset")


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


def load_csv(filename):
    path = data_path(filename)
    if not os.path.exists(path):
        print(f"  [missing] {filename}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    return df


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ── Metric helper ─────────────────────────────────────────────────────────

def safe_metrics(y_true, y_pred, y_score=None, label=""):
    """Computes classification metrics without crashing on edge cases
    (e.g. only one class present in a slice)."""
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(int).reset_index(drop=True)

    metrics = {}
    try:
        metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
        metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
        metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
        metrics["f1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"], metrics["fp"] = int(tn), int(fp)
        metrics["fn"], metrics["tp"] = int(fn), int(tp)
    except Exception as e:
        metrics["error"] = str(e)

    if y_score is not None:
        try:
            y_score = pd.Series(y_score).reset_index(drop=True)
            if y_true.nunique() > 1:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
            else:
                metrics["roc_auc"] = None
        except Exception:
            metrics["roc_auc"] = None

    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:14s}: {v}")
    if "confusion_matrix" in metrics:
        print(f"  confusion_matrix [[TN FP] [FN TP]] = {metrics['confusion_matrix']}")
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    report = {}

    section("Loading pipeline outputs")
    s1_df    = load_csv("stage1_results.csv")
    s2a_df   = load_csv("stage2a_results.csv")
    s2b_df   = load_csv("stage2b_results.csv")
    blast_df = load_csv("blast_radius.csv")
    recov_df = load_csv("recovery_report.csv")

    if s1_df.empty:
        print("\nstage1_results.csv not found — run stage1_anomaly.py first.")
        sys.exit(1)

    # Ground truth derived from event_type (clean / mass_move / ransomware)
    s1_df["y_true_anomaly"]    = (s1_df["event_type"] != "clean").astype(int)
    s1_df["y_true_ransomware"] = (s1_df["event_type"] == "ransomware").astype(int)

    # ── Stage 1 ────────────────────────────────────────────────────────────
    section("STAGE 1 — Isolation Forest (Anomaly Detection)")
    if "anomaly_flag" in s1_df.columns:
        y_score = (-s1_df["anomaly_score"]) if "anomaly_score" in s1_df.columns else None
        report["stage1"] = safe_metrics(
            s1_df["y_true_anomaly"], s1_df["anomaly_flag"], y_score,
            label="Stage 1 (flag anything abnormal vs. clean)"
        )
    else:
        print("  anomaly_flag column missing in stage1_results.csv — skipping.")

    # ── Stage 2A ───────────────────────────────────────────────────────────
    section("STAGE 2A — Mass-Move False-Positive Filter")
    if not s2a_df.empty:
        print(f"  columns found: {list(s2a_df.columns)}")
        decision_col = next(
            (c for c in ["is_mass_move", "decision", "mass_move_flag", "filtered",
                          "predicted_mass_move", "mass_move_pred"]
             if c in s2a_df.columns),
            None
        )
        score_col = next(
            (c for c in ["mass_move_score", "mass_move_prob", "score"]
             if c in s2a_df.columns),
            None
        )
        if decision_col and "event_type" in s2a_df.columns:
            # stage2a_results.csv already carries its own event_type — use it
            # directly rather than merging (merging would collide on the
            # column name and silently produce event_type_x/event_type_y).
            y_true = (s2a_df["event_type"] == "mass_move").astype(int)
            y_pred = s2a_df[decision_col].astype(int)
            y_score = s2a_df[score_col] if score_col else None
            report["stage2a"] = safe_metrics(
                y_true, y_pred, y_score,
                label=f"Stage 2A (predict mass_move using '{decision_col}')"
            )
        elif decision_col and "snapshot_id" in s2a_df.columns:
            merged = s2a_df.merge(
                s1_df[["snapshot_id", "event_type"]],
                on="snapshot_id", how="left", suffixes=("", "_truth")
            )
            truth_col = "event_type_truth" if "event_type_truth" in merged.columns else "event_type"
            y_true = (merged[truth_col] == "mass_move").astype(int)
            y_pred = merged[decision_col].astype(int)
            y_score = merged[score_col] if score_col else None
            report["stage2a"] = safe_metrics(
                y_true, y_pred, y_score,
                label=f"Stage 2A (predict mass_move using '{decision_col}')"
            )
        else:
            print("  Could not auto-detect a decision column + snapshot_id.")
            print("  Tell me the real column names and I'll wire this up exactly.")
            report["stage2a"] = {"raw_row_count": len(s2a_df),
                                  "columns": list(s2a_df.columns)}
    else:
        print("  stage2a_results.csv not found — skipping.")

    # ── Stage 2B ───────────────────────────────────────────────────────────
    section("STAGE 2B — DNN Encryption Detector")
    if not s2b_df.empty and "ransomware_confirmed" in s2b_df.columns:
        if "true_label" in s2b_df.columns:
            y_true = s2b_df["true_label"].astype(int)
        elif "event_type" in s2b_df.columns:
            y_true = (s2b_df["event_type"] == "ransomware").astype(int)
        elif "snapshot_id" in s2b_df.columns:
            merged = s2b_df.merge(
                s1_df[["snapshot_id", "y_true_ransomware"]], on="snapshot_id", how="left"
            )
            y_true = merged["y_true_ransomware"]
        else:
            y_true = None

        if y_true is not None:
            y_score = s2b_df["encryption_prob"] if "encryption_prob" in s2b_df.columns else None
            report["stage2b"] = safe_metrics(
                y_true, s2b_df["ransomware_confirmed"], y_score,
                label="Stage 2B (confirm ransomware via encryption signal)"
            )
        else:
            print("  Could not determine ground truth for stage2b_results.csv — skipping.")
    else:
        print("  stage2b_results.csv not found or missing ransomware_confirmed — skipping.")

    # ── End-to-end ─────────────────────────────────────────────────────────
    section("END-TO-END PIPELINE (final ransomware verdict)")
    if not s2b_df.empty and "ransomware_confirmed" in s2b_df.columns:
        cols = ["snapshot_id", "ransomware_confirmed"]
        if "encryption_prob" in s2b_df.columns:
            cols.append("encryption_prob")
        full = s1_df.merge(s2b_df[cols], on="snapshot_id", how="left")
        full["ransomware_confirmed"] = full["ransomware_confirmed"].fillna(0).astype(int)
        y_score = full["encryption_prob"] if "encryption_prob" in full.columns else None
        report["end_to_end"] = safe_metrics(
            full["y_true_ransomware"], full["ransomware_confirmed"], y_score,
            label="End-to-end (Stage1 -> Stage2A -> Stage2B verdict)"
        )
    else:
        print("  Cannot assemble end-to-end verdict without Stage 2B output.")

    # ── Blast radius summary ────────────────────────────────────────────────
    section("BLAST RADIUS SUMMARY")
    if not blast_df.empty:
        total_enc = int(blast_df["encrypted"].sum()) if "encrypted" in blast_df.columns else None
        total_del = int(blast_df["deleted"].sum()) if "deleted" in blast_df.columns else None
        print(f"  Directories analysed : {len(blast_df)}")
        print(f"  Files encrypted      : {total_enc}")
        print(f"  Files deleted        : {total_del}")
        report["blast_radius"] = {
            "directories": len(blast_df), "encrypted": total_enc, "deleted": total_del
        }
    else:
        print("  blast_radius.csv not found — skipping.")

    # ── Recovery / MTTR summary ─────────────────────────────────────────────
    section("RECOVERY / MTTR SUMMARY")
    if not recov_df.empty:
        r = recov_df.iloc[0].to_dict()
        for k in ["clean_recovery_snapshot", "attack_snapshot", "data_loss_hours",
                  "ml_mttr_hours", "manual_mttr_hours", "hours_saved", "mttr_reduction_pct"]:
            if k in r:
                print(f"  {k:25s}: {r[k]}")
        report["recovery"] = r
    else:
        print("  recovery_report.csv not found — skipping.")

    # ── Save report ──────────────────────────────────────────────────────────
    section("SAVING EVALUATION REPORT")
    out_json = data_path("evaluation_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved -> {out_json}")

    # ── Final scorecard ──────────────────────────────────────────────────────
    section("FINAL SCORECARD")
    rows = []
    for stage_name, key in [
        ("Stage 1 (Anomaly Detection)", "stage1"),
        ("Stage 2A (Mass-Move Filter)", "stage2a"),
        ("Stage 2B (Encryption DNN)",   "stage2b"),
        ("End-to-End Pipeline",         "end_to_end"),
    ]:
        m = report.get(key, {})
        rows.append([
            stage_name,
            m.get("accuracy", "-"), m.get("precision", "-"),
            m.get("recall", "-"), m.get("f1", "-"), m.get("roc_auc", "-"),
        ])

    scorecard = pd.DataFrame(
        rows, columns=["Stage", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    )
    print("\n" + scorecard.to_string(index=False))

    out_csv = data_path("evaluation_scorecard.csv")
    scorecard.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")


if __name__ == "__main__":
    main()