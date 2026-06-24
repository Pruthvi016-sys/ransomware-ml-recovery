"""
run_pipeline.py
Single entry point — runs the full ransomware recovery ML pipeline end to end.

Steps:
  1. Data Generation   — simulator + event injector + FMD generator
  2. Feature Engineering
  3. Stage 1           — Isolation Forest anomaly detection
  4. Stage 2A          — Mass move false positive filter
  5. Stage 2B          — PyTorch DNN encryption detection
  6. Stage 3A          — Blast radius analysis
  7. Stage 3B          — Recovery point identification
  8. Evaluation        — Metrics + scorecard

Then prints instructions to launch the Streamlit dashboard.

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-data   (skip data gen if already done)
"""

import os
import sys
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def step(n: int, total: int, name: str):
    print(f"\n[{n}/{total}] {name}")
    print("-" * 50)

def success(msg: str):
    print(f"  ✓ {msg}")

def elapsed(start: float) -> str:
    secs = time.time() - start
    return f"{secs:.1f}s"

# ── Pipeline Steps ────────────────────────────────────────────────────────────

def run_data_generation():
    from data.simulator      import run_simulation
    from data.event_injector import run_injection
    from data.fmd_generator  import run_fmd_generation

    t = time.time()
    run_simulation()
    success(f"Snapshots generated ({elapsed(t)})")

    t = time.time()
    run_injection()
    success(f"Events injected ({elapsed(t)})")

    t = time.time()
    run_fmd_generation()
    success(f"FMD files generated ({elapsed(t)})")

def run_features():
    from features.feature_engineering import run_feature_engineering
    t = time.time()
    run_feature_engineering()
    success(f"Features engineered ({elapsed(t)})")

def run_stage1():
    from models.stage1_anomaly import run_stage1 as _run
    t = time.time()
    results = _run()
    success(f"Stage 1 complete ({elapsed(t)})")
    return results

def run_stage2a(stage1_results):
    from models.stage2a_mass_move import run_stage2a as _run
    t = time.time()
    flagged = {r["snapshot_id"] for r in stage1_results if r["stage2_trigger"]}
    results = _run(flagged)
    success(f"Stage 2A complete ({elapsed(t)})")
    return results

def run_stage2b():
    from models.stage2b_encryption import run_stage2b as _run
    t = time.time()
    results = _run()
    success(f"Stage 2B complete ({elapsed(t)})")
    return results

def run_blast_radius():
    from analysis.blast_radius import run_blast_radius as _run
    t = time.time()
    _run()
    success(f"Blast radius analysis complete ({elapsed(t)})")

def run_recovery():
    from analysis.recovery_point import run_recovery_point as _run
    t = time.time()
    clean_id, mttr, diff = _run()
    success(f"Recovery point identified: Snapshot [{clean_id:03d}] ({elapsed(t)})")
    return clean_id, mttr, diff

def run_evaluation():
    import subprocess
    t = time.time()
    eval_path = os.path.join(BASE_DIR, "evaluation", "eval.py")
    result = subprocess.run(
        [sys.executable, eval_path],
        capture_output=False
    )
    if result.returncode == 0:
        success(f"Evaluation complete ({elapsed(t)})")
    else:
        print(f"  ⚠ Evaluation exited with code {result.returncode}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ransomware Recovery ML Pipeline"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip data generation (use existing snapshots/FMDs)"
    )
    args = parser.parse_args()

    total_steps = 7 if args.skip_data else 8
    pipeline_start = time.time()

    banner("RANSOMWARE RECOVERY — ML PIPELINE")
    print(f"  Base directory : {BASE_DIR}")
    print(f"  Skip data gen  : {args.skip_data}")

    step_n = 1

    # Step 1: Data generation
    if not args.skip_data:
        step(step_n, total_steps, "DATA GENERATION")
        run_data_generation()
        step_n += 1
    else:
        print("\n[--skip-data] Skipping data generation.")

    # Step 2: Feature engineering
    step(step_n, total_steps, "FEATURE ENGINEERING")
    run_features()
    step_n += 1

    # Step 3: Stage 1 — Anomaly detection
    step(step_n, total_steps, "STAGE 1 — ANOMALY DETECTION (Isolation Forest)")
    stage1_results = run_stage1()
    step_n += 1

    # Step 4: Stage 2A — Mass move filter
    step(step_n, total_steps, "STAGE 2A — MASS MOVE FALSE POSITIVE FILTER")
    run_stage2a(stage1_results)
    step_n += 1

    # Step 5: Stage 2B — Encryption detection
    step(step_n, total_steps, "STAGE 2B — ENCRYPTION DETECTION (PyTorch DNN)")
    run_stage2b()
    step_n += 1

    # Step 6: Blast radius
    step(step_n, total_steps, "STAGE 3A — BLAST RADIUS ANALYSIS")
    run_blast_radius()
    step_n += 1

    # Step 7: Recovery point
    step(step_n, total_steps, "STAGE 3B — RECOVERY POINT IDENTIFICATION")
    clean_id, mttr, diff = run_recovery()
    step_n += 1

    # Step 8: Evaluation
    step(step_n, total_steps, "EVALUATION — METRICS & SCORECARD")
    run_evaluation()

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    banner("PIPELINE COMPLETE")
    print(f"  Total time          : {total_time:.1f}s")
    print(f"  Clean recovery point: Snapshot [{clean_id:03d}]")
    print(f"  Data loss window    : {mttr['data_loss_hours']} hours")
    print(f"  MTTR with ML        : {mttr['ml_mttr_hours']} hours")
    print(f"  MTTR without ML     : {mttr['manual_mttr_hours']} hours")
    print(f"  Time saved          : {mttr['hours_saved']} hours "
          f"({mttr['mttr_reduction_pct']}% reduction)")
    print(f"\n  Output files in: data/dataset/")
    print(f"    stage1_results.csv")
    print(f"    stage2a_results.csv")
    print(f"    stage2b_results.csv")
    print(f"    blast_radius.csv")
    print(f"    affected_files.csv")
    print(f"    recovery_report.csv")
    print(f"    recovery_diff.csv")
    print(f"    evaluation_scorecard.csv")
    print(f"    evaluation_report.json")

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  Launch dashboard:                               │
  │    streamlit run dashboard/app.py                │
  └─────────────────────────────────────────────────┘
""")

if __name__ == "__main__":
    main()
