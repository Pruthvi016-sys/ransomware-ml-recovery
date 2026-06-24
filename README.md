# Ransomware Recovery — ML Pipeline

An end-to-end machine learning pipeline for ransomware detection and recovery, inspired by **Rubrik Polaris Radar**. Detects ransomware attacks on filesystem snapshot timelines, identifies blast radius, and recommends an optimal clean recovery point — reducing MTTR by ~89%.

---

## Architecture

```
Snapshot Timeline
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Anomaly Detection │  ← Isolation Forest (unsupervised)
│  Filesystem behavior signals│    trained on clean snapshots only
└──────────────┬──────────────┘
               │ SUSPICIOUS
               ▼
┌─────────────────────────────┐
│  Stage 2A: Mass Move Filter │  ← Rule-based scoring
│  False positive suppression │    delete/add ratio + entropy check
└──────────────┬──────────────┘
               │ NOT a mass move
               ▼
┌─────────────────────────────┐
│  Stage 2B: Encryption DNN   │  ← PyTorch MLP (3-layer)
│  Encryption probability     │    trained on entropy features
└──────────────┬──────────────┘
               │ RANSOMWARE CONFIRMED
               ▼
┌─────────────────────────────┐
│  Stage 3A: Blast Radius     │  ← Per-directory impact analysis
│  Affected files + severity  │    propagation trace
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Stage 3B: Recovery Point   │  ← Timeline walkback
│  Clean snapshot + MTTR      │    data loss estimation
└─────────────────────────────┘
               │
               ▼
   Streamlit Dashboard (4 tabs)
```

---

## Results

| Stage | Model | F1 | ROC-AUC |
|---|---|---|---|
| Stage 1 — Anomaly Detection | Isolation Forest | 0.29 | 0.60 |
| Stage 2B — Encryption Detection | PyTorch DNN | 0.80 | 0.92 |
| End-to-End Pipeline | Combined | 0.80 | 0.92 |

**Business Impact:**
- MTTR with ML: **~8 hours**
- MTTR without ML (manual): **72 hours**
- MTTR reduction: **~89%**

---

## Project Structure

```
ransomware-recovery-ml/
│
├── run_pipeline.py             ← single entry point (run this)
│
├── data/
│   ├── simulator.py            ← generates snapshot timeline
│   ├── event_injector.py       ← injects ransomware + mass move events
│   ├── fmd_generator.py        ← computes filesystem metadata diffs
│   └── dataset/                ← generated CSVs (auto-created)
│
├── features/
│   └── feature_engineering.py  ← behavior + entropy + temporal features
│
├── models/
│   ├── stage1_anomaly.py        ← Isolation Forest anomaly detector
│   ├── stage2a_mass_move.py     ← mass move false positive filter
│   ├── stage2b_encryption.py    ← PyTorch DNN encryption detector
│   └── saved/                   ← trained model weights (auto-created)
│
├── analysis/
│   ├── blast_radius.py          ← affected files, directory heatmap
│   └── recovery_point.py        ← clean snapshot ID + MTTR estimate
│
├── dashboard/
│   └── app.py                   ← Streamlit 4-tab dashboard
│
├── evaluation/
│   └── eval.py                  ← precision, recall, F1, ROC-AUC
│
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/ransomware-recovery-ml
cd ransomware-recovery-ml

pip install -r requirements.txt
```

**requirements.txt:**
```
pandas
numpy
scikit-learn
torch
streamlit
plotly
```

---

## Usage

### Run full pipeline (one command):
```bash
python run_pipeline.py
```

### Skip data generation (if already run once):
```bash
python run_pipeline.py --skip-data
```

### Launch dashboard:
```bash
streamlit run dashboard/app.py
```

### Run individual stages:
```bash
python data/simulator.py
python data/event_injector.py
python data/fmd_generator.py
python features/feature_engineering.py
python models/stage1_anomaly.py
python models/stage2a_mass_move.py
python models/stage2b_encryption.py
python analysis/blast_radius.py
python analysis/recovery_point.py
python evaluation/eval.py
```

---

## How It Works

### Data Layer
The simulator generates 50 filesystem snapshots over time, each containing file metadata (path, size, entropy, ACL, UID, GID). Two events are injected:
- **Snapshot 20** — mass file move (false positive case): 40% of files relocated, entropy unchanged
- **Snapshot 35** — ransomware attack: 75% of files renamed to `.locked` with entropy spiked to 7.5–8.0, 10% deleted, ransom notes added

Filesystem Metadata Diff (FMD) files are generated between consecutive snapshots — mirroring Rubrik CDM's actual FMD pipeline.

### Feature Engineering
Per-snapshot features extracted from FMDs:
- **Behavior features** (Stage 1): `files_added`, `files_deleted`, `churn_rate`, `delete_add_ratio`, `bulk_rename_flag`, rolling averages, spikes
- **Entropy features** (Stage 2B): `avg_entropy`, `entropy_delta`, `entropy_variance`, `high_entropy_count`, `entropy_zscore`, `locked_file_count`

### Stage 1 — Isolation Forest
Trained only on clean snapshots. Flags statistical deviations in filesystem behavior without requiring labeled attack data. Deliberately conservative — over-flags to avoid missing real attacks.

### Stage 2A — Mass Move Filter
Scores each flagged snapshot on 5 signals (delete/add ratio, entropy level, high entropy count, rename pattern, move volume). Snapshots scoring >0.5 are classified as mass moves and dropped. Prevents false positives from routine file reorganizations.

### Stage 2B — PyTorch DNN
3-layer MLP (64→32→16→1) with BatchNorm and Dropout. Trained on entropy features to compute encryption probability. Threshold at 0.5: above = ransomware confirmed.

### Blast Radius Analysis
Compares attack snapshot against previous clean snapshot. Identifies encrypted/deleted files per directory, computes severity (LOW/MEDIUM/CRITICAL), and produces an affected file list with recovery flags.

### Recovery Point Identification
Walks back through snapshot timeline to find the last snapshot where both anomaly score and encryption probability are below thresholds. Computes data loss window and MTTR estimate.

---

## Dashboard

Four-tab Streamlit UI:

| Tab | Content |
|---|---|
| Timeline View | Anomaly scores + encryption probabilities across all snapshots |
| Blast Radius | Directory heatmap, file type breakdown, affected files table |
| Recovery Recommendation | Clean snapshot card, MTTR comparison, recovery checklist |
| Event Log | Full snapshot history, filterable, exportable CSV |

---

## Inspiration

This project replicates and extends the core architecture described in Rubrik's [Applying Machine Learning Models to Ransomware Recovery](https://www.rubrik.com/blog/technology/19/5/machine-learning-models-ransomware-recovery), specifically the two-stage anomaly + encryption detection pipeline used in **Rubrik Polaris Radar**.

Extensions beyond the paper:
- Mass move false positive filter (inspired by Rubrik intern work on CFD resolution)
- Blast radius analysis with per-directory severity scoring
- MTTR estimation with data loss quantification
- Unified investigation dashboard

---

## Author

**M M Pruthvi Raj** — IIT Madras, AI & Data Science (DA24B016)
