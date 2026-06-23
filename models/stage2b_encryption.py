"""
stage2b_encryption.py
Stage 2B: Encryption Detection using a PyTorch DNN.

Takes snapshots forwarded from Stage 2A and computes
an encryption_probability (0.0 → 1.0).

prob > 0.5 → RANSOMWARE CONFIRMED
prob < 0.5 → False alarm, resume monitoring

Since we have limited snapshots, we train on ALL labeled data
(not just Stage 1 flagged) so the DNN learns the full
clean vs ransomware signal. At inference time it scores
whatever Stage 2A forwards.

Output:
  models/saved/stage2b_model.pt      → trained DNN weights
  data/dataset/stage2b_results.csv   → encryption probabilities
"""

import os
import csv
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.dirname(__file__))
FEATURES_PATH   = os.path.join(BASE_DIR, "data", "dataset", "features_stage2.csv")
RESULTS_PATH    = os.path.join(BASE_DIR, "data", "dataset", "stage2b_results.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "saved", "stage2b_model.pt")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "saved", "stage2b_scaler.pkl")

# DNN hyperparameters
HIDDEN_DIMS     = [64, 32, 16]
DROPOUT         = 0.3
LEARNING_RATE   = 1e-3
EPOCHS          = 150
BATCH_SIZE      = 16
RANDOM_SEED     = 42

# Features for Stage 2B — pure encryption signals
STAGE2_FEATURES = [
    "avg_entropy",
    "entropy_delta",
    "entropy_variance",
    "high_entropy_ratio",   # keep ratio, not raw count
    "rolling_avg_entropy",
    "rolling_std_entropy",
    "entropy_zscore",
    "avg_size_delta",
    "delete_add_ratio",
    "churn_rate",
    # REMOVED: locked_file_count, bulk_rename_flag, high_entropy_count
    # These directly encode the .locked extension = label leakage
]

# ── Reproducibility ───────────────────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── DNN Architecture ──────────────────────────────────────────────────────────

class EncryptionDetector(nn.Module):
    """
    3-layer MLP with BatchNorm + Dropout.
    Input:  len(STAGE2_FEATURES) features
    Output: single sigmoid probability (encryption likelihood)
    """
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_features() -> tuple:
    rows = []
    with open(FEATURES_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    X, y, meta = [], [], []
    for row in rows:
        vec   = [float(row[f]) for f in STAGE2_FEATURES]
        # Binary label: 1 = ransomware, 0 = clean or mass_move
        label = 1 if row["event_type"] == "ransomware" else 0
        X.append(vec)
        y.append(label)
        meta.append({
            "snapshot_id": int(row["snapshot_id"]),
            "event_type":  row["event_type"],
            "true_label":  label,
        })

    return X, y, meta

def train_test_split(X, y, meta, test_ratio=0.25):
    """
    Stratified split: keep class balance in train/test.
    """
    pos_idx = [i for i, yi in enumerate(y) if yi == 1]
    neg_idx = [i for i, yi in enumerate(y) if yi == 0]

    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    n_pos_test = max(1, int(len(pos_idx) * test_ratio))
    n_neg_test = max(1, int(len(neg_idx) * test_ratio))

    test_idx  = pos_idx[:n_pos_test] + neg_idx[:n_neg_test]
    train_idx = pos_idx[n_pos_test:] + neg_idx[n_neg_test:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
    meta_test = [meta[i] for i in test_idx]

    return X_train, y_train, X_test, y_test, meta_test

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, input_dim: int):
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train,  dtype=torch.float32)

    dataset    = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = EncryptionDetector(input_dim, HIDDEN_DIMS, DROPOUT)
    # Weighted loss: ransomware is rare, penalise missing it more
    pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train) + 1e-6)])
    criterion  = nn.BCELoss()
    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    print(f"\nTraining DNN: {input_dim} → {HIDDEN_DIMS} → 1")
    print(f"Epochs={EPOCHS}, LR={LEARNING_RATE}, Batch={BATCH_SIZE}\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 30 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch [{epoch+1:03d}/{EPOCHS}]  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return model, scaler

# ── Inference ─────────────────────────────────────────────────────────────────

def predict_all(model, scaler, X_all, y_all, meta_all) -> list:
    """Run all snapshots through DNN, return full results."""
    X_scaled = scaler.transform(X_all)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        probs = model(X_tensor).numpy()

    results = []
    for i, (prob, true_label, m) in enumerate(zip(probs, y_all, meta_all)):
        pred_label = 1 if prob > 0.5 else 0
        results.append({
            "snapshot_id":          m["snapshot_id"],
            "event_type":           m["event_type"],
            "true_label":           true_label,
            "encryption_prob":      round(float(prob), 4),
            "pred_label":           pred_label,
            "ransomware_confirmed": int(pred_label == 1),
            "correct":              int(pred_label == true_label),
        })

    return results

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results: list, X_test, y_test, meta_test, model, scaler):
    print(f"\n{'='*55}")
    print("STAGE 2B — FULL DATASET RESULTS")
    print(f"{'='*55}")
    print(f"{'Snapshot':<10} {'Event':<14} {'Prob':<8} {'Pred':<12} {'Correct'}")
    print("-"*55)
    for r in results:
        marker = "✓" if r["correct"] else "✗"
        pred_str = "RANSOMWARE" if r["ransomware_confirmed"] else "clean"
        print(
            f"  [{r['snapshot_id']:03d}]  "
            f"{r['event_type']:<14} "
            f"{r['encryption_prob']:<8.4f} "
            f"{pred_str:<12} "
            f"{marker}"
        )

    # Held-out test set metrics
    X_sc   = scaler.transform(X_test)
    X_t    = torch.tensor(X_sc, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs_test = model(X_t).numpy()

    preds_test = [1 if p > 0.5 else 0 for p in probs_test]

    precision = precision_score(y_test, preds_test, zero_division=0)
    recall    = recall_score(y_test, preds_test, zero_division=0)
    f1        = f1_score(y_test, preds_test, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs_test)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_test, preds_test)

    print(f"\n{'='*55}")
    print("STAGE 2B — TEST SET EVALUATION")
    print(f"{'='*55}")
    print(f"  Test samples      : {len(y_test)}")
    print(f"  Precision         : {precision:.3f}")
    print(f"  Recall            : {recall:.3f}")
    print(f"  F1 Score          : {f1:.3f}")
    print(f"  ROC-AUC           : {auc:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"{'='*55}")

# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(results: list):
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {RESULTS_PATH}")

def save_model(model, scaler):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_stage2b() -> list:
    print("="*55)
    print("STAGE 2B — ENCRYPTION DETECTION (PyTorch DNN)")
    print("="*55)

    X, y, meta = load_features()
    print(f"Total snapshots   : {len(X)}")
    print(f"Ransomware        : {sum(y)}")
    print(f"Clean/mass_move   : {len(y) - sum(y)}")
    print(f"Features          : {len(STAGE2_FEATURES)}")

    X_train, y_train, X_test, y_test, meta_test = train_test_split(X, y, meta)
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    model, scaler = train_model(X_train, y_train, input_dim=len(STAGE2_FEATURES))

    # Score all snapshots (full pipeline needs scores for every snapshot)
    results = predict_all(model, scaler, X, y, meta)
    evaluate(results, X_test, y_test, meta_test, model, scaler)

    save_results(results)
    save_model(model, scaler)

    return results

if __name__ == "__main__":
    run_stage2b()