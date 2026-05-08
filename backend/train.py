"""
train.py
--------
Training pipeline for PipeGuard AI.

Modes:
    python train.py                   # hybrid: real SCADA + calibrated synthetic
    python train.py --mode synthetic  # synthetic only (no real data needed)
    python train.py --mode real       # real 1000 rows only (fast, less accurate)
"""

import argparse
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    LR_STEP_SIZE, LR_GAMMA, GRAD_CLIP_NORM, TRAIN_VAL_SPLIT,
    SIGMA_MULTIPLIER, N_NORMAL_SAMPLES, MODEL_PATH, THRESHOLD_PATH,
    HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, FEATURES, SCADA_CSV_PATH,
)
from data_simulation import build_dataset, save_dataset
from preprocessing   import MinMaxScaler, make_sequences, train_val_split
from model           import build_model
from utils           import get_logger, threshold_report, log_epoch

logging.basicConfig(level=logging.INFO)
logger = get_logger("train")


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss, total_samples = 0.0, 0
    for (batch,) in loader:
        batch = batch.to(device)
        loss  = criterion(model(batch), batch)
        if training and optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        total_loss    += loss.item() * len(batch)
        total_samples += len(batch)
    return total_loss / total_samples


def compute_errors(model, seqs, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(seqs)), batch_size=256):
            errors.extend(model.reconstruction_error(batch.to(device)).cpu().numpy().tolist())
    return np.array(errors, dtype=np.float32)


def train(mode: str = "hybrid") -> dict:
    print()
    print("═" * 62)
    print("  ⬡  PipeGuard AI — Model Training")
    mode_labels = {
        "hybrid":    "Hybrid  (694 real SCADA + 5,000 synthetic)",
        "synthetic": "Synthetic only (5,000 calibrated samples)",
        "real":      "Real SCADA only (694 normal rows)",
    }
    print(f"     Mode : {mode_labels.get(mode, mode)}")
    print("═" * 62)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n  [1 / 5]  Loading data …")

    if mode == "hybrid":
        if not SCADA_CSV_PATH.exists():
            print(f"\n  ⚠️  SCADA CSV not found — switching to synthetic mode")
            mode = "synthetic"

    if mode == "hybrid":
        from data_loader import build_hybrid_dataset
        normal_df, full_df = build_hybrid_dataset(n_synthetic=N_NORMAL_SAMPLES)
    elif mode == "real":
        from data_loader import load_real_data
        normal_df, full_df = load_real_data()
    else:
        normal_df, full_df = build_dataset(n_normal=N_NORMAL_SAMPLES)
        save_dataset(full_df)

    print(f"          Normal rows : {len(normal_df):>7,}")
    print(f"          Total rows  : {len(full_df):>7,}")
    print(f"          Fault rows  : {int(full_df['is_anomaly'].sum()):>7,}  "
          f"({100 * full_df['is_anomaly'].mean():.1f}%)")

    if "event_label" in full_df.columns:
        print(f"\n          Event distribution:")
        for label, cnt in full_df["event_label"].value_counts().items():
            bar = "█" * int(35 * cnt / len(full_df))
            print(f"            {label:<15} {cnt:>6,}  {bar}")

    # ── 2. Scaler ─────────────────────────────────────────────────
    print("\n  [2 / 5]  Fitting MinMaxScaler on normal data …")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(normal_df, features=FEATURES)
    scaler.save()
    print(f"          {scaler}")

    # ── 3. Sequences ──────────────────────────────────────────────
    print(f"\n  [3 / 5]  Building sequences (window={SEQ_LEN}) …")
    seqs = make_sequences(scaled, seq_len=SEQ_LEN)
    print(f"          Shape : {seqs.shape}  →  {seqs.shape[0]:,} windows")

    train_seqs, val_seqs = train_val_split(seqs, split=TRAIN_VAL_SPLIT)
    print(f"          Train / Val : {len(train_seqs):,} / {len(val_seqs):,}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_seqs)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(val_seqs)),
                              batch_size=BATCH_SIZE)

    # ── 4. Train ──────────────────────────────────────────────────
    print(f"\n  [4 / 5]  Training ({EPOCHS} epochs, batch={BATCH_SIZE}) …")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"          Device     : {device}")

    arch  = "cnn_lstm" if not getattr(args, "lstm_only", False) else "lstm"
    model = build_model(architecture=arch, input_dim=len(FEATURES)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)

    print(f"          Parameters : {model.count_parameters():,}\n")

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        t = run_epoch(model, train_loader, criterion, optimizer, device, True)
        v = run_epoch(model, val_loader,   criterion, None,      device, False)
        train_losses.append(t)
        val_losses.append(v)
        scheduler.step()
        if epoch % 10 == 0 or epoch in (1, EPOCHS):
            log_epoch(epoch, EPOCHS, t, v)

    # ── 5. Threshold ──────────────────────────────────────────────
    print(f"\n  [5 / 5]  Computing threshold (σ={SIGMA_MULTIPLIER}) …")
    errors = compute_errors(model, seqs, device)
    report = threshold_report(errors, sigma=SIGMA_MULTIPLIER)

    print(f"          Error mean : {report['mean']:.6f}")
    print(f"          Error std  : {report['std']:.6f}")
    print(f"          Threshold  : {report['threshold']:.6f}")

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({
            "threshold":    report["threshold"],
            "seq_len":      SEQ_LEN,
            "features":     FEATURES,
            "n_features":   len(FEATURES),
            "training_mode": mode,
            "architecture": arch,
            "stats":        report,
        }, f, indent=2)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print()
    print("═" * 62)
    print("  ✅  Training complete!")
    print(f"     Model     → {MODEL_PATH}")
    print(f"     Threshold → {report['threshold']:.6f}")
    print("═" * 62)
    print()

    return {
        "threshold":    report["threshold"],
        "train_losses": train_losses,
        "val_losses":   val_losses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PipeGuard AI")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "synthetic", "real"],
        default="hybrid",
        help=(
            "hybrid    = real SCADA (694 rows) + 5000 synthetic (RECOMMENDED)\n"
            "synthetic = 5000 synthetic rows only\n"
            "real      = 694 real rows only (too few, not recommended)"
        ),
    )
    args    = parser.parse_args()
    results = train(mode=args.mode)
    print(f"Final train loss : {results['train_losses'][-1]:.6f}")
    print(f"Final val   loss : {results['val_losses'][-1]:.6f}")
    print(f"Threshold        : {results['threshold']:.6f}")
