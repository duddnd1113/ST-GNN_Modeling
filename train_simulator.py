"""
Training, validation, and test evaluation script for the Seoul PM2.5 ST-GNN.

Pipeline:
    1. Generate synthetic data.
    2. Build static graph and precompute all edge features.
    3. Normalise node features (min-max on training set).
    4. Create sliding-window datasets and DataLoaders.
    5. Train for 50 epochs (Adam, MSE loss); save best model.
    6. Evaluate on test set: report MAE, RMSE (original scale).
    7. Print node embedding h_i shape.

Usage:
    python train.py
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_generator import STATION_COORDS, SeoulPM25Dataset, generate_synthetic_data
from graph_builder import (
    build_static_graph,
    compute_all_dynamic_edge_features,
    get_full_edge_features,
)
from model import STGNNModel


# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

WINDOW        = 12          # lookback window (hours)
BATCH_SIZE    = 32
EPOCHS        = 50
LR            = 1e-3
THRESHOLD_KM  = 20.0
GAT_HIDDEN    = 64
GRU_HIDDEN    = 64
NUM_HEADS     = 4
CHECKPOINT    = "best_model.pt"

# Train / Val / Test split ratios (time-ordered, no shuffle)
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO = 0.15  (remainder)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def minmax_normalize(
    data: np.ndarray,
    feat_min: np.ndarray,
    feat_max: np.ndarray,
) -> np.ndarray:
    """Min-max scale features to [0, 1]."""
    return (data - feat_min) / (feat_max - feat_min + 1e-8)


def minmax_denormalize(
    data_norm: np.ndarray,
    feat_min: np.ndarray,
    feat_max: np.ndarray,
) -> np.ndarray:
    """Inverse of minmax_normalize."""
    return data_norm * (feat_max - feat_min + 1e-8) + feat_min


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: STGNNModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    pm25_min: float,
    pm25_max: float,
):
    """Compute MAE and RMSE in the original PM2.5 scale (µg/m³).

    Returns:
        mae: float
        rmse: float
        last_h_i: torch.Tensor [N, gru_hidden] — embeddings from the last batch.
    """
    model.eval()
    all_pred, all_true = [], []
    last_h_i = None

    with torch.no_grad():
        for node_w, edge_w, target in loader:
            node_w  = node_w.to(device)   # [B, T, N, 6]
            edge_w  = edge_w.to(device)   # [B, T, E, 5]
            target  = target.to(device)   # [B, N, 1]

            pred, h_i = model(node_w, edge_index, edge_w)   # [B,N,1], [B,N,64]
            last_h_i = h_i

            # Denormalise PM2.5 for metric computation
            pred_np   = pred.cpu().numpy() * (pm25_max - pm25_min) + pm25_min
            target_np = target.cpu().numpy() * (pm25_max - pm25_min) + pm25_min
            all_pred.append(pred_np)
            all_true.append(target_np)

    all_pred = np.concatenate(all_pred, axis=0)   # [total, N, 1]
    all_true = np.concatenate(all_true, axis=0)

    mae  = np.mean(np.abs(all_pred - all_true))
    rmse = math.sqrt(np.mean((all_pred - all_true) ** 2))
    return mae, rmse, last_h_i


# ──────────────────────────────────────────────────────────────────────────────
# Main training script
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Generate data ──────────────────────────────────────────────────
    print("Generating synthetic data...")
    raw_features = generate_synthetic_data(T=8760, seed=42)  # [T, N, 6]
    T, N, F = raw_features.shape
    print(f"  node_features shape: {raw_features.shape}")

    # ── 2. Build static graph + precompute full edge features ─────────────
    print("Building graph...")
    edge_index_np, static_attr_np, edge_bearings_np = build_static_graph(
        STATION_COORDS, threshold_km=THRESHOLD_KM
    )
    E = edge_index_np.shape[1]
    print(f"  Nodes: {N}, Edges (directed): {E}")

    dynamic_attr_np = compute_all_dynamic_edge_features(
        edge_index_np, raw_features, edge_bearings_np
    )                                                       # [T, E, 2]
    full_edge_np = get_full_edge_features(
        static_attr_np[None].repeat(T, axis=0),            # [T, E, 3]
        dynamic_attr_np                                     # [T, E, 2]
    )                                                       # [T, E, 5]
    print(f"  Edge features shape: {full_edge_np.shape}")

    # ── 3. Train/Val/Test split ───────────────────────────────────────────
    n_train  = int(T * TRAIN_RATIO)
    n_val    = int(T * VAL_RATIO)
    n_test   = T - n_train - n_val

    train_nodes = raw_features[:n_train]         # [T_train, N, 6]
    val_nodes   = raw_features[n_train:n_train + n_val]
    test_nodes  = raw_features[n_train + n_val:]

    train_edges = full_edge_np[:n_train]
    val_edges   = full_edge_np[n_train:n_train + n_val]
    test_edges  = full_edge_np[n_train + n_val:]

    # ── 4. Normalise node features using training-set statistics ─────────
    # Shape of stats: [1, 1, 6] for broadcasting over [T, N, 6]
    feat_min = train_nodes.min(axis=(0, 1), keepdims=True)   # [1, 1, 6]
    feat_max = train_nodes.max(axis=(0, 1), keepdims=True)

    train_nodes_norm = minmax_normalize(train_nodes, feat_min, feat_max)
    val_nodes_norm   = minmax_normalize(val_nodes,   feat_min, feat_max)
    test_nodes_norm  = minmax_normalize(test_nodes,  feat_min, feat_max)

    # PM2.5 scale factors for denormalisation (channel 0)
    pm25_min = float(feat_min[0, 0, 0])
    pm25_max = float(feat_max[0, 0, 0])

    # ── 5. Build tensors and datasets ────────────────────────────────────
    def to_tensor(arr):
        return torch.from_numpy(arr)

    edge_index_t = to_tensor(edge_index_np).long().to(device)   # [2, E]

    train_ds = SeoulPM25Dataset(to_tensor(train_nodes_norm), edge_index_t.cpu(),
                                to_tensor(train_edges), WINDOW)
    val_ds   = SeoulPM25Dataset(to_tensor(val_nodes_norm),   edge_index_t.cpu(),
                                to_tensor(val_edges),   WINDOW)
    test_ds  = SeoulPM25Dataset(to_tensor(test_nodes_norm),  edge_index_t.cpu(),
                                to_tensor(test_edges),  WINDOW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    print(f"  Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── 6. Model, optimiser, loss ─────────────────────────────────────────
    model = STGNNModel(
        node_dim=F,
        edge_dim=5,
        gat_hidden=GAT_HIDDEN,
        gru_hidden=GRU_HIDDEN,
        num_heads=NUM_HEADS,
        num_nodes=N,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── 7. Training loop ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    print("\nTraining...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_acc = 0.0
        n_batches = 0

        for node_w, edge_w, target in train_loader:
            node_w  = node_w.to(device)
            edge_w  = edge_w.to(device)
            target  = target.to(device)

            optimizer.zero_grad()
            pred, _ = model(node_w, edge_index_t, edge_w)   # [B, N, 1]
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_acc += loss.item()
            n_batches += 1

        avg_train = train_loss_acc / n_batches

        # Validation loss (normalised scale)
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for node_w, edge_w, target in val_loader:
                node_w  = node_w.to(device)
                edge_w  = edge_w.to(device)
                target  = target.to(device)
                pred, _ = model(node_w, edge_index_t, edge_w)
                val_loss_acc += criterion(pred, target).item()
                val_batches += 1
        avg_val = val_loss_acc / val_batches

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CHECKPOINT)

        if epoch % 5 == 0:
            print(f"  Epoch [{epoch:3d}/{EPOCHS}]  "
                  f"Train Loss: {avg_train:.6f}  "
                  f"Val Loss: {avg_val:.6f}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    # ── 8. Load best model and evaluate on test set ───────────────────────
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    mae, rmse, last_h_i = evaluate(
        model, test_loader, edge_index_t, device, pm25_min, pm25_max
    )

    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    # h_i for a single sample: take first item in last batch
    h_i_single = last_h_i[0]   # [N, gru_hidden]
    print(f"Node embeddings h_i shape: {list(h_i_single.shape)}")


if __name__ == "__main__":
    main()
