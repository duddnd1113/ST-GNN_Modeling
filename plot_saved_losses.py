"""
Evaluate saved ST-GNN checkpoints and plot train/val/test loss.

This script does not reconstruct per-epoch learning curves. The current
checkpoints contain only model weights, so it reloads each saved best model and
computes masked MSE on the train, validation, and test splits.

Examples:
    python3 plot_saved_losses.py --scenario S1_transport_pm10 --graph_mode static
    python3 plot_saved_losses.py --all
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_scenario_split, STGNNScenarioDataset
from graph_builder import compute_all_dynamic_edge_features, get_full_edge_features
from model import STGNNModel
from train import (
    ALL_SCENARIOS,
    BATCH_SIZE,
    GRAPH_MODES,
    GAT_HIDDEN,
    GRU_HIDDEN,
    NUM_HEADS,
    THRESHOLD_KM,
    WINDOW,
    load_or_build_graph,
    masked_mse,
    minmax_normalize,
)


def _mean_loss(model, loader, edge_index_t, device):
    model.eval()
    total, batches = 0.0, 0

    with torch.no_grad():
        for node_w, edge_w, target, mask in loader:
            node_w = node_w.to(device)
            edge_w = edge_w.to(device)
            target = target.to(device)
            mask = mask.to(device)

            pred, _ = model(node_w, edge_index_t, edge_w)
            total += masked_mse(pred, target, mask).item()
            batches += 1

    return total / max(batches, 1)


def evaluate_checkpoint(scenario_name, graph_mode, window=WINDOW):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = os.path.join("checkpoints", scenario_name, graph_mode, "best_model.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    (
        train_nodes,
        val_nodes,
        test_nodes,
        train_mask,
        val_mask,
        test_mask,
        coords,
        feature_cols,
    ) = load_scenario_split(scenario_name)

    edge_index_np, static_attr_np, edge_bearings_np = load_or_build_graph(
        graph_mode, coords, train_nodes, THRESHOLD_KM
    )

    feat_min = train_nodes.min(axis=(0, 1), keepdims=True)
    feat_max = train_nodes.max(axis=(0, 1), keepdims=True)

    train_norm = minmax_normalize(train_nodes, feat_min, feat_max)
    val_norm = minmax_normalize(val_nodes, feat_min, feat_max)
    test_norm = minmax_normalize(test_nodes, feat_min, feat_max)
    pm10_idx = feature_cols.index("PM10")

    def build_full_edges(raw_arr):
        t_steps = raw_arr.shape[0]
        dyn = compute_all_dynamic_edge_features(edge_index_np, raw_arr, edge_bearings_np)
        static_rep = np.broadcast_to(static_attr_np[None], (t_steps, edge_index_np.shape[1], 3)).copy()
        return get_full_edge_features(static_rep, dyn)

    train_edges = build_full_edges(train_nodes)
    val_edges = build_full_edges(val_nodes)
    test_edges = build_full_edges(test_nodes)

    if graph_mode == "soft_dynamic":
        def active_mask(edges_np):
            return torch.from_numpy((edges_np[:, :, 3] > 0).astype(np.float32))

        train_active = active_mask(train_edges)
        val_active = active_mask(val_edges)
        test_active = active_mask(test_edges)
    else:
        train_active = val_active = test_active = None

    edge_index_t = torch.from_numpy(edge_index_np).long().to(device)

    def to_t(arr):
        return torch.from_numpy(arr) if arr is not None else None

    def make_loader(nodes, edges, loss_mask, edge_active):
        ds = STGNNScenarioDataset(
            node_features=to_t(nodes),
            edge_index=edge_index_t.cpu(),
            edge_features_all=to_t(edges),
            window=window,
            mask=to_t(loss_mask),
            pm10_idx=pm10_idx,
            edge_active_mask=edge_active,
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    loaders = {
        "train": make_loader(train_norm, train_edges, train_mask, train_active),
        "val": make_loader(val_norm, val_edges, val_mask, val_active),
        "test": make_loader(test_norm, test_edges, test_mask, test_active),
    }

    model = STGNNModel(
        node_dim=train_nodes.shape[2],
        edge_dim=5,
        gat_hidden=GAT_HIDDEN,
        gru_hidden=GRU_HIDDEN,
        num_heads=NUM_HEADS,
        num_nodes=train_nodes.shape[1],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    losses = {split: _mean_loss(model, loader, edge_index_t, device) for split, loader in loaders.items()}
    return {"scenario": scenario_name, "graph_mode": graph_mode, **losses}


def save_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "saved_checkpoint_losses.csv")
    png_path = os.path.join(out_dir, "saved_checkpoint_losses.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "graph_mode", "train", "val", "test"])
        writer.writeheader()
        writer.writerows(rows)

    labels = [f"{r['scenario']}\n{r['graph_mode']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.25

    fig_width = max(10, len(rows) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(x - width, [r["train"] for r in rows], width, label="Train")
    ax.bar(x, [r["val"] for r in rows], width, label="Val")
    ax.bar(x + width, [r["test"] for r in rows], width, label="Test")
    ax.set_ylabel("Masked MSE loss")
    ax.set_title("Saved Best Checkpoint Loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return csv_path, png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S1_transport_pm10")
    parser.add_argument("--graph_mode", default="static", choices=GRAPH_MODES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--graph_modes", nargs="+", default=None, choices=GRAPH_MODES)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--out_dir", default="loss_plots")
    args = parser.parse_args()

    if args.all:
        scenarios = args.scenarios or ALL_SCENARIOS
        graph_modes = args.graph_modes or list(GRAPH_MODES)
        combos = [(scenario, graph_mode) for scenario in scenarios for graph_mode in graph_modes]
    else:
        combos = [(args.scenario, args.graph_mode)]

    rows = []
    for scenario, graph_mode in combos:
        print(f"Evaluating {scenario} / {graph_mode}")
        rows.append(evaluate_checkpoint(scenario, graph_mode, args.window))

    csv_path, png_path = save_outputs(rows, args.out_dir)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
