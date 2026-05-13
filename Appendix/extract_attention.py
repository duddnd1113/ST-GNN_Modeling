"""
Shared utility: load best model (S3+static+w12) and extract
GAT attention weights over the entire test set.

Returns:
    attn_mean   : np.ndarray [E]       — mean attention (over T, H, samples)
    attn_per_ts : np.ndarray [N_test, T, E, H] — raw per-sample attention
    edge_index  : np.ndarray [2, E]
    static_attr : np.ndarray [E, 3]
    edge_bearings: np.ndarray [E]
    coords      : list of (lat, lon), length N
    stations    : list of station names, length N
    test_pm10   : np.ndarray [N_test, N] — true PM10 at t+1 per sample
    test_times  : list of time strings for each test window
"""

import sys, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT   = Path("/workspace/ST-GNN Modeling")
sys.path.insert(0, str(ROOT))

from model   import STGNNModel
from dataset import load_scenario_split, STGNNScenarioDataset
from graph_builder import compute_all_dynamic_edge_features, get_full_edge_features

SCENARIO    = "S3_transport_pm10_pollutants"
GRAPH_MODE  = "static"
WINDOW      = 12
CKPT_PATH   = ROOT / "checkpoints/window_12/S3_transport_pm10_pollutants/static/best_model.pt"
GRAPH_DIR   = ROOT / "graphs/static"
SPLIT_PATH  = ROOT / "split_info.pkl"
SCENARIO_DIR = Path("/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios")

def load_everything():
    # ── Graph ────────────────────────────────────────────────────────────────
    edge_index   = np.load(GRAPH_DIR / "edge_index.npy")    # [2, E]
    static_attr  = np.load(GRAPH_DIR / "static_attr.npy")   # [E, 3]
    edge_bearings = np.load(GRAPH_DIR / "edge_bearings.npy") # [E]

    # ── Dataset ──────────────────────────────────────────────────────────────
    (train_nodes, val_nodes, test_nodes,
     train_mask,  val_mask,  test_mask,
     coords, feat_cols) = load_scenario_split(SCENARIO)

    # Normalisation (same as training)
    all_nodes = np.concatenate([train_nodes, val_nodes, test_nodes], axis=0)
    a_min = all_nodes.min(axis=(0, 1))
    a_max = all_nodes.max(axis=(0, 1))
    norm = lambda x: (x - a_min) / (a_max - a_min + 1e-8)

    test_norm = norm(test_nodes)   # [T_test, N, F]

    # Dynamic edge features for test set
    dyn_all = compute_all_dynamic_edge_features(edge_index, test_nodes, edge_bearings)
    full_attr = get_full_edge_features(
        np.broadcast_to(static_attr[None], (len(test_nodes), *static_attr.shape)),
        dyn_all
    )   # [T_test, E, 5]

    # Station names & coords
    csv = pd.read_csv(SCENARIO_DIR / f"{SCENARIO}.csv")
    station_list = sorted(csv["측정소명"].unique())
    coord_df = csv.drop_duplicates("측정소명").set_index("측정소명")[["위도","경도"]]
    coord_list = [(float(coord_df.loc[s,"위도"]), float(coord_df.loc[s,"경도"])) for s in station_list]

    # Time index for test windows
    with open(SPLIT_PATH, "rb") as f:
        split_info = pickle.load(f)
    test_time_idx = sorted(split_info["test_times"])

    # ── Model ────────────────────────────────────────────────────────────────
    state = torch.load(CKPT_PATH, map_location="cpu")
    model = STGNNModel(node_dim=test_norm.shape[-1], edge_dim=5, num_nodes=40)
    model.load_state_dict(state)
    model.eval()

    ei_t = torch.from_numpy(edge_index).long()

    # PM10 index in feature cols
    pm10_idx  = feat_cols.index("PM10")
    pm10_min  = float(a_min[pm10_idx])
    pm10_max  = float(a_max[pm10_idx])

    T_test = len(test_norm) - WINDOW
    all_attn = []     # [N_test, T, E, H]
    all_pm10 = []     # [N_test, N]  — true PM10 at t+WINDOW

    with torch.no_grad():
        for start in range(T_test):
            node_w = torch.from_numpy(test_norm[start:start+WINDOW]).float()   # [T, N, F]
            edge_w = torch.from_numpy(full_attr[start:start+WINDOW]).float()   # [T, E, 5]

            node_w = node_w.unsqueeze(0)   # [1, T, N, F]
            edge_w = edge_w.unsqueeze(0)   # [1, T, E, 5]

            _, _, attn_stack = model(node_w, ei_t, edge_w)  # attn: [T, E, H]
            all_attn.append(attn_stack.numpy())

            true_pm10 = test_nodes[start + WINDOW, :, pm10_idx]   # [N]
            all_pm10.append(true_pm10)

    attn_per_ts = np.stack(all_attn, axis=0)   # [N_test, T, E, H]
    test_pm10   = np.stack(all_pm10, axis=0)   # [N_test, N]
    attn_mean   = attn_per_ts.mean(axis=(0, 1, 3))  # [E]

    return dict(
        attn_mean    = attn_mean,
        attn_per_ts  = attn_per_ts,
        edge_index   = edge_index,
        static_attr  = static_attr,
        edge_bearings = edge_bearings,
        coords       = coord_list,
        stations     = station_list,
        test_pm10    = test_pm10,
        test_time_idx = test_time_idx,
        pm10_min     = pm10_min,
        pm10_max     = pm10_max,
    )


if __name__ == "__main__":
    print("Extracting attention weights...")
    data = load_everything()
    out = ROOT / "Appendix/attn_cache.npz"
    np.savez(out,
             attn_mean=data["attn_mean"],
             attn_per_ts=data["attn_per_ts"],
             edge_index=data["edge_index"],
             static_attr=data["static_attr"],
             edge_bearings=data["edge_bearings"],
             test_pm10=data["test_pm10"])
    print(f"Saved cache → {out}")
    print(f"  attn_per_ts shape: {data['attn_per_ts'].shape}")
    print(f"  test_pm10   shape: {data['test_pm10'].shape}")
