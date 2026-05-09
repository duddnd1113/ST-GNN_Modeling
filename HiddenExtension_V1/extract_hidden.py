"""
ST-GNN에서 h_i,t (hidden vector)와 PM10 실측값을 추출해 저장합니다.
학습 완료된 모델을 그대로 사용 (가중치 변경 없음).

출력:
    data/hidden_vectors/h_train.npy   [T_train, N, 64]
    data/hidden_vectors/h_val.npy     [T_val,   N, 64]
    data/hidden_vectors/h_test.npy    [T_test,  N, 64]
    data/hidden_vectors/pm_train.npy  [T_train, N]   (μg/m³, 역정규화)
    data/hidden_vectors/pm_val.npy    [T_val,   N]
    data/hidden_vectors/pm_test.npy   [T_test,  N]
    data/hidden_vectors/coords.npy    [N, 2]         (lat, lon)
    data/hidden_vectors/stations.npy  (N,)           측정소명 배열
"""
import sys, os
# 부모 디렉토리(ST-GNN Modeling)를 먼저 탐색 → STGNNModel, load_scenario_split 등 사용
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    STGNN_CHECKPOINT, STGNN_SCENARIO, STGNN_WINDOW,
    HIDDEN_DIR, BATCH_SIZE,
    H_DIM, ATT_HIDDEN,
)
from dataset import load_scenario_split, STGNNScenarioDataset
from graph_builder import compute_all_dynamic_edge_features, get_full_edge_features
from model import STGNNModel


def minmax(arr, a_min, a_max):
    return (arr - a_min) / (a_max - a_min + 1e-8)


def extract_and_save():
    os.makedirs(HIDDEN_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"Device: GPU {idx} — {torch.cuda.get_device_name(idx)}")
        print(f"  VRAM: {torch.cuda.memory_allocated(idx)/1e9:.2f} GB used / "
              f"{torch.cuda.get_device_properties(idx).total_memory/1e9:.1f} GB total")
    else:
        print("Device: CPU")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    (train_nodes, val_nodes, test_nodes,
     train_mask, val_mask, test_mask,
     coords, feature_cols) = load_scenario_split(STGNN_SCENARIO)

    feat_min = train_nodes.min(axis=(0, 1), keepdims=True)
    feat_max = train_nodes.max(axis=(0, 1), keepdims=True)
    pm10_idx = feature_cols.index("PM10")
    pm10_min = float(feat_min[0, 0, pm10_idx])
    pm10_max = float(feat_max[0, 0, pm10_idx])

    N, F = train_nodes.shape[1], train_nodes.shape[2]

    # ── 그래프 로드 ──────────────────────────────────────────────────────────
    graph_path = os.path.join(os.path.dirname(__file__), "../graphs/static")
    edge_index_np  = np.load(os.path.join(graph_path, "edge_index.npy"))
    static_attr_np = np.load(os.path.join(graph_path, "static_attr.npy"))
    edge_bearings  = np.load(os.path.join(graph_path, "edge_bearings.npy"))
    E = edge_index_np.shape[1]
    edge_index_t = torch.from_numpy(edge_index_np).long().to(device)

    def build_edges(raw_arr):
        T = raw_arr.shape[0]
        dyn = compute_all_dynamic_edge_features(edge_index_np, raw_arr, edge_bearings)
        static_rep = np.broadcast_to(static_attr_np[None], (T, E, 3)).copy()
        return get_full_edge_features(static_rep, dyn)

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    model = STGNNModel(
        node_dim=F, edge_dim=5,
        gat_hidden=H_DIM, gru_hidden=H_DIM,
        num_heads=4, num_nodes=N,
    ).to(device)
    model.load_state_dict(torch.load(STGNN_CHECKPOINT, map_location=device))
    model.eval()

    # ── split별 추출 ─────────────────────────────────────────────────────────
    splits = {
        "train": (train_nodes, train_mask),
        "val":   (val_nodes,   val_mask),
        "test":  (test_nodes,  test_mask),
    }

    for split_name, (raw_nodes, mask) in splits.items():
        norm_nodes = minmax(raw_nodes, feat_min, feat_max)
        edges = build_edges(raw_nodes)

        ds = STGNNScenarioDataset(
            node_features=torch.from_numpy(norm_nodes),
            edge_index=edge_index_t.cpu(),
            edge_features_all=torch.from_numpy(edges),
            window=STGNN_WINDOW,
            mask=torch.from_numpy(mask) if mask is not None else None,
            pm10_idx=pm10_idx,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        all_h, all_pm = [], []
        with torch.no_grad():
            for node_w, edge_w, target, _ in loader:
                _, h_i, _ = model(node_w.to(device), edge_index_t, edge_w.to(device))
                all_h.append(h_i.cpu().numpy())
                all_pm.append(target.cpu().numpy())

        h_arr  = np.concatenate(all_h,  axis=0)             # [T, N, 64]
        pm_arr = np.concatenate(all_pm, axis=0)[:, :, 0]   # [T, N] (normalized)
        pm_arr = pm_arr * (pm10_max - pm10_min) + pm10_min  # 역정규화 → μg/m³

        np.save(os.path.join(HIDDEN_DIR, f"h_{split_name}.npy"),  h_arr)
        np.save(os.path.join(HIDDEN_DIR, f"pm_{split_name}.npy"), pm_arr)
        print(f"  {split_name}: h={h_arr.shape}, pm={pm_arr.shape}")

    # ── 좌표 & 측정소명 저장 ──────────────────────────────────────────────────
    import pickle, pandas as pd
    from scipy.spatial import cKDTree
    from dataset import SCENARIO_DIR, _ID_COLS
    df = pd.read_csv(SCENARIO_DIR / f"{STGNN_SCENARIO}.csv")
    stations = sorted(df["측정소명"].unique())
    coords_arr = np.array([(c[0], c[1]) for c in coords], dtype=np.float32)  # [N, 2]

    np.save(os.path.join(HIDDEN_DIR, "coords.npy"),   coords_arr)
    np.save(os.path.join(HIDDEN_DIR, "stations.npy"), np.array(stations))
    print(f"  coords: {coords_arr.shape}")

    # ── 측정소 → grid_basic 인덱스 매핑 ──────────────────────────────────────
    from config import GRID_CSV_PATH
    grid_basic = pd.read_csv(GRID_CSV_PATH)
    grid_coords = grid_basic[["lat", "lon"]].values.astype(np.float32)
    tree = cKDTree(grid_coords)
    _, sta_to_grid = tree.query(coords_arr)                # [N] — 각 측정소의 grid_basic 인덱스
    np.save(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"), sta_to_grid.astype(np.int32))
    print(f"  station_to_grid_idx: {sta_to_grid.shape}")
    print(f"  저장 완료 → {HIDDEN_DIR}")


if __name__ == "__main__":
    extract_and_save()
