"""
V4 best_model.pt 평가 스크립트

핵심: h_test.npy(원본 hidden) 대신 V4 fine-tuned ST-GNN으로
      실제 테스트 데이터를 통과시켜 fresh hidden 생성 후 평가.

실행:
    python3 evaluate.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from config import (
    STGNN_PARAMS, STGNN_SCENARIO, STGNN_WINDOW, CKPT_DIR,
    SCENARIO_DIR, SPLIT_INFO, H_DIM, R_DIM, ATT_HIDDEN, N_HEADS_ATT, DROPOUT,
    HIDDEN_DIR, GEO_CLUSTERS,
)
from joint_model import JointSpatialSTGNN
from geo_loo import GeoLOOSampler


def compute_metrics(pred, true):
    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r2   = float(1 - np.sum((pred - true)**2) / (np.sum((true - true.mean())**2) + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 데이터 로드 (train_joint.py와 동일 방식) ────────────────────────────
    from model import STGNNModel
    from dataset import load_scenario_split, STGNNScenarioDataset
    from graph_builder import (build_static_graph, get_full_edge_features,
                                compute_all_dynamic_edge_features)

    (train_nodes, val_nodes, test_nodes,
     train_mask, val_mask, test_mask,
     coords, feature_cols) = load_scenario_split(
        STGNN_SCENARIO,
        scenario_dir=Path(SCENARIO_DIR),
        split_info_path=Path(SPLIT_INFO),
    )

    edge_index, edge_attr_static, edge_bearings = build_static_graph(coords, threshold_km=10.0)
    E = edge_index.shape[1]

    def build_full_edges(raw_arr):
        T = raw_arr.shape[0]
        dyn = compute_all_dynamic_edge_features(edge_index, raw_arr, edge_bearings)
        static_rep = np.broadcast_to(edge_attr_static[None], (T, E, 3)).copy()
        return get_full_edge_features(static_rep, dyn)

    def to_tensor(x):
        return torch.from_numpy(x) if x is not None else None

    ei_tensor = torch.from_numpy(edge_index.astype(np.int64))

    test_ds = STGNNScenarioDataset(
        to_tensor(test_nodes), ei_tensor, to_tensor(build_full_edges(test_nodes)),
        window=STGNN_WINDOW, mask=to_tensor(test_mask),
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0,
                             collate_fn=lambda b: {
                                 "nf": torch.stack([x[0] for x in b]),
                                 "ef": torch.stack([x[1] for x in b]),
                                 "tgt": torch.stack([x[2] for x in b]),
                             })

    # pm10 column index
    pm10_idx = feature_cols.index("PM10")
    coords_t = torch.tensor(np.array(coords), dtype=torch.float32).to(device)

    # ── V4 모델 로드 ──────────────────────────────────────────────────────────
    stgnn = STGNNModel(**STGNN_PARAMS)
    model = JointSpatialSTGNN(stgnn=stgnn, h_dim=H_DIM, r_dim=R_DIM,
                               n_heads=N_HEADS_ATT, att_hidden=ATT_HIDDEN, dropout=DROPOUT)
    ckpt = os.path.join(CKPT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    print(f"V4 모델 로드: {ckpt}")

    # ── Direct head 평가 ──────────────────────────────────────────────────────
    # V4 fine-tuned ST-GNN으로 fresh hidden 생성 → direct_head 적용
    d_preds, d_trues = [], []

    with torch.no_grad():
        for batch in test_loader:
            nf  = batch["nf"].to(device)   # (B, T, N, F)
            ef  = batch["ef"].to(device)   # (B, T, E, 5)
            tgt = batch["tgt"]             # (B, N, 1)

            _, h, _ = model.stgnn(nf, ei_tensor.to(device), ef)  # h: (B, N, d)
            pred    = model.direct_head(h)                        # (B, N, 1)

            d_preds.append(pred.squeeze(-1).cpu().numpy())
            d_trues.append(tgt.squeeze(-1).numpy())

    d_pred = np.concatenate(d_preds).flatten()   # (T*N,)
    d_true = np.concatenate(d_trues).flatten()   # (T*N,)
    direct_m = compute_metrics(d_pred, d_true)

    # ── Spatial (LOO) 평가 ────────────────────────────────────────────────────
    # Geographic cluster LOO: cluster 별로 context → target 예측
    # (V4 fine-tuned hidden 기준)
    h_all    = []
    pm_all   = []

    with torch.no_grad():
        for batch in test_loader:
            nf  = batch["nf"].to(device)
            ef  = batch["ef"].to(device)
            _, h, _ = model.stgnn(nf, ei_tensor.to(device), ef)
            h_all.append(h.cpu())
            pm_all.append(batch["tgt"].squeeze(-1))

    h_all  = torch.cat(h_all,  dim=0)   # (T, N, d)
    pm_all = torch.cat(pm_all, dim=0)   # (T, N)

    sampler = GeoLOOSampler(GEO_CLUSTERS)
    spatial_preds = np.zeros_like(pm_all.numpy())
    spatial_mask  = np.zeros(h_all.shape[1], dtype=bool)

    with torch.no_grad():
        for cluster_name, target_idx in sampler.all_clusters().items():
            N = h_all.shape[1]
            ctx_mask = np.ones(N, dtype=bool)
            ctx_mask[target_idx] = False
            spatial_mask[target_idx] = True

            T_total = h_all.shape[0]
            BATCH = 64
            for t_s in range(0, T_total, BATCH):
                t_e  = min(t_s + BATCH, T_total)
                h_b  = h_all[t_s:t_e].to(device)           # (B, N, d)
                h_ctx = h_b[:, ctx_mask, :]
                c_all = coords_t.unsqueeze(0).expand(t_e-t_s, -1, -1)
                c_ctx = c_all[:, ctx_mask, :]
                c_tgt = c_all[:, ~ctx_mask, :]

                pred = model.spatial_head(h_ctx, c_ctx, c_tgt)   # (B, N_tgt, 1)
                spatial_preds[t_s:t_e, ~ctx_mask] = pred.squeeze(-1).cpu().numpy()

    sp_m = compute_metrics(
        spatial_preds[:, spatial_mask].flatten(),
        pm_all[:, spatial_mask].numpy().flatten(),
    )

    # ── 원본 ST-GNN 직접 비교 (V4 fine-tuned ST-GNN의 output_head) ──────────
    stgnn_preds, stgnn_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            nf  = batch["nf"].to(device)
            ef  = batch["ef"].to(device)
            pred_fc, _, _ = model.stgnn(nf, ei_tensor.to(device), ef)
            stgnn_preds.append(pred_fc.squeeze(-1).cpu().numpy())
            stgnn_trues.append(batch["tgt"].squeeze(-1).numpy())

    stgnn_m = compute_metrics(
        np.concatenate(stgnn_preds).flatten(),
        np.concatenate(stgnn_trues).flatten(),
    )

    # ── 기존 ST-GNN baseline (metrics.json) ──────────────────────────────────
    baseline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json"
    )
    orig_mae = json.load(open(baseline_path))["mae"] if os.path.exists(baseline_path) else None

    # ── 출력 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  V4 평가 결과 (V4 fine-tuned hidden 기준)")
    print(f"{'='*65}")
    if orig_mae:
        print(f"  원본 ST-GNN (frozen)         MAE={orig_mae:.4f}  (기준)")
    print(f"  V4 ST-GNN head (fine-tuned)  MAE={stgnn_m['mae']:.4f}  "
          f"R²={stgnn_m['r2']:.4f}  ← forecasting 보존 여부")
    print(f"  V4 Direct head               MAE={direct_m['mae']:.4f}  "
          f"R²={direct_m['r2']:.4f}  ← direct PM 예측")
    print(f"  V4 Spatial LOO               MAE={sp_m['mae']:.4f}  "
          f"R²={sp_m['r2']:.4f}  ← 공간 확장")
    print(f"{'='*65}")

    result = {
        "orig_stgnn_mae":     orig_mae,
        "v4_stgnn_mae":       stgnn_m["mae"],   "v4_stgnn_r2":   stgnn_m["r2"],
        "v4_direct_mae":      direct_m["mae"],  "v4_direct_r2":  direct_m["r2"],
        "v4_spatial_mae":     sp_m["mae"],       "v4_spatial_r2": sp_m["r2"],
    }
    out_path = os.path.join(CKPT_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"저장 → {out_path}")


if __name__ == "__main__":
    main()
