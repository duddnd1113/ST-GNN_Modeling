"""
V4 best_model.pt 평가 스크립트

V2/V3와 동일한 기준(pm_test.npy, 역정규화 scale)으로 성능 비교.

실행:
    python3 evaluate.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    STGNN_PARAMS, STGNN_WINDOW, CKPT_DIR,
    H_DIM, R_DIM, ATT_HIDDEN, N_HEADS_ATT, DROPOUT,
    HIDDEN_DIR, GEO_CLUSTERS,
)
from joint_model import JointSpatialSTGNN
from geo_loo import GeoLOOSampler


def compute_metrics(pred, true):
    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r2   = float(1 - np.sum((pred - true)**2) / np.sum((true - true.mean())**2))
    return dict(mae=mae, rmse=rmse, r2=r2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    from model import STGNNModel
    stgnn = STGNNModel(**STGNN_PARAMS)
    model = JointSpatialSTGNN(stgnn=stgnn, h_dim=H_DIM, r_dim=R_DIM,
                               n_heads=N_HEADS_ATT, att_hidden=ATT_HIDDEN, dropout=DROPOUT)
    ckpt = os.path.join(CKPT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    print(f"모델 로드: {ckpt}")

    # ── pm_test.npy 기준 평가 (V2/V3와 동일 scale) ─────────────────────────
    h_test  = torch.from_numpy(np.load(os.path.join(HIDDEN_DIR, "h_test.npy")))
    pm_test = np.load(os.path.join(HIDDEN_DIR, "pm_test.npy"))   # (T, N) raw scale
    coords  = torch.from_numpy(
        np.load(os.path.join(HIDDEN_DIR, "coords.npy"))
    ).to(device)

    T, N, d = h_test.shape
    sampler = GeoLOOSampler(GEO_CLUSTERS)

    # ── Direct head 평가 (h_target → PM 직접 예측) ─────────────────────────
    direct_preds = []
    with torch.no_grad():
        for t in range(T):
            h_t = h_test[t].unsqueeze(0).to(device)   # (1, N, d)
            pred = model.direct_head(h_t)              # (1, N, 1)
            direct_preds.append(pred.squeeze().cpu().numpy())

    direct_pred = np.stack(direct_preds)   # (T, N)
    direct_m    = compute_metrics(direct_pred.flatten(), pm_test.flatten())

    # ── Spatial (LOO) 평가: cluster별로 context → target 예측 ─────────────
    spatial_preds = np.zeros_like(pm_test)
    spatial_mask  = np.zeros(N, dtype=bool)

    with torch.no_grad():
        for cluster_name, target_idx in sampler.all_clusters().items():
            ctx_mask = np.ones(N, dtype=bool)
            ctx_mask[target_idx] = False
            spatial_mask[target_idx] = True

            for t in range(T):
                h_t   = h_test[t].unsqueeze(0).to(device)  # (1, N, d)
                h_ctx = h_t[:, ctx_mask, :]                 # (1, N_ctx, d)

                c_all = coords.unsqueeze(0)                 # (1, N, 2)
                c_ctx = c_all[:, ctx_mask, :]
                c_tgt = c_all[:, ~ctx_mask, :]

                pred = model.spatial_head(h_ctx, c_ctx, c_tgt)  # (1, N_tgt, 1)
                spatial_preds[t, ~ctx_mask] = pred.squeeze().cpu().numpy()

    # cluster에 포함된 station들만 spatial 평가
    sp_m = compute_metrics(
        spatial_preds[:, spatial_mask].flatten(),
        pm_test[:, spatial_mask].flatten(),
    )

    # ── ST-GNN baseline 로드 ─────────────────────────────────────────────
    baseline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json"
    )
    stgnn_mae = json.load(open(baseline_path))["mae"] if os.path.exists(baseline_path) else None

    # ── 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  V4 평가 결과 (pm_test.npy 기준, V2/V3와 동일 scale)")
    print(f"{'='*60}")
    if stgnn_mae:
        print(f"  ST-GNN baseline       MAE={stgnn_mae:.4f}")
    print(f"  V4 Direct head        MAE={direct_m['mae']:.4f}  "
          f"RMSE={direct_m['rmse']:.4f}  R²={direct_m['r2']:.4f}")
    print(f"  V4 Spatial (LOO)      MAE={sp_m['mae']:.4f}  "
          f"RMSE={sp_m['rmse']:.4f}  R²={sp_m['r2']:.4f}")
    print(f"  (spatial: {spatial_mask.sum()}개 station 기준)")
    print(f"{'='*60}")

    result = {
        "direct_mae":  direct_m["mae"],
        "direct_rmse": direct_m["rmse"],
        "direct_r2":   direct_m["r2"],
        "spatial_mae": sp_m["mae"],
        "spatial_rmse":sp_m["rmse"],
        "spatial_r2":  sp_m["r2"],
        "stgnn_baseline_mae": stgnn_mae,
    }
    out_path = os.path.join(CKPT_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"저장 → {out_path}")
    return result


if __name__ == "__main__":
    main()
