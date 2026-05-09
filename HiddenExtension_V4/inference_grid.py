"""
V4 Grid-level PM10 추론

학습된 JointSpatialSTGNN의 Spatial Head로 city-wide grid hidden 생성
→ PM10 예측 map 출력

실행:
    python3 inference_grid.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    STGNN_CKPT, STGNN_SCENARIO, STGNN_WINDOW, STGNN_PARAMS,
    SCENARIO_DIR, SPLIT_INFO, GRAPH_DIR,
    H_DIM, R_DIM, ATT_HIDDEN, N_HEADS_ATT, DROPOUT,
    CKPT_DIR, HIDDEN_DIR, GRID_CSV, WIND_PATH,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH,
)
from joint_model import JointSpatialSTGNN


def run_grid_inference(split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    from model import STGNNModel
    stgnn = STGNNModel(**STGNN_PARAMS)
    model = JointSpatialSTGNN(stgnn=stgnn, h_dim=H_DIM, r_dim=R_DIM,
                               n_heads=N_HEADS_ATT, att_hidden=ATT_HIDDEN, dropout=DROPOUT)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pt"), map_location=device))
    model.to(device).eval()

    # ── hidden vector 로드 (V1에서 추출된 것 재사용) ───────────────────────
    h_all    = np.load(os.path.join(HIDDEN_DIR, f"h_{split}.npy"))      # (T, N, 64)
    coords_s = np.load(os.path.join(HIDDEN_DIR, "coords.npy"))          # (N, 2)
    sta2grid = np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"))

    grid_csv    = pd.read_csv(GRID_CSV)
    coords_grid = grid_csv[["lat", "lon"]].values.astype(np.float32)    # (G, 2)
    G, T        = len(grid_csv), len(h_all)

    coords_sta_t = torch.tensor(coords_s, dtype=torch.float32).to(device)  # (N, 2)
    coords_grid_t = torch.tensor(coords_grid, dtype=torch.float32).to(device)  # (G, 2)

    # ── Grid PM 예측 ─────────────────────────────────────────────────────
    pm_grid = np.zeros((T, G), dtype=np.float32)

    BATCH_G = 512   # grid 배치 크기 (메모리 절약)

    with torch.no_grad():
        for t in tqdm(range(T), desc=f"Grid inference [{split}]"):
            h_t = torch.tensor(h_all[t], dtype=torch.float32).to(device)  # (N, d)

            for g_start in range(0, G, BATCH_G):
                g_end  = min(g_start + BATCH_G, G)
                c_tgt  = coords_grid_t[g_start:g_end].unsqueeze(0)  # (1, Bg, 2)
                c_ctx  = coords_sta_t.unsqueeze(0)                   # (1, N, 2)
                h_ctx  = h_t.unsqueeze(0)                            # (1, N, d)

                pm_pred = model.spatial_head(h_ctx, c_ctx, c_tgt)   # (1, Bg, 1)
                pm_grid[t, g_start:g_end] = pm_pred.squeeze().cpu().numpy()

    # ── 저장 ─────────────────────────────────────────────────────────────
    npy_path = os.path.join(CKPT_DIR, f"grid_pm10_{split}.npy")
    np.save(npy_path, pm_grid)

    mean_pm = pm_grid.mean(axis=0)
    pd.DataFrame({
        "CELL_ID":   grid_csv["CELL_ID"],
        "lat":       grid_csv["lat"],
        "lon":       grid_csv["lon"],
        "pm10_mean": mean_pm,
    }).to_csv(os.path.join(CKPT_DIR, f"grid_pm10_mean_{split}.csv"), index=False)

    print(f"\n저장 완료 → {CKPT_DIR}/")
    print(f"  grid_pm10_{split}.npy  {pm_grid.shape}")
    print(f"  PM10: mean={mean_pm.mean():.2f}  std={mean_pm.std():.2f}")
    return pm_grid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train","val","test"])
    args = parser.parse_args()
    run_grid_inference(args.split)
