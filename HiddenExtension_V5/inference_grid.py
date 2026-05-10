"""
V5 Grid-level PM10 추론

Wind-aware IDW로 grid hidden 생성 → FixedEffectPMModel 적용

실행:
    python3 inference_grid.py --exp V5-hier
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import joblib

from config import (
    CKPT_DIR, H_DIM, N_STATION, LUR_DIM, TEMPORAL_NAMES,
    HIDDEN_DIR, GRID_CSV, TIMESTAMPS, TIME_IDX, STGNN_WINDOW,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH, WIND_PATH,
    LUR_NAMES,
)
from model import FixedEffectPMModel
from dataset import V5Dataset, build_temporal_features, get_season


def run_grid_inference(exp_id: str, split: str = "test"):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(CKPT_DIR, exp_id)

    # ── 실험 설정 로드 ────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "metrics.json")) as f:
        cfg = json.load(f)

    model = FixedEffectPMModel(
        n_stations=N_STATION, h_dim=H_DIM, lur_dim=LUR_DIM,
        temporal_dim=len(TEMPORAL_NAMES),
        mlp_hidden=cfg["mlp_hidden"],
        use_bias=cfg["use_bias"], use_seasonal_bias=cfg["use_seasonal"],
        use_hier_lur=cfg["use_hier"],
    )
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"),
                                      map_location=device))
    model.to(device).eval()

    # ── scaler 재현 (train split 기준) ───────────────────────────────────
    train_ds = V5Dataset("train")
    h_scaler  = train_ds.h_scaler
    lur_scaler = train_ds.lur_scaler

    # ── 기본 데이터 로드 ──────────────────────────────────────────────────
    h_all    = np.load(os.path.join(HIDDEN_DIR, f"h_{split}.npy"))   # (T, N, 64)
    coords_s = np.load(os.path.join(HIDDEN_DIR, "coords.npy"))       # (N, 2)
    sta2grid = np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"))

    grid_csv    = pd.read_csv(GRID_CSV)
    coords_grid = grid_csv[["lat","lon"]].values.astype(np.float32)  # (G, 2)
    G, T        = len(grid_csv), len(h_all)

    time_idx   = np.load(TIME_IDX[split])
    global_idx = time_idx[STGNN_WINDOW:]
    ts_all     = np.load(TIMESTAMPS)
    ts_split   = ts_all[global_idx]

    ndvi_all = np.load(NDVI_PATH)
    ibi_all  = np.load(IBI_PATH)
    lc_all   = np.load(LC_PATH)
    bldg_all = np.load(BLDG_PATH)
    wind_all = np.load(WIND_PATH)

    # ── Grid LUR 피처 (시간 평균) ─────────────────────────────────────────
    ndvi_g   = ndvi_all[global_idx].mean(axis=0)              # (G,)
    ibi_g    = ibi_all[global_idx].mean(axis=0)
    X_lur_g  = np.column_stack([ndvi_g, ibi_g, lc_all, bldg_all]).astype(np.float32)
    X_lur_g_norm = torch.from_numpy(
        lur_scaler.transform(X_lur_g).astype(np.float32)).to(device)  # (G, 9)

    # ── Wind-aware IDW (V3 재사용) ────────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "../HiddenExtension_V3"))
    from grid_hidden import GridHiddenGenerator
    gen = GridHiddenGenerator(coords_s, coords_grid, method="wind", k=10)
    gen.fit()

    # ── 타임스텝별 추론 ───────────────────────────────────────────────────
    pm_grid = np.zeros((T, G), dtype=np.float32)
    temp_feats = build_temporal_features(ts_split)   # (T, 9)

    with torch.no_grad():
        for t in tqdm(range(T), desc=f"Grid inference [{exp_id}/{split}]"):
            global_t = global_idx[t]
            wind_t   = wind_all[global_t]                  # (N, 2)

            # Grid hidden (IDW)
            h_g_np = gen.transform_timestep(h_all[t], wind_t)  # (G, 64)
            h_g    = torch.from_numpy(
                h_scaler.transform(h_g_np).astype(np.float32)).to(device)

            # Temporal
            temp_t = torch.from_numpy(temp_feats[t]).to(device)   # (9,)
            season = get_season(pd.to_datetime(ts_split[t]).month)

            pm_grid[t] = model.predict_grid(
                h_g, X_lur_g_norm, temp_t, season
            ).cpu().numpy()

    # ── 저장 ─────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, f"grid_pm10_{split}.npy"), pm_grid)
    mean_pm = pm_grid.mean(axis=0)
    pd.DataFrame({
        "CELL_ID": grid_csv["CELL_ID"],
        "lat": grid_csv["lat"], "lon": grid_csv["lon"],
        "pm10_mean": mean_pm,
    }).to_csv(os.path.join(out_dir, f"grid_pm10_mean_{split}.csv"), index=False)

    print(f"\n저장 → {out_dir}/grid_pm10_{split}.npy  {pm_grid.shape}")
    print(f"PM10: mean={mean_pm.mean():.2f}  std={mean_pm.std():.2f}")
    return pm_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   type=str, default="V5-hier")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train","val","test"])
    args = parser.parse_args()
    run_grid_inference(args.exp, args.split)
