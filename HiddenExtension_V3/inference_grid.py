"""
학습된 RF 모델로 city-wide grid PM10 예측.

출력:
    checkpoints/{exp_id}/grid_pm10_predictions.npy  (T_test, G)
    checkpoints/{exp_id}/grid_pm10_mean.csv
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from config import (
    HIDDEN_DIR, GRID_CSV_PATH, CKPT_DIR, STGNN_WINDOW,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH, WIND_PATH, POPULATION_PATH,
    IDW_SIGMA_D, IDW_SIGMA_W, IDW_K,
)
from grid_hidden import GridHiddenGenerator
from feature_builder import (
    build_temporal_features, build_lur_features, load_split_data,
)


def run_grid_inference(exp_id: str, split: str = "test"):
    out_dir = os.path.join(CKPT_DIR, exp_id)
    assert os.path.exists(os.path.join(out_dir, "rf_model.pkl")), \
        f"학습된 모델 없음: {out_dir}/rf_model.pkl"

    rf      = joblib.load(os.path.join(out_dir, "rf_model.pkl"))
    builder = joblib.load(os.path.join(out_dir, "feature_builder.pkl"))

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    data        = load_split_data(split, STGNN_WINDOW)
    grid_csv    = pd.read_csv(GRID_CSV_PATH)
    coords_sta  = np.load(os.path.join(HIDDEN_DIR, "coords.npy"))
    coords_grid = grid_csv[["lat", "lon"]].values.astype(np.float32)
    G           = len(grid_csv)
    T           = len(data["h"])

    ndvi       = np.load(NDVI_PATH)
    ibi        = np.load(IBI_PATH)
    lc         = np.load(LC_PATH)
    bldg       = np.load(BLDG_PATH)
    wind_all   = np.load(WIND_PATH)
    population = np.load(POPULATION_PATH) if builder.use_population else None

    temp_all = build_temporal_features(data["timestamps"])  # (T, 9)

    # ── IDW 생성기 (hidden 사용 시) ──────────────────────────────────────────
    gen = None
    if builder.use_hidden:
        gen = GridHiddenGenerator(
            coords_sta, coords_grid,
            method=builder.idw_method if hasattr(builder, "idw_method") else "wind",
            sigma_d=IDW_SIGMA_D, sigma_w=IDW_SIGMA_W, k=IDW_K,
        )
        gen.fit()

    # ── LUR (grid-level 전체) ─────────────────────────────────────────────────
    lur_grid_all, lur_names = build_lur_features(
        ndvi=ndvi, ibi=ibi, lc=lc, bldg=bldg,
        population=population,
        target_global_idx=data["global_idx"],
    )  # (T, G, n_lur)

    # ── 타임스텝별 추론 ───────────────────────────────────────────────────────
    pm_grid = np.zeros((T, G), dtype=np.float32)

    for t in tqdm(range(T), desc=f"Grid inference [{exp_id}]"):
        global_t = data["global_idx"][t]

        # Grid hidden
        h_grid_t = None
        if gen is not None:
            wind_t   = wind_all[global_t]            # (N, 2)
            h_grid_t = gen.transform_timestep(data["h"][t], wind_t)  # (G, d)

        lur_t  = lur_grid_all[t]    # (G, n_lur)
        temp_t = np.tile(temp_all[t], (G, 1))  # (G, 9)

        X_grid = builder.assemble(
            h_flat       = h_grid_t  if builder.use_hidden else None,
            lur_flat     = lur_t     if builder.use_lur    else None,
            temporal_flat = temp_t   if builder.use_temporal else None,
        )
        pm_grid[t] = rf.predict(X_grid)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    npy_path = os.path.join(out_dir, f"grid_pm10_{split}.npy")
    np.save(npy_path, pm_grid)

    mean_pm = pm_grid.mean(axis=0)
    pd.DataFrame({
        "CELL_ID":   grid_csv["CELL_ID"],
        "lat":       grid_csv["lat"],
        "lon":       grid_csv["lon"],
        "pm10_mean": mean_pm,
    }).to_csv(os.path.join(out_dir, f"grid_pm10_mean_{split}.csv"), index=False)

    print(f"\n저장 완료 → {out_dir}/")
    print(f"  grid_pm10_{split}.npy  {pm_grid.shape}")
    print(f"  PM10 통계: mean={mean_pm.mean():.2f}  "
          f"min={mean_pm.min():.2f}  max={mean_pm.max():.2f}")
    return pm_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()
    run_grid_inference(args.exp, args.split)
