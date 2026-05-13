"""
[Step 3] V5-base Grid PM10 추론 — 전체 기간 (train + val + test)

출력:
    HiddenExtension_V5/checkpoints/V5-base/grid_pm10_train.npy  (T_train, G)
    HiddenExtension_V5/checkpoints/V5-base/grid_pm10_val.npy    (T_val, G)
    HiddenExtension_V5/checkpoints/V5-base/grid_pm10_test.npy   (T_test, G)
    checkpoints/v5_ts_lookup.csv

실행:
    python3 step3_v5_inference_all.py [--split train|val|test|all]
"""
import os, sys, argparse, json, importlib

# ── V2 config를 먼저 로드 (sys.path 오염 전) ──────────────────────────────
_V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _V2_DIR)
import config as v2cfg  # RoadExtension_V2/config.py

_ROOT = os.path.abspath(os.path.join(_V2_DIR, ".."))
_V5   = os.path.join(_ROOT, "HiddenExtension_V5")
_V3   = os.path.join(_ROOT, "HiddenExtension_V3")

import numpy as np
import pandas as pd
import torch


def load_v5_modules():
    """V5 전용 모듈 로드 (sys.path 조작 필요)."""
    # 기존 config/model/dataset 캐시 제거
    for mod in ["model", "config", "dataset", "grid_hidden"]:
        sys.modules.pop(mod, None)

    # V5, V3 경로 추가 (V5 우선)
    for p in [_ROOT, _V3, _V5]:
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, _V3)
    sys.path.insert(0, _V5)

    from config import (          # noqa: F401  ← V5 config
        H_DIM, N_STATION, LUR_DIM, TEMPORAL_NAMES,
    )
    from model import FixedEffectPMModel       # noqa: F401
    from dataset import V5Dataset, build_temporal_features, get_season  # noqa: F401
    from grid_hidden import GridHiddenGenerator  # noqa: F401

    return H_DIM, N_STATION, LUR_DIM, TEMPORAL_NAMES, \
           FixedEffectPMModel, V5Dataset, build_temporal_features, get_season, \
           GridHiddenGenerator


def run_split(split: str, device):
    out_path = v2cfg.V5_GRID_PM[split]
    if os.path.exists(out_path):
        arr = np.load(out_path)
        print("  {} 이미 존재: {}  {}".format(split, out_path, arr.shape))
        return

    (H_DIM, N_STATION, LUR_DIM, TEMPORAL_NAMES,
     FixedEffectPMModel, V5Dataset, build_temporal_features,
     get_season, GridHiddenGenerator) = load_v5_modules()

    exp_dir = v2cfg.V5_CKPT_DIR
    with open(os.path.join(exp_dir, "metrics.json")) as f:
        cfg = json.load(f)

    model = FixedEffectPMModel(
        n_stations=N_STATION, h_dim=H_DIM, lur_dim=LUR_DIM,
        temporal_dim=len(TEMPORAL_NAMES),
        mlp_hidden=cfg["mlp_hidden"],
        use_bias=cfg["use_bias"],
        use_seasonal_bias=cfg["use_seasonal"],
        use_hier_lur=cfg["use_hier"],
    )
    model.load_state_dict(
        torch.load(os.path.join(exp_dir, "best_model.pt"), map_location=device))
    model.to(device).eval()

    train_ds   = V5Dataset("train")
    h_scaler   = train_ds.h_scaler
    lur_scaler = train_ds.lur_scaler

    h_all    = np.load(os.path.join(v2cfg.HIDDEN_DIR, "h_{}.npy".format(split)))
    coords_s = np.load(os.path.join(v2cfg.HIDDEN_DIR, "coords.npy"))
    grid_csv = pd.read_csv(v2cfg.GRID_CSV)
    coords_g = grid_csv[["lat", "lon"]].values.astype(np.float32)
    G, T     = len(grid_csv), len(h_all)

    time_idx   = np.load(v2cfg.TIME_IDX[split])
    global_idx = time_idx[v2cfg.STGNN_WINDOW:]
    ts_all     = np.load(v2cfg.TIMESTAMPS_PATH, allow_pickle=True)
    ts_split   = ts_all[global_idx]

    ndvi_all = np.load(v2cfg.NDVI_PATH)
    ibi_all  = np.load(v2cfg.IBI_PATH)
    lc_all   = np.load(v2cfg.LC_PATH)
    bldg_all = np.load(v2cfg.BLDG_PATH)
    wind_all = np.load(v2cfg.WIND_PATH)

    ndvi_g = ndvi_all[global_idx].mean(axis=0)
    ibi_g  = ibi_all[global_idx].mean(axis=0)
    X_lur  = np.column_stack([ndvi_g, ibi_g, lc_all, bldg_all]).astype(np.float32)
    X_lur_t = torch.from_numpy(
        lur_scaler.transform(X_lur).astype(np.float32)).to(device)

    gen = GridHiddenGenerator(coords_s, coords_g, method="wind", k=10)
    gen.fit()

    pm_grid    = np.zeros((T, G), dtype=np.float32)
    temp_feats = build_temporal_features(ts_split)

    from tqdm import tqdm
    with torch.no_grad():
        for t in tqdm(range(T), desc="V5 grid [{}/{}]".format(split, T)):
            gt     = global_idx[t]
            wind_t = wind_all[gt]
            h_g_np = gen.transform_timestep(h_all[t], wind_t)
            h_g    = torch.from_numpy(
                h_scaler.transform(h_g_np).astype(np.float32)).to(device)
            temp_t = torch.from_numpy(temp_feats[t]).to(device)
            season = get_season(pd.to_datetime(ts_split[t]).month)
            pm_grid[t] = model.predict_grid(
                h_g, X_lur_t, temp_t, season).cpu().numpy()

    np.save(out_path, pm_grid)
    print("  저장: {}  {}".format(out_path, pm_grid.shape))


def build_ts_lookup():
    """(global_ts_idx, split, local_idx, timestamp) lookup CSV 생성."""
    lookup_path = os.path.join(v2cfg.CKPT_DIR, "v5_ts_lookup.csv")
    if os.path.exists(lookup_path):
        return pd.read_csv(lookup_path, parse_dates=["timestamp"])

    ts_all = np.load(v2cfg.TIMESTAMPS_PATH, allow_pickle=True)
    rows = []
    for split in ["train", "val", "test"]:
        time_idx   = np.load(v2cfg.TIME_IDX[split])
        global_idx = time_idx[v2cfg.STGNN_WINDOW:]
        for local_i, gi in enumerate(global_idx):
            rows.append({
                "global_ts_idx": int(gi),
                "split":         split,
                "local_idx":     local_i,
                "timestamp":     pd.to_datetime(ts_all[gi]),
            })

    df = pd.DataFrame(rows).sort_values("global_ts_idx").reset_index(drop=True)
    df.to_csv(lookup_path, index=False)
    print("TS lookup 저장: {}  ({} rows)".format(lookup_path, len(df)))
    return df


def main(splits=None):
    if splits is None:
        splits = ["train", "val", "test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== [Step 3] V5 Grid Inference ===")
    print("device: {}  splits: {}".format(device, splits))

    for sp in splits:
        print("\n[{}]".format(sp))
        run_split(sp, device)

    print("\nTS lookup 생성...")
    build_ts_lookup()
    print("\n완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    args = parser.parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    main(splits)
