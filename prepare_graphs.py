"""
Pre-build and save graph structures for all modes.

Run once before training to cache graph files:
    python3 prepare_graphs.py

Saves to:
    graphs/static/         — used by 'static' and 'soft_dynamic' modes
    graphs/climatological/ — used by 'climatological' mode

Each folder contains:
    edge_index.npy     [2, E]  int64
    static_attr.npy    [E, 3]  float32  — [dist_norm, sin_bearing, cos_bearing]
    edge_bearings.npy  [E]     float32  — bearing in degrees, src→dst
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import SCENARIO_DIR, SPLIT_INFO_PATH, _ID_COLS
from graph_builder import build_static_graph, build_climatological_graph

GRAPH_DIR    = Path("graphs")
THRESHOLD_KM = 10.0
BASE_SCENARIO = "S1_transport_pm10"   # 좌표·바람 데이터 추출용 기준 시나리오


def load_coords_and_train_wind() -> tuple:
    """Load station coordinates and training-period wind data.

    All scenarios share the same stations and wind columns (uu, vv),
    so we extract them from the base scenario CSV.

    Returns:
        coords      : list of (lat, lon) — N stations, alphabetical order
        stations    : list of station names
        train_nodes : np.ndarray [T_train, N, 2] — [uu, vv] only (for climatological)
    """
    csv_path = SCENARIO_DIR / f"{BASE_SCENARIO}.csv"
    df = pd.read_csv(csv_path)

    with open(SPLIT_INFO_PATH, "rb") as f:
        split_info = pickle.load(f)

    stations = sorted(df["측정소명"].unique())
    N = len(stations)

    coord_df = df.drop_duplicates("측정소명").set_index("측정소명")[["위도", "경도"]]
    coords = [
        (float(coord_df.loc[s, "위도"]), float(coord_df.loc[s, "경도"]))
        for s in stations
    ]

    # Extract uu, vv columns for climatological graph (indices 2, 3 in feature order)
    feat_cols = [c for c in df.columns if c not in _ID_COLS and not c.endswith("_mask")]
    uu_col = "동서 방향 풍속"
    vv_col = "남북 방향 풍속"

    train_set = set(split_info["train_times"])
    df_train = (
        df[df["time"].isin(train_set)]
        .sort_values(["time", "측정소명"])
    )
    T_train = df_train["time"].nunique()

    uu_arr = df_train[uu_col].values.reshape(T_train, N).astype(np.float32)
    vv_arr = df_train[vv_col].values.reshape(T_train, N).astype(np.float32)

    # Build [T_train, N, F] with uu at idx 2, vv at idx 3 (matching graph_builder defaults)
    train_nodes = np.zeros((T_train, N, len(feat_cols)), dtype=np.float32)
    for fi, col in enumerate(feat_cols):
        vals = df_train[col].values.reshape(T_train, N)
        train_nodes[:, :, fi] = vals

    return coords, stations, train_nodes


def save_graph(mode: str, edge_index: np.ndarray, static_attr: np.ndarray, edge_bearings: np.ndarray):
    out_dir = GRAPH_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "edge_index.npy",    edge_index)
    np.save(out_dir / "static_attr.npy",   static_attr)
    np.save(out_dir / "edge_bearings.npy", edge_bearings)
    print(f"  [{mode}]  E={edge_index.shape[1]}  → {out_dir}/")


def main():
    print(f"Base scenario : {BASE_SCENARIO}")
    print(f"Threshold     : {THRESHOLD_KM} km")
    print(f"Output dir    : {GRAPH_DIR}/\n")

    coords, stations, train_nodes = load_coords_and_train_wind()
    N = len(stations)
    print(f"Stations: {N}")

    # ── Static ────────────────────────────────────────────────────────────
    print("Building static graph...")
    ei, sa, eb = build_static_graph(coords, threshold_km=THRESHOLD_KM)
    save_graph("static", ei, sa, eb)

    # ── Climatological ────────────────────────────────────────────────────
    print("Building climatological graph...")
    ei_c, sa_c, eb_c = build_climatological_graph(
        coords, train_nodes, threshold_km=THRESHOLD_KM
    )
    save_graph("climatological", ei_c, sa_c, eb_c)

    print("\nDone. Run train.py — graphs will be loaded automatically.")


if __name__ == "__main__":
    main()
