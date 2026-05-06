"""
Scenario-based dataset loader for Seoul PM10 ST-GNN.

Loads pre-built scenario CSVs, splits by pre-saved time indices,
pivots to (T, N, F) tensors, and separates mask columns for loss weighting.

Feature column ordering in all scenarios (fixed by data_preprocess.ipynb):
    [풍향_10m, 풍속_10m, 동서 방향 풍속, 남북 방향 풍속, PM10, ...]
     idx 0       idx 1       idx 2             idx 3        idx 4
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SCENARIO_DIR = Path("/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios")
SPLIT_INFO_PATH = Path("/workspace/ST-GNN Modeling/split_info.pkl")

_ID_COLS = {"측정소명", "위도", "경도", "time"}


def load_scenario_split(
    scenario_name: str,
    scenario_dir: Path = SCENARIO_DIR,
    split_info_path: Path = SPLIT_INFO_PATH,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    List[Tuple[float, float]],
    List[str],
]:
    """Load a scenario CSV and split into train / val / test.

    Mask columns (ending with '_mask') are stripped from node features
    and returned separately as a combined binary mask (OR of all mask cols).
    This mask is used as a loss weight during training: 1 = imputed value
    (downweight), 0 = real measurement (full weight).

    Returns:
        train_nodes, val_nodes, test_nodes : np.ndarray [T, N, F] float32
        train_mask, val_mask, test_mask    : np.ndarray [T, N] float32, or None
        coords                             : list of (lat, lon) per station
        feature_cols                       : non-mask feature column names
    """
    csv_path = scenario_dir / f"{scenario_name}.csv"
    df = pd.read_csv(csv_path)

    with open(split_info_path, "rb") as f:
        split_info = pickle.load(f)

    # Alphabetical station ordering — matches how CSVs were saved
    stations = sorted(df["측정소명"].unique())
    N = len(stations)

    coord_df = df.drop_duplicates("측정소명").set_index("측정소명")[["위도", "경도"]]
    coords = [
        (float(coord_df.loc[s, "위도"]), float(coord_df.loc[s, "경도"]))
        for s in stations
    ]

    all_cols = [c for c in df.columns if c not in _ID_COLS]
    mask_cols    = [c for c in all_cols if c.endswith("_mask")]
    feature_cols = [c for c in all_cols if not c.endswith("_mask")]

    train_set = set(split_info["train_times"])
    valid_set  = set(split_info["valid_times"])
    test_set   = set(split_info["test_times"])

    df_train = df[df["time"].isin(train_set)].sort_values(["time", "측정소명"])
    df_val   = df[df["time"].isin(valid_set)].sort_values(["time", "측정소명"])
    df_test  = df[df["time"].isin(test_set)].sort_values(["time", "측정소명"])

    def to_arrays(sub_df: pd.DataFrame):
        T = sub_df["time"].nunique()
        F = len(feature_cols)

        node_arr = sub_df[feature_cols].values.reshape(T, N, F).astype(np.float32)

        if mask_cols:
            # Combined mask: 1 if any mask column is 1 for that (time, station)
            mask_arr = (
                sub_df[mask_cols].max(axis=1).values.reshape(T, N).astype(np.float32)
            )
        else:
            mask_arr = None

        return node_arr, mask_arr

    train_nodes, train_mask = to_arrays(df_train)
    val_nodes,   val_mask   = to_arrays(df_val)
    test_nodes,  test_mask  = to_arrays(df_test)

    return (
        train_nodes, val_nodes, test_nodes,
        train_mask, val_mask, test_mask,
        coords, feature_cols,
    )


class STGNNScenarioDataset(Dataset):
    """Sliding-window dataset for scenario-based PM10 forecasting.

    Each sample: (node_window, edge_window, target_pm10, loss_mask)

    Args:
        node_features     : [T, N, F] float32 — already normalised
        edge_index        : [2, E] long
        edge_features_all : [T, E, 5] float32
        window            : lookback window size (hours)
        mask              : [T, N] float32, or None
                            1 = imputed point (use as downweight in loss)
        pm10_idx          : index of PM10 in the feature dimension
        edge_active_mask  : [T, E] float32, or None  (soft_dynamic mode only)
                            1 = wind blows along this edge at time t,
                            0 = reverse wind → zero out all edge features
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features_all: torch.Tensor,
        window: int = 12,
        mask: Optional[torch.Tensor] = None,
        pm10_idx: int = 4,
        edge_active_mask: Optional[torch.Tensor] = None,
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features_all = edge_features_all
        self.window = window
        self.mask = mask
        self.pm10_idx = pm10_idx
        self.edge_active_mask = edge_active_mask

    def __len__(self) -> int:
        return len(self.node_features) - self.window

    def __getitem__(self, idx: int):
        """
        Returns:
            node_w  : [window, N, F]
            edge_w  : [window, E, 5]
            target  : [N, 1]  — PM10 at t+1
            mask_t  : [N]     — 1 where imputed (for loss weighting)
        """
        node_w = self.node_features[idx: idx + self.window]
        edge_w = self.edge_features_all[idx: idx + self.window].clone()

        # soft_dynamic: zero out edge features where wind blows against the edge
        if self.edge_active_mask is not None:
            active = self.edge_active_mask[idx: idx + self.window].unsqueeze(-1)  # [W, E, 1]
            edge_w = edge_w * active

        target = self.node_features[idx + self.window, :, self.pm10_idx: self.pm10_idx + 1]

        if self.mask is not None:
            mask_t = self.mask[idx + self.window]
        else:
            mask_t = torch.zeros(target.shape[0], dtype=torch.float32)

        return node_w, edge_w, target, mask_t
