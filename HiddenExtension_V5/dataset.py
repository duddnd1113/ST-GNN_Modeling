"""
V5 Dataset — station_idx, hidden, LUR, temporal, PM 반환
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional

from config import (
    HIDDEN_DIR, GRID_DIR, TIMESTAMPS, TIME_IDX, STGNN_WINDOW,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH,
    LUR_NAMES, TEMPORAL_NAMES,
)


def build_temporal_features(timestamps: np.ndarray) -> np.ndarray:
    """timestamps (T,) → (T, 9) temporal features."""
    dt = pd.to_datetime(timestamps)
    return np.column_stack([
        np.sin(2 * np.pi * dt.hour / 24),
        np.cos(2 * np.pi * dt.hour / 24),
        np.sin(2 * np.pi * dt.month / 12),
        np.cos(2 * np.pi * dt.month / 12),
        np.sin(2 * np.pi * dt.dayofyear / 365),
        np.cos(2 * np.pi * dt.dayofyear / 365),
        (dt.dayofweek >= 5).astype(np.float32),
        dt.month.isin([12, 1, 2]).astype(np.float32),
        dt.month.isin([3, 4, 5]).astype(np.float32),
    ]).astype(np.float32)


def get_season(month: int) -> int:
    """0=겨울, 1=봄, 2=여름, 3=가을"""
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]


def get_hour_bin(hour: int) -> int:
    """0=야간(0-6), 1=오전(7-12), 2=오후(13-18), 3=저녁(19-23)"""
    if hour <= 6:   return 0
    if hour <= 12:  return 1
    if hour <= 18:  return 2
    return 3


class V5Dataset(Dataset):
    """
    반환:
        station_idx : (N,) int    — station 인덱스
        h           : (N, H)      — frozen ST-GNN hidden
        lur         : (N, L)      — 정규화된 LUR 피처
        temporal    : (N, T_feat) — 시간 피처 (T마다 동일)
        season_idx  : (N,) int    — 계절 인덱스 (0~3)
        pm          : (N,)        — PM10 실측값 (μg/m³)
    """

    def __init__(
        self,
        split: str = "train",
        h_scaler: Optional[StandardScaler] = None,
        lur_scaler: Optional[StandardScaler] = None,
    ):
        # ── hidden vector + PM 로드 ───────────────────────────────────────
        h_raw  = np.load(os.path.join(HIDDEN_DIR, f"h_{split}.npy"))    # (T, N, 64)
        pm_raw = np.load(os.path.join(HIDDEN_DIR, f"pm_{split}.npy"))   # (T, N)
        T, N, d = h_raw.shape

        # ── LUR 피처 (station 위치) ────────────────────────────────────────
        sta2grid   = np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"))
        time_idx   = np.load(TIME_IDX[split])
        global_idx = time_idx[STGNN_WINDOW:]   # h[t] 대응 global 타임스텝

        ndvi_all = np.load(NDVI_PATH)   # (T_all, G)
        ibi_all  = np.load(IBI_PATH)
        lc_all   = np.load(LC_PATH)     # (G, 4)
        bldg_all = np.load(BLDG_PATH)   # (G, 3)
        timestamps_all = np.load(TIMESTAMPS)

        # NDVI/IBI: 시간별 → station 위치 → 학습셋 평균으로 정적화
        ndvi_sta = ndvi_all[global_idx][:, sta2grid].mean(axis=0)   # (N,)
        ibi_sta  = ibi_all[global_idx][:, sta2grid].mean(axis=0)    # (N,)
        lc_sta   = lc_all[sta2grid]                                  # (N, 4)
        bldg_sta = bldg_all[sta2grid]                                # (N, 3)

        X_lur = np.column_stack([ndvi_sta, ibi_sta, lc_sta, bldg_sta]).astype(np.float32)
        # (N, 9) — station 공간 피처 (시간 불변)

        # ── LUR scaler ────────────────────────────────────────────────────
        if lur_scaler is None:
            self.lur_scaler = StandardScaler().fit(X_lur)
        else:
            self.lur_scaler = lur_scaler
        X_lur_norm = self.lur_scaler.transform(X_lur).astype(np.float32)

        # ── Hidden scaler ─────────────────────────────────────────────────
        h_flat = h_raw.reshape(T * N, d)
        if h_scaler is None:
            self.h_scaler = StandardScaler().fit(h_flat)
        else:
            self.h_scaler = h_scaler
        h_norm = self.h_scaler.transform(h_flat).reshape(T, N, d).astype(np.float32)

        # ── Temporal 피처 ─────────────────────────────────────────────────
        ts_split   = timestamps_all[global_idx]           # (T,)
        temp_feats = build_temporal_features(ts_split)    # (T, 9)
        seasons    = np.array([get_season(pd.to_datetime(t).month)
                                for t in ts_split], dtype=np.int64)   # (T,)
        months     = np.array([pd.to_datetime(t).month - 1
                                for t in ts_split], dtype=np.int64)   # (T,) 0-based
        hour_bins  = np.array([get_hour_bin(pd.to_datetime(t).hour)
                                for t in ts_split], dtype=np.int64)   # (T,)

        # ── Flatten to (T*N,) samples ────────────────────────────────────
        # station 인덱스
        sta_idx = np.tile(np.arange(N, dtype=np.int64), T)          # (T*N,)

        # h: (T, N, d) → (T*N, d)
        self.h        = torch.from_numpy(h_norm.reshape(T * N, d))
        # LUR: broadcast to (T*N, L)
        self.lur      = torch.from_numpy(
            np.tile(X_lur_norm, (T, 1)).astype(np.float32))         # (T*N, 9)
        # temporal: (T, 9) → (T*N, 9)
        self.temporal = torch.from_numpy(
            np.repeat(temp_feats, N, axis=0).astype(np.float32))    # (T*N, 9)
        # season / month / hour_bin: (T,) → (T*N,)
        self.season   = torch.from_numpy(np.repeat(seasons,   N).astype(np.int64))
        self.month    = torch.from_numpy(np.repeat(months,    N).astype(np.int64))
        self.hour_bin = torch.from_numpy(np.repeat(hour_bins, N).astype(np.int64))
        # station idx
        self.sta_idx  = torch.from_numpy(sta_idx)                   # (T*N,)
        # PM target
        self.pm       = torch.from_numpy(pm_raw.reshape(T * N).astype(np.float32))

        self.T, self.N, self.d = T, N, d
        # grid 추론용 보관
        self.X_lur_raw = X_lur   # (N, 9) unnormalized

    def __len__(self):
        return len(self.pm)

    def __getitem__(self, idx):
        return (self.sta_idx[idx], self.h[idx], self.lur[idx],
                self.temporal[idx], self.season[idx],
                self.month[idx], self.hour_bin[idx], self.pm[idx])
