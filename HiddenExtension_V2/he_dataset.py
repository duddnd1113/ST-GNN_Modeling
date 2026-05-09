"""
PseudoGridDataset V2

X 구성 (x_mode별):
    'all'         : NDVI+IBI+lc(4)+bldg(3)  → 9차원
    'no_building' : NDVI+IBI+lc(4)          → 6차원  (V1 baseline)
    'satellite'   : NDVI+IBI                → 2차원
    'static_only' : lc(4)+bldg(3)           → 7차원
    'none'        : 없음                     → 0차원

wind (u, v): 항상 로드, attention score에 별도 입력
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import (
    HIDDEN_DIR, STGNN_WINDOW,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH, WIND_PATH,
    TIME_IDX, FEATURE_NAMES,
)

X_SLICE = {
    'all':         slice(0, 9),
    'no_building': slice(0, 6),
    'satellite':   slice(0, 2),
    'static_only': list(range(2, 9)),
    'none':        None,
}


def feature_names_for_mode(x_mode: str) -> list:
    sl = X_SLICE[x_mode]
    if sl is None:
        return []
    if isinstance(sl, list):
        return [FEATURE_NAMES[i] for i in sl]
    return FEATURE_NAMES[sl]


class PseudoGridDataset(Dataset):
    def __init__(self, split: str = "train", x_mode: str = "all",
                 X_scaler=None, wind_scaler=None):
        assert x_mode in X_SLICE, f"x_mode must be one of {list(X_SLICE)}"
        self.x_mode = x_mode

        # ── hidden vector & PM 로드 ──────────────────────────────────────────
        h_raw  = np.load(os.path.join(HIDDEN_DIR, f"h_{split}.npy"))   # [T, N, H]
        pm_raw = np.load(os.path.join(HIDDEN_DIR, f"pm_{split}.npy"))  # [T, N]

        coords_np   = np.load(os.path.join(HIDDEN_DIR, "coords.npy"))
        sta_to_grid = np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"))

        # ── LUR / 풍속 데이터 로드 ────────────────────────────────────────────
        ndvi_all = np.load(NDVI_PATH)   # [T_all, G]
        ibi_all  = np.load(IBI_PATH)    # [T_all, G]
        lc_all   = np.load(LC_PATH)     # [G, 4]
        bldg_all = np.load(BLDG_PATH)   # [G, 3]
        wind_all = np.load(WIND_PATH)   # [T_all, N, 2]  (u, v)

        # ── 타임스텝 인덱스 ───────────────────────────────────────────────────
        time_idx          = np.load(TIME_IDX[split])
        target_global_idx = time_idx[STGNN_WINDOW:]   # [T_samp]
        T_samp, N = h_raw.shape[:2]

        # ── wind 정규화 [T_samp, N, 2] ───────────────────────────────────────
        wind_raw = wind_all[target_global_idx].astype(np.float32)      # [T, N, 2]
        wind_2d  = wind_raw.reshape(-1, 2)
        if wind_scaler is None:
            self.wind_scaler = StandardScaler().fit(wind_2d)
        else:
            self.wind_scaler = wind_scaler
        wind_norm = self.wind_scaler.transform(wind_2d).reshape(T_samp, N, 2)
        self.wind = torch.from_numpy(wind_norm.astype(np.float32))     # [T, N, 2]

        # ── 전체 X 구성 [T_samp, N, 9] ───────────────────────────────────────
        ndvi_sta = ndvi_all[target_global_idx][:, sta_to_grid]         # [T, N]
        ibi_sta  = ibi_all[target_global_idx][:, sta_to_grid]          # [T, N]
        lc_sta   = lc_all[sta_to_grid]                                 # [N, 4]
        bldg_sta = bldg_all[sta_to_grid]                               # [N, 3]
        lc_rep   = np.broadcast_to(lc_sta[None],  (T_samp, N, 4)).copy()
        bldg_rep = np.broadcast_to(bldg_sta[None], (T_samp, N, 3)).copy()

        X_full = np.concatenate([
            ndvi_sta[:, :, None],
            ibi_sta[:, :, None],
            lc_rep,
            bldg_rep,
        ], axis=-1).astype(np.float32)                                 # [T, N, 9]

        # ── x_mode 슬라이스 & 정규화 ─────────────────────────────────────────
        sl = X_SLICE[x_mode]
        if sl is not None:
            X_raw = X_full[:, :, sl]
            x_dim = X_raw.shape[-1]
            X_2d  = X_raw.reshape(-1, x_dim)
            if X_scaler is None:
                self.X_scaler = StandardScaler().fit(X_2d)
            else:
                self.X_scaler = X_scaler
            X_norm = self.X_scaler.transform(X_2d).reshape(T_samp, N, x_dim)
            self.X = torch.from_numpy(X_norm.astype(np.float32))
        else:
            self.X        = torch.zeros(T_samp, N, 0, dtype=torch.float32)
            self.X_scaler = None
            x_dim         = 0

        # ── 텐서 변환 ────────────────────────────────────────────────────────
        self.h      = torch.from_numpy(h_raw)
        self.pm     = torch.from_numpy(pm_raw)
        self.coords = torch.from_numpy(coords_np)
        self.T, self.N, self.x_dim = T_samp, N, x_dim

        # inference용 보관
        self.ndvi_all          = ndvi_all
        self.ibi_all           = ibi_all
        self.lc_all            = lc_all
        self.bldg_all          = bldg_all
        self.wind_all          = wind_all
        self.target_global_idx = target_global_idx
        self.sta_to_grid       = sta_to_grid

    def __len__(self) -> int:
        return self.T * self.N

    def __getitem__(self, idx: int):
        t = idx // self.N
        s = idx  % self.N

        src_mask = torch.ones(self.N, dtype=torch.bool)
        src_mask[s] = False

        return {
            "h_target":       self.h[t, s],
            "h_sources":      self.h[t][src_mask],
            "coords_target":  self.coords[s],
            "coords_sources": self.coords[src_mask],
            "X_target":       self.X[t, s],
            "X_sources":      self.X[t][src_mask],
            "wind_sources":   self.wind[t][src_mask],   # [N-1, 2]  ← 신규
            "pm_target":      self.pm[t, s],
        }
