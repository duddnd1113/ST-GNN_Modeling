"""
PseudoGridDataset

X_MODE에 따라 공간변수 구성이 달라짐:
    'all'       : NDVI + IBI + buildings + greenspace + road + river  (6차원)
    'satellite' : NDVI + IBI                                          (2차원)
    'landcover' : buildings + greenspace + road + river               (4차원)
    'none'      : 공간변수 없음                                        (0차원)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from config import (
    HIDDEN_DIR, STGNN_WINDOW,
    NDVI_PATH, IBI_PATH, LC_PATH,
    TIME_IDX,
)

# x_mode → full X (6차원) 에서의 슬라이스
X_SLICE = {
    'all':       slice(0, 6),
    'satellite': slice(0, 2),
    'landcover': slice(2, 6),
    'none':      None,
}


class PseudoGridDataset(Dataset):
    """
    Args:
        split    : 'train' | 'val' | 'test'
        x_mode   : 'all' | 'satellite' | 'landcover' | 'none'
        X_scaler : 학습 데이터로 fit된 StandardScaler (None이면 fit)
    """
    def __init__(self, split: str = "train", x_mode: str = "all", X_scaler=None):
        assert x_mode in X_SLICE, f"x_mode must be one of {list(X_SLICE)}"
        self.x_mode = x_mode

        # ── hidden vector & PM 로드 ──────────────────────────────────────────
        h_raw  = np.load(os.path.join(HIDDEN_DIR, f"h_{split}.npy"))   # [T_samp, N, H]
        pm_raw = np.load(os.path.join(HIDDEN_DIR, f"pm_{split}.npy"))  # [T_samp, N]

        coords_np   = np.load(os.path.join(HIDDEN_DIR, "coords.npy"))
        sta_to_grid = np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"))

        # ── LUR 변수 로드 ────────────────────────────────────────────────────
        ndvi_all = np.load(NDVI_PATH)   # [T_all, G]
        ibi_all  = np.load(IBI_PATH)    # [T_all, G]
        lc_all   = np.load(LC_PATH)     # [G, 4]

        # ── 타임스텝 인덱스 ───────────────────────────────────────────────────
        time_idx          = np.load(TIME_IDX[split])
        target_global_idx = time_idx[STGNN_WINDOW:]      # [T_samp]

        T_samp, N = h_raw.shape[:2]

        # ── 전체 X 구성 [T_samp, N, 6] ───────────────────────────────────────
        ndvi_sta = ndvi_all[target_global_idx][:, sta_to_grid]   # [T_samp, N]
        ibi_sta  = ibi_all[target_global_idx][:, sta_to_grid]    # [T_samp, N]
        lc_sta   = lc_all[sta_to_grid]                           # [N, 4]
        lc_rep   = np.broadcast_to(lc_sta[None], (T_samp, N, 4)).copy()

        X_full = np.concatenate([
            ndvi_sta[:, :, None],
            ibi_sta[:, :, None],
            lc_rep,
        ], axis=-1).astype(np.float32)                           # [T_samp, N, 6]

        # ── x_mode에 따라 슬라이스 ────────────────────────────────────────────
        sl = X_SLICE[x_mode]
        if sl is not None:
            X_raw  = X_full[:, :, sl]                            # [T_samp, N, x_dim]
            x_dim  = X_raw.shape[-1]
            X_2d   = X_raw.reshape(-1, x_dim)
            if X_scaler is None:
                self.X_scaler = StandardScaler()
                X_norm = self.X_scaler.fit_transform(X_2d)
            else:
                self.X_scaler = X_scaler
                X_norm = X_scaler.transform(X_2d)
            X_norm = X_norm.reshape(T_samp, N, x_dim).astype(np.float32)
            self.X = torch.from_numpy(X_norm)
        else:
            # x_mode='none': 빈 텐서
            self.X        = torch.zeros(T_samp, N, 0, dtype=torch.float32)
            self.X_scaler = None
            x_dim         = 0

        # ── 텐서 변환 ────────────────────────────────────────────────────────
        self.h      = torch.from_numpy(h_raw)
        self.pm     = torch.from_numpy(pm_raw)
        self.coords = torch.from_numpy(coords_np)
        self.T, self.N, self.x_dim = T_samp, N, x_dim

        # inference용
        self.ndvi_all          = ndvi_all
        self.ibi_all           = ibi_all
        self.lc_all            = lc_all
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
            "pm_target":      self.pm[t, s],
        }
