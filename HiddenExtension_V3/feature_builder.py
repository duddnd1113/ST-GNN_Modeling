"""
RF 입력 피처 조합 모듈

Station-level 학습 피처와 Grid-level 추론 피처를 동일한 방식으로 생성.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, Tuple


# ── Temporal Feature ─────────────────────────────────────────────────────────

def build_temporal_features(timestamps: np.ndarray) -> np.ndarray:
    """
    타임스텝 배열 → temporal feature 행렬.
    Args:
        timestamps: (T,) datetime string 배열
    Returns:
        feat: (T, 9) float32
    """
    dt = pd.to_datetime(timestamps)

    feat = np.column_stack([
        np.sin(2 * np.pi * dt.hour / 24),          # hour_sin
        np.cos(2 * np.pi * dt.hour / 24),          # hour_cos
        np.sin(2 * np.pi * dt.month / 12),         # month_sin
        np.cos(2 * np.pi * dt.month / 12),         # month_cos
        np.sin(2 * np.pi * dt.dayofyear / 365),    # doy_sin
        np.cos(2 * np.pi * dt.dayofyear / 365),    # doy_cos
        (dt.dayofweek >= 5).astype(np.float32),    # is_weekend
        dt.month.isin([12, 1, 2]).astype(np.float32),  # is_winter
        dt.month.isin([3, 4, 5]).astype(np.float32),   # is_spring
    ]).astype(np.float32)

    return feat  # (T, 9)


# ── Spatial (LUR) Feature ────────────────────────────────────────────────────

def build_lur_features(
    ndvi: np.ndarray,       # (T, G) or (G,) if static mean
    ibi: np.ndarray,        # (T, G) or (G,)
    lc: np.ndarray,         # (G, 4)
    bldg: np.ndarray,       # (G, 3)
    population: Optional[np.ndarray] = None,  # (T, G) optional
    target_global_idx: Optional[np.ndarray] = None,   # (T,) for hourly indexing
) -> Tuple[np.ndarray, list]:
    """
    공간 피처 조합.
    Returns:
        feat       : (T, G, n_lur) — 시간×격자별 LUR 피처
        feat_names : 피처 이름 목록
    """
    T = len(target_global_idx) if target_global_idx is not None else ndvi.shape[0]
    G = lc.shape[0]

    # NDVI, IBI (시간별)
    ndvi_t = ndvi[target_global_idx] if target_global_idx is not None else ndvi  # (T, G)
    ibi_t  = ibi[target_global_idx]  if target_global_idx is not None else ibi   # (T, G)

    # 정적 피처를 시간축으로 broadcast
    lc_t   = np.broadcast_to(lc[None],   (T, G, 4)).copy()    # (T, G, 4)
    bldg_t = np.broadcast_to(bldg[None], (T, G, 3)).copy()    # (T, G, 3)

    parts = [
        ndvi_t[:, :, None],  # (T, G, 1)
        ibi_t[:, :, None],   # (T, G, 1)
        lc_t,                # (T, G, 4)
        bldg_t,              # (T, G, 3)
    ]
    names = ["NDVI", "IBI",
             "buildings_%", "greenspace_%", "road_struc_%", "river_zone_%",
             "elev_mean", "sum_area", "sum_height"]

    if population is not None:
        pop_t = population[target_global_idx] if target_global_idx is not None else population
        parts.append(pop_t[:, :, None])
        names.append("population")

    feat = np.concatenate(parts, axis=-1).astype(np.float32)  # (T, G, n_lur)
    return feat, names


# ── 메인 피처 조합 클래스 ─────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Station-level 및 Grid-level 피처를 동일한 방식으로 조합.

    학습 흐름:
        builder.fit_scalers(h_train, lur_train, temporal_train)
        X_train, y_train = builder.build_station_features(split='train')
        X_val,   y_val   = builder.build_station_features(split='val')

    추론 흐름:
        X_grid_t = builder.build_grid_features_timestep(h_grid_t, lur_grid_t, temporal_t)
    """

    def __init__(
        self,
        use_hidden:     bool  = True,
        use_lur:        bool  = True,
        use_population: bool  = False,
        use_temporal:   bool  = True,
        pca_k:          Optional[int] = None,
    ):
        self.use_hidden     = use_hidden
        self.use_lur        = use_lur
        self.use_population = use_population
        self.use_temporal   = use_temporal
        self.pca_k          = pca_k

        self.h_scaler   = StandardScaler()
        self.lur_scaler = StandardScaler()
        self.pca        = PCA(n_components=pca_k) if pca_k else None
        self._fitted    = False
        self.feature_names: list = []

    def fit_scalers(
        self,
        h_flat:       np.ndarray,   # (T*N, d) hidden
        lur_flat:     np.ndarray,   # (T*N, n_lur) LUR
    ):
        """Scaler + PCA를 train 데이터로 fit."""
        if self.use_hidden:
            self.h_scaler.fit(h_flat)
            if self.pca:
                h_norm = self.h_scaler.transform(h_flat)
                self.pca.fit(h_norm)
                var_explained = self.pca.explained_variance_ratio_.cumsum()[-1]
                print(f"  PCA k={self.pca_k}: {var_explained:.1%} 설명 분산")
        if self.use_lur:
            self.lur_scaler.fit(lur_flat)
        self._fitted = True

    def transform_hidden(self, h_flat: np.ndarray) -> np.ndarray:
        """(T*N, d) → (T*N, k or d) 정규화 + 선택적 PCA."""
        h_norm = self.h_scaler.transform(h_flat)
        if self.pca:
            return self.pca.transform(h_norm).astype(np.float32)
        return h_norm.astype(np.float32)

    def transform_lur(self, lur_flat: np.ndarray) -> np.ndarray:
        return self.lur_scaler.transform(lur_flat).astype(np.float32)

    def build_feature_names(self, h_dim: int, lur_names: list) -> list:
        names = []
        if self.use_hidden:
            k = self.pca_k if self.pca_k else h_dim
            names += ([f"pca_{i}" for i in range(k)] if self.pca_k
                      else [f"h_{i}" for i in range(k)])
        if self.use_lur:
            names += lur_names
        if self.use_temporal:
            from config import TEMPORAL_COLS
            names += TEMPORAL_COLS
        self.feature_names = names
        return names

    def assemble(
        self,
        h_flat:       Optional[np.ndarray],  # (M, d_h)
        lur_flat:     Optional[np.ndarray],  # (M, n_lur)
        temporal_flat: Optional[np.ndarray], # (M, n_t)
    ) -> np.ndarray:
        """피처 조각들을 concat. Returns (M, n_feat)."""
        parts = []
        if self.use_hidden and h_flat is not None:
            parts.append(self.transform_hidden(h_flat))
        if self.use_lur and lur_flat is not None:
            parts.append(self.transform_lur(lur_flat))
        if self.use_temporal and temporal_flat is not None:
            parts.append(temporal_flat)
        return np.concatenate(parts, axis=-1).astype(np.float32)


# ── 데이터 로더 헬퍼 ─────────────────────────────────────────────────────────

def load_split_data(split: str, window: int = 12) -> dict:
    """
    한 split(train/val/test)의 데이터를 모두 로드.
    Returns dict with: h, pm, global_idx, timestamps
    """
    from config import HIDDEN_DIR, TIME_IDX, TIMESTAMPS_PATH

    h          = np.load(f"{HIDDEN_DIR}/h_{split}.npy")       # (T, N, 64)
    pm         = np.load(f"{HIDDEN_DIR}/pm_{split}.npy")      # (T, N)
    time_idx   = np.load(TIME_IDX[split])                     # (T_total,)
    timestamps = np.load(TIMESTAMPS_PATH)                      # (T_all,)

    # hidden vector t번째 → global 타임스텝 인덱스
    global_idx = time_idx[window:]   # (T,) — h[t] 대응 global 인덱스

    assert len(global_idx) == len(h), \
        f"shape 불일치: global_idx={len(global_idx)}, h={len(h)}"

    return {
        "h":          h,           # (T, N, 64)
        "pm":         pm,          # (T, N)
        "global_idx": global_idx,  # (T,) timestamps_all 인덱스
        "timestamps": timestamps[global_idx],  # (T,) datetime string
    }


def flatten_station_data(
    data: dict,
    lur_feat: np.ndarray,    # (T, N, n_lur)
    temporal_feat: np.ndarray,  # (T, n_t)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (T, N, ...) → (T*N, ...) flatten.
    Returns: h_flat, lur_flat, temporal_flat, pm_flat
    """
    T, N, d = data["h"].shape
    h_flat       = data["h"].reshape(T * N, d)
    pm_flat      = data["pm"].reshape(T * N)
    lur_flat     = lur_feat.reshape(T * N, -1)
    temporal_flat = np.repeat(temporal_feat, N, axis=0)   # (T*N, n_t)
    return h_flat, lur_flat, temporal_flat, pm_flat
