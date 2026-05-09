"""
Grid-level hidden vector 생성 (Wind-aware IDW)

Station hidden h(T, N, d) → Grid hidden h(T, G, d)

지원 방법:
  - 'wind'    : Wind-aware IDW (풍향/거리 결합)
  - 'idw'     : Inverse Distance Weighting (거리만)
  - 'nearest' : Nearest station assignment
"""
import numpy as np
from sklearn.preprocessing import normalize


def _haversine_distance(coord_a: np.ndarray, coord_b: np.ndarray) -> np.ndarray:
    """
    위경도 좌표 간 거리 계산 (km).
    coord_a: (..., 2) [lat, lon]
    coord_b: (..., 2) [lat, lon]
    """
    R = 6371.0
    lat1, lon1 = np.radians(coord_a[..., 0]), np.radians(coord_a[..., 1])
    lat2, lon2 = np.radians(coord_b[..., 0]), np.radians(coord_b[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_static_weights(
    coords_station: np.ndarray,  # (N, 2) [lat, lon]
    coords_grid: np.ndarray,     # (G, 2) [lat, lon]
    sigma_d: float = 0.05,
    k: int = 10,
) -> tuple:
    """
    정적(거리 기반) 가중치 사전 계산.

    Returns:
        dist_km  : (G, N) 거리 (km)
        w_static : (G, N) 정규화된 거리 가중치 (top-k만 nonzero)
        topk_idx : (G, k) 각 grid의 top-k station 인덱스
    """
    G, N = len(coords_grid), len(coords_station)

    # (G, N) 거리 행렬
    dist_km = np.zeros((G, N), dtype=np.float32)
    # 배치로 계산 (메모리 절약)
    BATCH = 500
    for g_start in range(0, G, BATCH):
        g_end = min(g_start + BATCH, G)
        cg = coords_grid[g_start:g_end, None, :]    # (batch, 1, 2)
        cs = coords_station[None, :, :]              # (1, N, 2)
        dist_km[g_start:g_end] = _haversine_distance(cg, cs)

    # top-k station 인덱스
    topk_idx = np.argsort(dist_km, axis=1)[:, :k]   # (G, k)

    # 거리 가중치 (Gaussian kernel, top-k만)
    w_static = np.zeros((G, N), dtype=np.float32)
    for g in range(G):
        idx = topk_idx[g]
        d   = dist_km[g, idx]
        w_static[g, idx] = np.exp(-d / (sigma_d * 111))  # sigma_d degree → km

    # 행 정규화
    row_sum = w_static.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    w_static /= row_sum

    return dist_km, w_static, topk_idx


def compute_wind_weights(
    wind_t: np.ndarray,           # (N, 2) u,v at time t
    coords_station: np.ndarray,   # (N, 2)
    coords_grid: np.ndarray,      # (G, 2)
    w_static: np.ndarray,         # (G, N) 정적 거리 가중치
    topk_idx: np.ndarray,         # (G, k)
    sigma_w: float = 1.0,
) -> np.ndarray:
    """
    풍향 정렬 가중치를 거리 가중치에 결합.
    바람이 station → grid 방향으로 불수록 가중치 높임.

    Returns:
        w : (G, N) 결합 가중치 (정규화됨)
    """
    G, k = topk_idx.shape
    w = w_static.copy()

    # 풍속이 모두 0이면 거리 가중치 그대로 반환
    if np.all(wind_t == 0):
        return w

    wind_norm = np.linalg.norm(wind_t, axis=-1, keepdims=True) + 1e-8
    wind_unit = wind_t / wind_norm  # (N, 2)

    for g in range(G):
        idx = topk_idx[g]
        # 격자 좌표를 lat/lon 비율로 변환하여 방향 벡터 계산
        diff = coords_grid[g] - coords_station[idx]   # (k, 2) [Δlat, Δlon]
        # Δlat→north/south, Δlon→east/west (y, x 순서 맞춤)
        diff_xy = np.stack([diff[:, 1], diff[:, 0]], axis=-1)  # (k, 2) [Δlon, Δlat]
        diff_norm = np.linalg.norm(diff_xy, axis=-1, keepdims=True) + 1e-8
        dir_unit  = diff_xy / diff_norm  # (k, 2)

        # 풍향 정렬: dot(wind[i], direction[i→g])
        alignment = (wind_unit[idx] * dir_unit).sum(axis=-1)   # (k,)
        wind_boost = np.exp(alignment / sigma_w)               # (k,)

        w[g, idx] = w_static[g, idx] * wind_boost

    # 행 정규화
    row_sum = w.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    w /= row_sum
    return w


def get_grid_hidden_batch(
    h_batch: np.ndarray,       # (B, N, d) hidden vectors for B timesteps
    w_batch: np.ndarray,       # (B, G, N) weights for B timesteps
) -> np.ndarray:
    """
    배치 행렬곱으로 그리드 hidden 계산.
    Returns: (B, G, d)
    """
    # (B, G, N) @ (B, N, d) → (B, G, d)
    return np.einsum("bgn,bnd->bgd", w_batch, h_batch).astype(np.float32)


class GridHiddenGenerator:
    """
    Station hidden vector를 Grid hidden vector로 변환.

    사용 예:
        gen = GridHiddenGenerator(coords_station, coords_grid, method='wind')
        gen.fit()  # 정적 가중치 사전 계산
        h_grid_t = gen.transform_timestep(h_t, wind_t)
    """

    def __init__(
        self,
        coords_station: np.ndarray,  # (N, 2)
        coords_grid: np.ndarray,     # (G, 2)
        method: str = "wind",        # 'wind' | 'idw' | 'nearest'
        sigma_d: float = 0.05,
        sigma_w: float = 1.0,
        k: int = 10,
    ):
        self.coords_station = coords_station
        self.coords_grid    = coords_grid
        self.method         = method
        self.sigma_d        = sigma_d
        self.sigma_w        = sigma_w
        self.k              = k
        self._fitted        = False

    def fit(self):
        """정적 가중치 사전 계산 (한 번만 실행)."""
        print(f"GridHiddenGenerator.fit() [{self.method}]  "
              f"G={len(self.coords_grid)}, N={len(self.coords_station)}, k={self.k}")
        self.dist_km, self.w_static, self.topk_idx = compute_static_weights(
            self.coords_station, self.coords_grid,
            sigma_d=self.sigma_d, k=self.k,
        )
        if self.method == "nearest":
            # nearest: 1개 station에만 weight=1
            nearest_idx = np.argmin(self.dist_km, axis=1)  # (G,)
            self.w_nearest = np.zeros_like(self.w_static)
            self.w_nearest[np.arange(len(self.coords_grid)), nearest_idx] = 1.0
        self._fitted = True
        print("  fit 완료.")

    def transform_timestep(
        self,
        h_t: np.ndarray,      # (N, d)
        wind_t: np.ndarray,   # (N, 2) u,v — wind 방법에서만 사용
    ) -> np.ndarray:
        """단일 타임스텝 grid hidden 계산. Returns (G, d)."""
        assert self._fitted, "fit()을 먼저 호출하세요"

        if self.method == "nearest":
            return self.w_nearest @ h_t   # (G, N) @ (N, d) = (G, d)

        elif self.method == "idw":
            return self.w_static @ h_t

        else:  # wind
            w = compute_wind_weights(
                wind_t, self.coords_station, self.coords_grid,
                self.w_static, self.topk_idx, self.sigma_w,
            )
            return w @ h_t   # (G, d)

    def transform_all(
        self,
        h: np.ndarray,      # (T, N, d)
        wind: np.ndarray,   # (T, N, 2) — wind 방법에서만 사용
        batch_size: int = 100,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        전체 타임스텝 grid hidden 계산.
        Returns (T, G, d)  — 메모리 주의: T×G×d×4 bytes
        """
        T, N, d = h.shape
        G = len(self.coords_grid)
        h_grid = np.zeros((T, G, d), dtype=np.float32)

        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)
            if verbose and t_start % 500 == 0:
                print(f"  [{t_start}/{T}]", end="\r")
            for t in range(t_start, t_end):
                h_grid[t] = self.transform_timestep(
                    h[t], wind[t] if wind is not None else None
                )
        if verbose:
            print(f"  transform_all 완료: {h_grid.shape}")
        return h_grid
