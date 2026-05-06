"""
Seoul PM10 Simulation Data Generator.

Generates synthetic hourly PM10 and meteorological data
for 40 virtual Seoul monitoring stations with realistic
spatial correlations and temporal patterns.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# 40 virtual Seoul stations (lat, lon) within [37.45~37.70, 126.80~127.18]
STATION_COORDS = [
    (37.5130, 126.8500),  # Gangseo-1
    (37.5490, 126.8700),  # Yangcheon
    (37.5170, 126.8990),  # Guro-W
    (37.5310, 126.8660),  # Yangcheon-S
    (37.5270, 126.9210),  # Dongjak
    (37.5660, 126.9100),  # Mapo
    (37.5790, 126.9250),  # Seodaemun
    (37.5850, 126.9430),  # Seodaemun-N
    (37.5690, 126.9530),  # Jongno
    (37.5660, 126.9750),  # Jung
    (37.5550, 126.9720),  # Yongsan
    (37.5440, 126.8980),  # Guro-N
    (37.4930, 126.9200),  # Gwanak-N
    (37.5140, 126.9360),  # Gwanak
    (37.5060, 126.9620),  # Seocho
    (37.4770, 126.9520),  # Seocho-S
    (37.5190, 127.0030),  # Gangnam
    (37.4640, 127.0240),  # Gangnam-S
    (37.5050, 127.0250),  # Songpa
    (37.4760, 127.0770),  # Songpa-S
    (37.5530, 127.0240),  # Seongdong
    (37.5780, 127.0390),  # Dongdaemun
    (37.5730, 126.9830),  # Seongbuk-S
    (37.5980, 126.9570),  # Seongbuk
    (37.6160, 126.9730),  # Gangbuk
    (37.6030, 126.9270),  # Eunpyeong
    (37.6380, 126.9230),  # Eunpyeong-N
    (37.6490, 127.0310),  # Dobong
    (37.6560, 127.0520),  # Nowon
    (37.6250, 127.0210),  # Dobong-Center
    (37.5980, 127.0730),  # Jungnang
    (37.6110, 127.0450),  # Jungnang-N
    (37.5630, 127.0750),  # Gwangjin
    (37.5710, 127.1150),  # Gwangjin-E
    (37.5450, 127.1050),  # Gangdong
    (37.5350, 127.0800),  # Gangdong-W
    (37.4820, 127.1230),  # Songpa-E
    (37.5130, 127.1060),  # Songpa-NE
    (37.6080, 127.1100),  # Nowon-W
    (37.4700, 126.8800),  # Gangseo-S
]


def _haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distance matrix in km.

    Args:
        coords: [N, 2] array of (lat, lon) in degrees.

    Returns:
        dist_matrix: [N, N] symmetric distance matrix in km.
    """
    R = 6371.0
    lat = np.radians(coords[:, 0])  # [N]
    lon = np.radians(coords[:, 1])  # [N]

    dlat = lat[:, None] - lat[None, :]   # [N, N]
    dlon = lon[:, None] - lon[None, :]   # [N, N]
    cos_lat = np.cos(lat)

    a = (np.sin(dlat / 2) ** 2
         + cos_lat[:, None] * cos_lat[None, :] * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def generate_synthetic_data(T: int = 8760, seed: int = 42) -> np.ndarray:
    """Generate synthetic hourly PM2.5 and meteorological data.

    Uses spatially correlated AR(1) processes + daily/seasonal cycles
    to produce plausible data for 40 Seoul monitoring stations.

    Args:
        T: Number of hourly time steps (default: 8760 = 1 year).
        seed: Random seed for reproducibility.

    Returns:
        features: np.ndarray [T, N, 6] with channels
                  [pm25, pm10, wind_speed, wind_direction, temperature, humidity].
    """
    rng = np.random.RandomState(seed)
    coords = np.array(STATION_COORDS)
    N = len(coords)

    # ── Spatial covariance kernel (exponential decay, length scale 10 km) ──
    dist_mat = _haversine_matrix(coords)
    cov = np.exp(-dist_mat / 10.0)
    cov += 1e-6 * np.eye(N)          # numerical stability
    L = np.linalg.cholesky(cov)      # [N, N] lower triangular

    # ── Time indices ──
    hours = np.arange(T) % 24                   # [T]
    days = np.arange(T) / 24.0                  # [T]

    # ── Daily cycle: rush-hour peaks at 08:00 and 19:00 ──
    daily = (20.0 * np.exp(-0.5 * ((hours - 8) / 2.0) ** 2)
             + 20.0 * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2))  # [T]

    # ── Seasonal cycle: peak in late winter / early spring (day ~60) ──
    seasonal = 20.0 * (1.0 + np.cos(2 * np.pi * (days - 60) / 365.0))  # [T]

    # ── Spatially correlated AR(1) process for PM2.5 ──
    phi = 0.92                    # temporal autocorrelation
    z = L @ rng.randn(N)          # initial spatial field
    spatial = np.zeros((T, N))
    for t in range(T):
        z = phi * z + L @ rng.randn(N)
        spatial[t] = z

    # Scale to meaningful range (σ ≈ 15 µg/m³)
    spatial = 15.0 * spatial / (spatial.std() + 1e-8)

    pm25 = 30.0 + daily[:, None] + seasonal[:, None] + spatial
    pm25 += rng.randn(T, N) * 3.0
    pm25 = np.clip(pm25, 5.0, 150.0).astype(np.float32)

    # ── PM10: strongly correlated with PM2.5 ──
    pm10 = 1.6 * pm25 + rng.randn(T, N) * 12.0
    pm10 = np.clip(pm10, 10.0, 200.0).astype(np.float32)

    # ── Wind speed: Weibull-like, [0, 10] m/s ──
    wind_speed = rng.weibull(2.0, (T, N)) * 3.5
    wind_speed = np.clip(wind_speed, 0.0, 10.0).astype(np.float32)

    # ── Wind direction: spatially coherent, [0, 360) degrees ──
    base_wd = rng.uniform(0, 360, T)   # dominant direction per hour
    wind_direction = (base_wd[:, None] + rng.randn(T, N) * 25.0) % 360.0
    wind_direction = wind_direction.astype(np.float32)

    # ── Temperature: seasonal + daily cycle, [-10, 35] °C ──
    temp_seasonal = 12.5 * np.cos(2 * np.pi * (days - 200) / 365.0)
    temp_daily = 5.0 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    temperature = (12.5 + temp_seasonal[:, None] + temp_daily[:, None]
                   + rng.randn(T, N) * 2.5)
    temperature = np.clip(temperature, -10.0, 35.0).astype(np.float32)

    # ── Humidity: inversely correlated with temperature, [20, 90] % ──
    humidity = 65.0 - 0.6 * temperature + rng.randn(T, N) * 8.0
    humidity = np.clip(humidity, 20.0, 90.0).astype(np.float32)

    # ── Stack into [T, N, 6] ──
    features = np.stack([pm25, pm10, wind_speed, wind_direction,
                         temperature, humidity], axis=-1)
    return features


class SeoulPM25Dataset(Dataset):
    """Sliding-window dataset for Seoul PM2.5 forecasting.

    Each sample is a pair (lookback window, target) where the target
    is the PM2.5 value at the next time step for all nodes.

    Args:
        node_features: torch.Tensor [T, N, 6] — all time steps.
        edge_index: torch.Tensor [2, E] — fixed directed graph.
        edge_features_all: torch.Tensor [T, E, 5] — full edge features over time.
        window: int — number of past hours to use as input (default: 12).
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features_all: torch.Tensor,
        window: int = 12,
    ):
        super().__init__()
        self.node_features = node_features        # [T, N, 6]
        self.edge_index = edge_index              # [2, E]
        self.edge_features_all = edge_features_all  # [T, E, 5]
        self.window = window

    def __len__(self) -> int:
        return len(self.node_features) - self.window

    def __getitem__(self, idx: int):
        """Return (node_window, edge_window, target).

        Returns:
            node_window: [window, N, 6]
            edge_window: [window, E, 5]
            target: [N, 1] — PM2.5 at t = idx + window
        """
        node_window = self.node_features[idx: idx + self.window]         # [T, N, 6]
        edge_window = self.edge_features_all[idx: idx + self.window]     # [T, E, 5]
        target = self.node_features[idx + self.window, :, 0:1]          # [N, 1]
        return node_window, edge_window, target
