"""
Geographic Cluster LOO (Leave-One-Out) 전략

V1/V2의 random LOO와 달리, 지리적으로 인접한 station 군집을 통째로 mask.
→ 모델이 "한 구역에 관측소가 없는 상황"을 학습
→ 실제 grid 추론과 더 유사한 조건
"""
import numpy as np
import torch
from typing import List, Dict, Tuple


def build_loo_batch(
    h: torch.Tensor,             # (B, N, d)
    pm: torch.Tensor,            # (B, N)
    target_idx: List[int],       # mask할 target station 인덱스
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    target_idx 위치를 제외한 context로 target PM 예측을 위한 배치 구성.

    Returns:
        h_ctx    : (B, N_ctx, d) context station hidden
        h_tgt    : (B, N_tgt, d) target station hidden  (ground truth용)
        pm_tgt   : (B, N_tgt)    target PM (정답)
        ctx_mask : (N,) bool     True = context station
    """
    N = h.size(1)
    ctx_mask = torch.ones(N, dtype=torch.bool, device=h.device)
    ctx_mask[target_idx] = False

    h_ctx  = h[:, ctx_mask, :]         # (B, N_ctx, d)
    h_tgt  = h[:, ~ctx_mask, :]        # (B, N_tgt, d)
    pm_tgt = pm[:, ~ctx_mask]          # (B, N_tgt)

    return h_ctx, h_tgt, pm_tgt, ctx_mask


class GeoLOOSampler:
    """
    매 epoch마다 다른 geographic cluster를 target으로 선택.

    사용법:
        sampler = GeoLOOSampler(GEO_CLUSTERS)
        for epoch in range(N_epochs):
            target_idx = sampler.sample(epoch)
            h_ctx, h_tgt, pm_tgt, _ = build_loo_batch(h, pm, target_idx)
    """

    def __init__(self, clusters: Dict[str, List[int]], seed: int = 42):
        self.clusters   = clusters
        self.names      = list(clusters.keys())
        self.rng        = np.random.default_rng(seed)

    def sample(self, epoch: int = None) -> Tuple[str, List[int]]:
        """cluster 이름과 해당 station 인덱스 반환."""
        name = self.names[epoch % len(self.names)] if epoch is not None \
               else self.rng.choice(self.names)
        return name, self.clusters[name]

    def all_clusters(self) -> Dict[str, List[int]]:
        return self.clusters

    def n_clusters(self) -> int:
        return len(self.names)


def compute_coords_for_cluster(
    coords: np.ndarray,    # (N, 2)
    target_idx: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """target/context 좌표 분리."""
    N = len(coords)
    ctx_mask = np.ones(N, dtype=bool)
    ctx_mask[target_idx] = False
    return coords[ctx_mask], coords[~ctx_mask]
