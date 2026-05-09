"""공간 계산 유틸리티 (haversine 거리, 방위각 sin/cos)."""
import torch


def haversine_km_batch(target: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """단일 타겟 → N개 소스.  target:(2,)  sources:(N,2)  →  (N,)"""
    R = 6371.0
    lat1 = torch.deg2rad(target[0])
    lat2 = torch.deg2rad(sources[:, 0])
    dlon = torch.deg2rad(sources[:, 1] - target[1])
    dlat = lat2 - torch.deg2rad(target[0])
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def bearing_sincos_batch(target: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """단일 타겟 → N개 소스.  target:(2,)  sources:(N,2)  →  (N,2)"""
    lat1 = torch.deg2rad(target[0])
    lat2 = torch.deg2rad(sources[:, 0])
    dlon = torch.deg2rad(sources[:, 1] - target[1])
    x = torch.sin(dlon) * torch.cos(lat2)
    y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    b = torch.atan2(x, y)
    return torch.stack([torch.sin(b), torch.cos(b)], dim=-1)


def spatial_features(target: torch.Tensor, sources: torch.Tensor, max_dist_km: float = 50.0) -> torch.Tensor:
    """단일 타겟 버전.  →  (N, 3)  [dist_norm, sin, cos]"""
    dist   = haversine_km_batch(target, sources)
    sincos = bearing_sincos_batch(target, sources)
    return torch.cat([dist.unsqueeze(-1) / max_dist_km, sincos], dim=-1)


# ── 배치 버전 (B개 타겟 동시 처리) ─────────────────────────────────────────
def haversine_km_batch2d(targets: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """배치 타겟 → 배치 소스.  targets:(B,2)  sources:(B,N,2)  →  (B,N)"""
    R = 6371.0
    lat1 = torch.deg2rad(targets[:, 0]).unsqueeze(1)          # (B, 1)
    lat2 = torch.deg2rad(sources[:, :, 0])                    # (B, N)
    dlon = torch.deg2rad(sources[:, :, 1] - targets[:, 1].unsqueeze(1))  # (B, N)
    dlat = lat2 - lat1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def bearing_sincos_batch2d(targets: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """배치 타겟 → 배치 소스.  targets:(B,2)  sources:(B,N,2)  →  (B,N,2)"""
    lat1 = torch.deg2rad(targets[:, 0]).unsqueeze(1)          # (B, 1)
    lat2 = torch.deg2rad(sources[:, :, 0])                    # (B, N)
    dlon = torch.deg2rad(sources[:, :, 1] - targets[:, 1].unsqueeze(1))  # (B, N)
    x = torch.sin(dlon) * torch.cos(lat2)
    y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    b = torch.atan2(x, y)
    return torch.stack([torch.sin(b), torch.cos(b)], dim=-1)  # (B, N, 2)


def spatial_features_batch(targets: torch.Tensor, sources: torch.Tensor, max_dist_km: float = 50.0) -> torch.Tensor:
    """배치 버전.  targets:(B,2)  sources:(B,N,2)  →  (B,N,3)  [dist_norm, sin, cos]"""
    dist   = haversine_km_batch2d(targets, sources)            # (B, N)
    sincos = bearing_sincos_batch2d(targets, sources)          # (B, N, 2)
    return torch.cat([dist.unsqueeze(-1) / max_dist_km, sincos], dim=-1)  # (B, N, 3)
