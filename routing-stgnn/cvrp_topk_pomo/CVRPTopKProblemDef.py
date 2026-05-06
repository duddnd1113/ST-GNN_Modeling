import torch


def select_topk_cells(coords: torch.Tensor, scores: torch.Tensor, top_k: int):
    """Select top-k cells by score for each batch.

    Args:
        coords: (batch, total_grid_size, 2)
        scores: (batch, total_grid_size)
        top_k: int

    Returns:
        selected_coords: (batch, top_k, 2)
        selected_scores: (batch, top_k)
        selected_indices: (batch, top_k)
    """
    selected_scores, selected_indices = torch.topk(
        scores, k=top_k, dim=1, largest=True, sorted=True
    )
    gather_idx = selected_indices[:, :, None].expand(-1, -1, 2)
    selected_coords = coords.gather(dim=1, index=gather_idx)
    return selected_coords, selected_scores, selected_indices


def make_demand_from_scores(selected_scores: torch.Tensor, demand_scaler: float = None):
    """Convert selected grid scores into CVRP demand values in (0, 1].

    Demand is interpreted as cleaning workload. Higher score means more dust/priority,
    therefore larger cleaning workload. The values are normalized so that the CVRP
    capacity constraint remains stable for POMO training.
    """
    if demand_scaler is None:
        # Similar spirit to original CVRP scaling: around 10~20 nodes per route.
        demand_scaler = max(float(selected_scores.size(1)) / 2.0, 1.0)

    score_min = selected_scores.min(dim=1, keepdim=True).values
    score_max = selected_scores.max(dim=1, keepdim=True).values
    normalized = (selected_scores - score_min) / (score_max - score_min + 1e-12)

    # Keep demand positive. Range approximately [0.02, 0.20] for top_k=50.
    demand = (0.1 + 0.9 * normalized) / demand_scaler
    return demand.clamp(min=1e-6, max=1.0)


def get_random_scored_grids(batch_size: int, total_grid_size: int, device=None):
    """Generate synthetic grid-like coordinates and scores.

    This is for training/debugging. At inference, replace coords/scores with real
    LUR/ST-GNN outputs.
    """
    grid_size = int(total_grid_size ** 0.5)
    if grid_size * grid_size != total_grid_size:
        raise ValueError("total_grid_size must be a perfect square, e.g. 22500 = 150x150.")

    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    try:
        xv, yv = torch.meshgrid(x, y, indexing='ij')
    except TypeError:  # Python/PyTorch older compatibility
        xv, yv = torch.meshgrid(x, y)

    base_coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1)
    coords = base_coords[None, :, :].expand(batch_size, total_grid_size, 2).clone()

    # Synthetic score: hotspot + noise. Replace with real PM10/road-dust score.
    center = torch.tensor([0.55, 0.45], device=device)
    dist = ((coords - center) ** 2).sum(dim=2).sqrt()
    hotspot = torch.exp(-8.0 * dist)
    noise = 0.25 * torch.rand(batch_size, total_grid_size, device=device)
    scores = hotspot + noise
    return coords, scores


def augment_xy_data_by_8_fold(xy_data: torch.Tensor):
    """8-fold coordinate augmentation used in POMO."""
    x = xy_data[:, :, 0:1]
    y = xy_data[:, :, 1:2]
    return torch.cat([
        torch.cat([x, y], dim=2),
        torch.cat([1 - x, y], dim=2),
        torch.cat([x, 1 - y], dim=2),
        torch.cat([1 - x, 1 - y], dim=2),
        torch.cat([y, x], dim=2),
        torch.cat([1 - y, x], dim=2),
        torch.cat([y, 1 - x], dim=2),
        torch.cat([1 - y, 1 - x], dim=2),
    ], dim=0)
