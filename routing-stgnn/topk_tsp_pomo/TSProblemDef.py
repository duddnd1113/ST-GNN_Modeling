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


def augment_xy_data_by_8_fold(xy_data: torch.Tensor):
    """8-fold data augmentation (used in POMO paper).

    Input:
        (batch, N, 2)

    Output:
        (8*batch, N, 2)
    """

    x = xy_data[:, :, 0:1]
    y = xy_data[:, :, 1:2]

    data_aug = torch.cat([
        torch.cat([ x,  y], dim=2),
        torch.cat([1-x,  y], dim=2),
        torch.cat([ x, 1-y], dim=2),
        torch.cat([1-x, 1-y], dim=2),
        torch.cat([ y,  x], dim=2),
        torch.cat([1-y,  x], dim=2),
        torch.cat([ y, 1-x], dim=2),
        torch.cat([1-y, 1-x], dim=2),
    ], dim=0)

    return data_aug