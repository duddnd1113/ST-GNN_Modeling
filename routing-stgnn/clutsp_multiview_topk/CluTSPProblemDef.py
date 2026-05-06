import torch
import numpy as np
from sklearn.cluster import KMeans


def select_topk_cells(coords: torch.Tensor, scores: torch.Tensor, top_k: int):
    selected_scores, selected_indices = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=True)
    gather_idx = selected_indices[:, :, None].expand(-1, -1, 2)
    selected_coords = coords.gather(dim=1, index=gather_idx)
    return selected_coords, selected_scores, selected_indices


def augment_xy_features_8(xy: torch.Tensor):
    """Return 16-dim feature vector: eight symmetric (x,y) coordinate views."""
    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    return torch.cat([
        x, y,
        1 - x, y,
        x, 1 - y,
        1 - x, 1 - y,
        y, x,
        1 - y, x,
        y, 1 - x,
        1 - y, 1 - x,
    ], dim=2)


def assign_kmeans_clusters(xy: torch.Tensor, n_cluster: int, seed: int = 1234):
    """Assign cluster labels to non-depot nodes. Depot cluster is label 0."""
    if xy.dim() != 3:
        raise ValueError("xy must have shape (batch, n_nodes, 2)")
    batch_size, n_nodes, _ = xy.shape
    if n_nodes < 2:
        raise ValueError("Need at least depot + one node")
    if n_cluster > n_nodes - 1:
        raise ValueError("n_cluster cannot exceed number of non-depot nodes")

    labels = torch.zeros(batch_size, n_nodes, dtype=torch.long, device=xy.device)
    for b in range(batch_size):
        arr = xy[b, 1:, :].detach().cpu().numpy()
        km = KMeans(n_clusters=n_cluster, init="k-means++", max_iter=100, random_state=seed, n_init=10)
        labels_np = km.fit_predict(arr) + 1
        labels[b, 1:] = torch.as_tensor(labels_np, dtype=torch.long, device=xy.device)
    return labels


def build_clutsp_input(selected_coords: torch.Tensor, n_cluster: int, depot_xy=None, seed: int = 1234):
    """Build model input shape (batch, top_k+1, 17).

    selected_coords: top-k nodes, shape (batch, top_k, 2), normalized to [0,1].
    depot_xy: optional tensor/list shape (2,). Defaults to center (0.5, 0.5).
    """
    batch_size = selected_coords.size(0)
    device = selected_coords.device

    if depot_xy is None:
        depot = torch.tensor([0.5, 0.5], dtype=selected_coords.dtype, device=device).view(1, 1, 2)
    else:
        depot = torch.as_tensor(depot_xy, dtype=selected_coords.dtype, device=device).view(1, 1, 2)
    depot = depot.expand(batch_size, 1, 2)

    xy = torch.cat([depot, selected_coords], dim=1)
    aug = augment_xy_features_8(xy)
    labels = assign_kmeans_clusters(xy, n_cluster=n_cluster, seed=seed).to(selected_coords.dtype)
    return torch.cat([aug, labels[:, :, None]], dim=2)


def create_graph_data(pos_input: torch.Tensor):
    """Create global and intra-cluster graph batches as torch_geometric Data lists.

    pos_input shape: (batch, n_nodes_with_depot, 17)
    Global graph: fully connected directed graph with edge_attr=1 if same cluster else 0.
    Local graph: only same-cluster edges.
    """
    batch_size, num_node, _ = pos_input.shape
    device = pos_input.device
    node_indices = torch.arange(num_node, device=device)
    edge_index = torch.combinations(node_indices, r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    cluster_info = pos_input[:, :, -1]
    same_cluster = (torch.abs(cluster_info[:, edge_index[0]] - cluster_info[:, edge_index[1]]) < 0.5)
    edge_attr = same_cluster.long()

    data_list = []
    cluster_data_list = []
    for b in range(batch_size):
        data = Data(x=pos_input[b, :, :16], edge_index=edge_index, edge_attr=edge_attr[b].unsqueeze(1))
        cluster_edge_index = edge_index[:, same_cluster[b]]
        cluster_data = Data(x=pos_input[b, :, :16], edge_index=cluster_edge_index)
        data_list.append(data)
        cluster_data_list.append(cluster_data)
    return data_list, cluster_data_list
