"""
Train/test a Multi-View Attention CluTSP model on ONE synthetic scored grid snapshot.

Pipeline:
    full scored grid -> select top-k cells -> k-means clusters over selected cells -> solve CluTSP

This is intentionally a one-snapshot demo, analogous to the POMO TSP snapshot script.
It overfits one static instance to make the model behavior easy to visualize.
"""

import argparse
import logging
import math
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from CluTSPProblemDef import select_topk_cells, build_clutsp_input
from MVAttnCluTSPModel import CluTSPSolver


def build_model_params(args):
    return {
        "input_dimension": 16,
        "embedding_dim": args.embedding_dim,
        "encoder_layer_num": args.encoder_layer_num,
        "qkv_dim": args.qkv_dim,
        "head_num": args.head_num,
        "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim,
    }


def generate_synthetic_snapshot(args, device):
    torch.manual_seed(args.seed)
    grid_size = int(math.sqrt(args.total_grid_size))
    if grid_size * grid_size != args.total_grid_size:
        raise ValueError("total_grid_size must be a perfect square, e.g. 10000, 22500")

    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)
    xv, yv = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1).unsqueeze(0)

    if args.score_mode == "random":
        scores = torch.rand(1, args.total_grid_size, device=device)
    else:
        center = torch.tensor([args.hotspot_x, args.hotspot_y], device=device)
        dist = ((coords - center) ** 2).sum(dim=2).sqrt()
        scores = torch.exp(-args.hotspot_strength * dist) + args.noise * torch.rand(1, args.total_grid_size, device=device)

    df = pd.DataFrame({
        "grid_id": list(range(args.total_grid_size)),
        "x": coords[0, :, 0].detach().cpu().numpy(),
        "y": coords[0, :, 1].detach().cpu().numpy(),
        "score": scores[0].detach().cpu().numpy(),
    })
    return df, coords, scores


def prepare_clutsp_instance(coords, scores, args):
    selected_coords, selected_scores, selected_indices = select_topk_cells(coords, scores, args.top_k)
    pos_input = build_clutsp_input(
        selected_coords,
        n_cluster=args.n_cluster,
        depot_xy=[args.depot_x, args.depot_y],
        seed=args.seed,
    )
    return pos_input, selected_coords, selected_scores, selected_indices


def train_one_snapshot(model, pos_input, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    for epoch in range(1, args.epochs + 1):
        cost, node_log_prob, cluster_log_prob = model(pos_input, sampling_size=args.multi_start)
        reward = -cost.view(1, args.multi_start)
        node_log_prob = node_log_prob.view(1, args.multi_start, -1)

        advantage = reward - reward.mean(dim=1, keepdim=True)
        node_loss = -advantage * node_log_prob.sum(dim=2)

        if cluster_log_prob is not None:
            cluster_log_prob = cluster_log_prob.view(1, args.multi_start, -1)
            cluster_loss = -advantage * cluster_log_prob.sum(dim=2)
            loss = (node_loss + cluster_loss).mean()
        else:
            loss = node_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        best_distance = cost.min().item()
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            logging.info("epoch %d/%d | best_distance=%.6f | loss=%.6f", epoch, args.epochs, best_distance, loss.item())


def test_one_snapshot(model, pos_input, selected_indices, args):
    model.eval()
    model.decoder.set_node_select("greedy")
    model.decoder.set_cluster_select("greedy")
    with torch.no_grad():
        cost, _, _, pi = model(pos_input, sampling_size=1, return_pi=True)
    local_route = pi[0].detach().cpu().tolist()
    topk_original_indices = selected_indices[0].detach().cpu().tolist()
    route_original_indices = [topk_original_indices[j] for j in local_route]
    return {
        "best_distance": cost[0].item(),
        "local_route": local_route,
        "topk_original_indices": topk_original_indices,
        "route_original_indices": route_original_indices,
        "cluster_labels": pos_input[0, 1:, -1].long().detach().cpu().tolist(),
    }


def save_route_csv(df, result, args):
    route_df = df.iloc[result["route_original_indices"]].copy()
    route_df.insert(0, "route_order", range(len(route_df)))
    route_df["cluster"] = [result["cluster_labels"][j] for j in result["local_route"]]
    route_df.to_csv(args.output_route, index=False)
    logging.info("Saved route CSV: %s", args.output_route)
    return route_df

def _convex_hull(points):
    """Monotonic chain convex hull. points: list of (x, y)."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def visualize(df, result, args):
    from matplotlib.patches import Polygon

    x = df["x"].values
    y = df["y"].values
    s = df["score"].values
    topk_idx = result["topk_original_indices"]
    route_idx = result["route_original_indices"]
    cluster_labels = result["cluster_labels"]

    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(x, y, c=s, cmap="Reds", s=args.grid_marker_size, alpha=0.45)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("score")

    topk_x = x[topk_idx]
    topk_y = y[topk_idx]

    # draw convex-hull boundary for each cluster
    unique_clusters = sorted(set(cluster_labels))
    for clu in unique_clusters:
        pts = [
            (topk_x[i], topk_y[i])
            for i, label in enumerate(cluster_labels)
            if label == clu
        ]

        if len(pts) >= 3:
            hull = _convex_hull(pts)
            poly = Polygon(
                hull,
                closed=True,
                fill=False,
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
            )
            ax.add_patch(poly)

            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            ax.text(cx, cy, f"C{clu}", fontsize=9, ha="center", va="center")

        elif len(pts) == 2:
            ax.plot(
                [pts[0][0], pts[1][0]],
                [pts[0][1], pts[1][1]],
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
            )

    # color selected nodes by cluster
    ax.scatter(
        topk_x,
        topk_y,
        c=cluster_labels,
        s=args.topk_marker_size,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        label=f"Top-{args.top_k} selected nodes / {args.n_cluster} clusters",
    )

    route_x = x[route_idx]
    route_y = y[route_idx]

    ax.plot(route_x, route_y, color="blue", linewidth=3.0, label="Learned CluTSP route")

    if len(route_idx) > 1:
        ax.plot(
            [route_x[-1], args.depot_x],
            [route_y[-1], args.depot_y],
            linestyle="--",
            linewidth=2.0,
            label="Return to depot",
        )
        ax.plot(
            [args.depot_x, route_x[0]],
            [args.depot_y, route_y[0]],
            linestyle="--",
            linewidth=2.0,
            label="Depot to first",
        )

    ax.scatter(
        args.depot_x,
        args.depot_y,
        s=args.start_marker_size,
        marker="*",
        label="Depot",
    )

    ax.set_title(
        f"Synthetic One-Snapshot Multi-View CluTSP | best distance = {result['best_distance']:.4f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=200)
    plt.close(fig)

    logging.info("Saved visualization: %s", args.output_fig)
def main():
    parser = argparse.ArgumentParser(description="One-snapshot top-k CluTSP demo using multi-view attention.")
    parser.add_argument("--total-grid-size", type=int, default=22500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-cluster", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--multi-start", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-mode", choices=["random", "hotspot"], default="random")
    parser.add_argument("--hotspot-x", type=float, default=0.55)
    parser.add_argument("--hotspot-y", type=float, default=0.45)
    parser.add_argument("--hotspot-strength", type=float, default=8.0)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--depot-x", type=float, default=0.5)
    parser.add_argument("--depot-y", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="checkpoint-clutsp-snapshot.pt")
    parser.add_argument("--output-route", default="clutsp_route_output.csv")
    parser.add_argument("--output-fig", default="clutsp_route_visualization.png")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device-num", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--encoder-layer-num", type=int, default=5)
    parser.add_argument("--qkv-dim", type=int, default=16)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--logit-clipping", type=float, default=10.0)
    parser.add_argument("--ff-hidden-dim", type=int, default=512)
    parser.add_argument("--grid-marker-size", type=float, default=8)
    parser.add_argument("--topk-marker-size", type=float, default=90)
    parser.add_argument("--start-marker-size", type=float, default=220)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.cuda_device_num)
        device = torch.device("cuda", args.cuda_device_num)
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)

    if args.n_cluster > args.top_k:
        raise ValueError("n_cluster must be <= top_k")

    df, coords, scores = generate_synthetic_snapshot(args, device)
    pos_input, selected_coords, selected_scores, selected_indices = prepare_clutsp_instance(coords, scores, args)

    model_params = build_model_params(args)
    model = CluTSPSolver(**model_params).to(device)

    logging.info("Training one-snapshot CluTSP | grid=%d | top_k=%d | clusters=%d | epochs=%d", args.total_grid_size, args.top_k, args.n_cluster, args.epochs)
    train_one_snapshot(model, pos_input, args)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_params": model_params,
        "args": vars(args),
    }, args.checkpoint)
    logging.info("Saved checkpoint: %s", args.checkpoint)

    result = test_one_snapshot(model, pos_input, selected_indices, args)
    save_route_csv(df, result, args)
    visualize(df, result, args)

    print("best_distance:", result["best_distance"])
    print("saved_checkpoint:", args.checkpoint)
    print("saved_route_csv:", args.output_route)
    print("saved_visualization:", args.output_fig)


if __name__ == "__main__":
    main()
