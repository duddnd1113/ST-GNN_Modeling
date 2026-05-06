"""
Train POMO TSP on ONE synthetic static grid snapshot, test on the same snapshot,
and visualize:
  1) full grid score heatmap/scatter
  2) selected top-k nodes
  3) learned route over selected nodes

This script intentionally trains and tests on the same one snapshot.
It is for demonstration / overfitting to one instance, not generalization.

Example:
    python train_test_visualize_synthetic_snapshot.py \
        --total-grid-size 18000 \
        --top-k 50 \
        --epochs 300 \
        --log-interval 10 \
        --output-route route_output.csv \
        --output-fig route_visualization.png \
        --checkpoint checkpoint-snapshot.pt
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from TSPTopKEnvironment import TSPTopKEnv as Env
from TSPTopKModel import TSPTopKModel as Model


# ============================================================
# 1. Parameters
# ============================================================

def build_params(args):
    env_params = {
        "total_grid_size": args.total_grid_size, # default 18,000
        "top_k": args.top_k, # default 50
        "pomo_size": args.top_k,   # For TSP POMO, set pomo_size equal to top_k/problem_size.
    }

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }

    optimizer_params = {
        "lr": args.lr, # default 1e-4
        "weight_decay": args.weight_decay, # default 1e-6
    }

    return env_params, model_params, optimizer_params


# ============================================================
# 2. Generate synthetic scored grid snapshot
# ============================================================
def generate_synthetic_snapshot(args, device):
    torch.manual_seed(args.seed)

    total_grid_size = args.total_grid_size

    # infer grid size (must be square)
    grid_size = int(total_grid_size ** 0.5)
    assert grid_size * grid_size == total_grid_size, \
        "total_grid_size must be a perfect square (e.g., 10000, 16384)"

    # create regular grid
    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)

    xv, yv = torch.meshgrid(x, y, indexing='ij')

    coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1)
    coords = coords.unsqueeze(0)  # (1, N, 2)

    # random scores (no hotspots)
    scores = torch.rand(1, total_grid_size, device=device)

    df = pd.DataFrame({
        "grid_id": list(range(total_grid_size)),
        "x": coords[0, :, 0].cpu().numpy(),
        "y": coords[0, :, 1].cpu().numpy(),
        "score": scores[0].cpu().numpy(),
    })

    logging.info(
        "Generated GRID snapshot | rows=%d | score_min=%.4f | score_max=%.4f | score_mean=%.4f",
        total_grid_size,
        scores.min().item(),
        scores.max().item(),
        scores.mean().item(),
    )

    return df, list(range(total_grid_size)), coords, scores


# ============================================================
# 3. Train on exactly one snapshot
# ============================================================

def train_one_snapshot(model, env, optimizer, coords, scores, args):
    model.train()

    for epoch in range(1, args.epochs + 1):
        env.load_problems(
            batch_size=1,
            aug_factor=1, # shouldnt this be 8? to 
            coords=coords,
            scores=scores,
        )

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(1, env.pomo_size, 0), device=coords.device)

        state, reward, done = env.pre_step()

        while not done:
            selected, prob = model(state)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)

        loss = -advantage * log_prob
        loss_mean = loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)
        best_distance = -max_pomo_reward.float().mean()

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            logging.info(
                "epoch %d/%d | best_distance=%.6f | loss=%.6f",
                epoch,
                args.epochs,
                best_distance.item(),
                loss_mean.item(),
            )

    return model


# ============================================================
# 4. Test on the same snapshot
# ============================================================

def test_one_snapshot(model, env, coords, scores):
    model.eval()

    with torch.no_grad():
        env.load_problems(
            batch_size=1,
            aug_factor=8,
            coords=coords,
            scores=scores,
        )

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()

        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

    best_pomo = reward[0].argmax().item()
    best_distance = -reward[0, best_pomo].item()

    local_route = env.selected_node_list[0, best_pomo].detach().cpu().tolist()
    topk_original_indices = env.selected_indices[0].detach().cpu().tolist()
    route_original_indices = [topk_original_indices[j] for j in local_route]

    return {
        "best_pomo": best_pomo,
        "best_distance": best_distance,
        "local_route": local_route,
        "topk_original_indices": topk_original_indices,
        "route_original_indices": route_original_indices,
    }


# ============================================================
# 5. Save route CSV
# ============================================================

def save_route_csv(df, result, args):
    route_df = df.iloc[result["route_original_indices"]].copy()
    route_df.insert(0, "route_order", range(len(route_df)))

    route_df.to_csv(args.output_route, index=False)

    logging.info("Saved route CSV: %s", args.output_route)

    return route_df


# ============================================================
# 6. Visualize full grid, selected top-k, and route
# ============================================================

def visualize(df, result, args):
    x = df["x"].values
    y = df["y"].values
    s = df["score"].values

    topk_idx = result["topk_original_indices"]
    route_idx = result["route_original_indices"]

    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        x,
        y,
        c=s,
        cmap='Reds',
        s=args.grid_marker_size,
        alpha=0.75,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("score")

    ax.scatter(
        x[topk_idx],
        y[topk_idx],
        s=args.topk_marker_size,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label=f"Top-{args.top_k} selected nodes",
    )

    route_x = x[route_idx]
    route_y = y[route_idx]

    ax.plot(
        route_x,
        route_y,
        color='blue',
        linewidth=3.0,
        label="Learned route",
    )

    if len(route_idx) > 1:
        ax.plot(
            [route_x[-1], route_x[0]],
            [route_y[-1], route_y[0]],
            linewidth=2.0,
            linestyle="--",
            label="Return edge",
        )

    ax.scatter(
        route_x[0],
        route_y[0],
        s=args.start_marker_size,
        marker="*",
        label="Start",
    )

    ax.set_title(f"Synthetic One-Snapshot POMO TSP | best distance = {result['best_distance']:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=200)
    plt.close(fig)

    logging.info("Saved visualization: %s", args.output_fig)


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train/test POMO TSP on one synthetic scored grid snapshot."
    )

    parser.add_argument("--total-grid-size", type=int, default=22500)
    parser.add_argument("--top-k", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint", default="checkpoint-snapshot.pt")
    parser.add_argument("--output-route", default="route_output.csv")
    parser.add_argument("--output-fig", default="route_visualization.png")

    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device-num", type=int, default=0)

    parser.add_argument("--grid-marker-size", type=float, default=8)
    parser.add_argument("--topk-marker-size", type=float, default=80)
    parser.add_argument("--start-marker-size", type=float, default=180)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    use_cuda = args.use_cuda and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(args.cuda_device_num)
        device = torch.device("cuda", args.cuda_device_num)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        logging.info("Using CUDA device %d", args.cuda_device_num)
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
        logging.info("Using CPU")

    if args.top_k > args.total_grid_size:
        raise ValueError(
            f"top_k={args.top_k} cannot be larger than total_grid_size={args.total_grid_size}."
        )

    env_params, model_params, optimizer_params = build_params(args)

    df, ids, coords, scores = generate_synthetic_snapshot(args, device)

    env = Env(**env_params)
    model = Model(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    logging.info(
        "Training on one synthetic snapshot | total_grid_size=%d | top_k=%d | epochs=%d",
        args.total_grid_size,
        args.top_k,
        args.epochs,
    )

    train_one_snapshot(model, env, optimizer, coords, scores, args)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "env_params": env_params,
            "model_params": model_params,
            "args": vars(args),
        },
        args.checkpoint,
    )

    logging.info("Saved checkpoint: %s", args.checkpoint)

    result = test_one_snapshot(model, env, coords, scores)

    logging.info("Best POMO start: %d", result["best_pomo"])
    logging.info("Best route distance: %.6f", result["best_distance"])

    save_route_csv(df, result, args)
    visualize(df, result, args)

    print("best_distance:", result["best_distance"])
    print("best_pomo:", result["best_pomo"])
    print("saved_checkpoint:", args.checkpoint)
    print("saved_route_csv:", args.output_route)
    print("saved_visualization:", args.output_fig)


if __name__ == "__main__":
    main()