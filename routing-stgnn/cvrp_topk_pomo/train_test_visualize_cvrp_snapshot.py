"""
Train POMO CVRP on ONE synthetic static 22500-cell snapshot, test on the same snapshot,
and save route CSV + visualization.

This is a demonstration/overfitting script, not a generalization experiment.

Example:
    python train_test_visualize_cvrp_snapshot.py \
        --total-grid-size 22500 \
        --top-k 50 \
        --epochs 300 \
        --output-route cvrp_route_output.csv \
        --output-fig cvrp_route_visualization.png \
        --checkpoint checkpoint-cvrp-snapshot.pt
"""

import argparse
import logging

import pandas as pd
import torch
import matplotlib.pyplot as plt

from CVRPTopKEnvironment import CVRPTopKEnv as Env
from CVRPTopKModel import CVRPTopKModel as Model
from CVRPTopKProblemDef import get_random_scored_grids


def build_params(args):
    env_params = {
        'total_grid_size': args.total_grid_size,
        'top_k': args.top_k,
        'pomo_size': args.top_k,
        'depot_xy': [args.depot_x, args.depot_y],
    }
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }
    optimizer_params = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }
    return env_params, model_params, optimizer_params

def generate_synthetic_snapshot(args, device):
    torch.manual_seed(args.seed)

    total_grid_size = args.total_grid_size

    # infer grid size (must be square)
    grid_size = int(total_grid_size ** 0.5)
    if grid_size * grid_size != total_grid_size:
        raise ValueError("total_grid_size must be a perfect square (e.g., 22500 = 150x150)")

    # create regular grid
    x = torch.linspace(0, 1, grid_size, device=device)
    y = torch.linspace(0, 1, grid_size, device=device)

    try:
        xv, yv = torch.meshgrid(x, y, indexing='ij')
    except TypeError:
        xv, yv = torch.meshgrid(x, y)

    coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=1)
    coords = coords.unsqueeze(0)  # (1, N, 2)

    # completely random scores (no spatial bias)
    scores = torch.rand(1, total_grid_size, device=device)

    df = pd.DataFrame({
        'grid_id': list(range(total_grid_size)),
        'x': coords[0, :, 0].cpu().numpy(),
        'y': coords[0, :, 1].cpu().numpy(),
        'score': scores[0].cpu().numpy(),
    })

    logging.info(
        'Generated GRID snapshot | rows=%d | score_min=%.4f | score_max=%.4f | score_mean=%.4f',
        total_grid_size,
        scores.min().item(),
        scores.max().item(),
        scores.mean().item(),
    )

    return df, coords, scores


def train_one_snapshot(model, env, optimizer, coords, scores, args):
    model.train()
    for epoch in range(1, args.epochs + 1):
        env.load_problems(batch_size=1, aug_factor=1, coords=coords, scores=scores)
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
        loss_mean = (-advantage * log_prob).mean()

        max_pomo_reward, _ = reward.max(dim=1)
        best_distance = -max_pomo_reward.float().mean()

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            logging.info('epoch %d/%d | best_distance=%.6f | loss=%.6f',
                         epoch, args.epochs, best_distance.item(), loss_mean.item())
    return model


def test_one_snapshot(model, env, coords, scores):
    model.eval()
    with torch.no_grad():
        env.load_problems(batch_size=1, aug_factor=8, coords=coords, scores=scores)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

    flat_best = reward.reshape(-1).argmax().item()
    best_row = flat_best // env.pomo_size
    best_pomo = flat_best % env.pomo_size
    best_distance = -reward[best_row, best_pomo].item()

    local_route = env.selected_node_list[best_row, best_pomo].detach().cpu().tolist()
    topk_original_indices = env.selected_indices[best_row].detach().cpu().tolist()

    route_original_indices = []
    for node in local_route:
        if node == 0:
            route_original_indices.append(None)
        else:
            route_original_indices.append(topk_original_indices[node - 1])

    return {
        'best_row': best_row,
        'best_pomo': best_pomo,
        'best_distance': best_distance,
        'local_route': local_route,
        'topk_original_indices': topk_original_indices,
        'route_original_indices': route_original_indices,
    }


def save_route_csv(df, result, args):
    rows = []
    for order, idx in enumerate(result['route_original_indices']):
        if idx is None:
            rows.append({
                'route_order': order,
                'is_depot': True,
                'grid_id': 'DEPOT',
                'x': args.depot_x,
                'y': args.depot_y,
                'score': None,
            })
        else:
            row = df.iloc[idx].to_dict()
            row.update({'route_order': order, 'is_depot': False})
            rows.append(row)
    route_df = pd.DataFrame(rows)
    route_df.to_csv(args.output_route, index=False)
    logging.info('Saved route CSV: %s', args.output_route)
    return route_df

def visualize(df, result, args):
    x = df['x'].values
    y = df['y'].values
    s = df['score'].values
    topk_idx = result['topk_original_indices']

    depot_point = (args.depot_x, args.depot_y)

    route_points = []
    for idx in result['route_original_indices']:
        if idx is None:
            route_points.append(depot_point)
        else:
            route_points.append((x[idx], y[idx]))

    # Split route by depot visits
    subroutes = []
    current = []

    for p in route_points:
        current.append(p)

        # depot visit ends one sub-route
        if p == depot_point and len(current) > 1:
            subroutes.append(current)
            current = [depot_point]

    if len(current) > 1:
        # make sure last route returns to depot visually
        if current[-1] != depot_point:
            current.append(depot_point)
        subroutes.append(current)

    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        x, y,
        c=s,
        cmap='Reds',
        s=args.grid_marker_size,
        alpha=0.75,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('score')

    ax.scatter(
        x[topk_idx],
        y[topk_idx],
        s=args.topk_marker_size,
        marker='o',
        facecolors='none',
        edgecolors='black',
        linewidths=1.2,
        label='Top-{} selected nodes'.format(args.top_k),
    )

    # Draw each depot-to-depot trip separately.
    # No explicit colors: matplotlib cycles colors automatically.
    for i, subroute in enumerate(subroutes):
        sub_x = [p[0] for p in subroute]
        sub_y = [p[1] for p in subroute]

        ax.plot(
            sub_x,
            sub_y,
            linewidth=2.5,
            label='Trip {}'.format(i + 1),
        )

    ax.scatter(
        [args.depot_x],
        [args.depot_y],
        s=args.depot_marker_size,
        marker='s',
        label='Depot',
    )

    ax.set_title(
        'Synthetic One-Snapshot POMO CVRP | best distance = {:.4f}'.format(
            result['best_distance']
        )
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=200)
    plt.close(fig)

    logging.info('Saved visualization: %s', args.output_fig)
def main():
    parser = argparse.ArgumentParser(description='Train/test POMO CVRP on one synthetic scored grid snapshot.')
    parser.add_argument('--total-grid-size', type=int, default=22500)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--depot-x', type=float, default=0.5)
    parser.add_argument('--depot-y', type=float, default=0.5)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--output-route', type=str, default='cvrp_route_output.csv')
    parser.add_argument('--output-fig', type=str, default='cvrp_route_visualization.png')
    parser.add_argument('--checkpoint', type=str, default='checkpoint-cvrp-snapshot.pt')
    parser.add_argument('--grid-marker-size', type=float, default=5.0)
    parser.add_argument('--topk-marker-size', type=float, default=70.0)
    parser.add_argument('--depot-marker-size', type=float, default=130.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    env_params, model_params, optimizer_params = build_params(args)
    df, coords, scores = generate_synthetic_snapshot(args, device)

    env = Env(**env_params)
    model = Model(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    train_one_snapshot(model, env, optimizer, coords, scores, args)
    torch.save({'model_state_dict': model.state_dict()}, args.checkpoint)
    logging.info('Saved checkpoint: %s', args.checkpoint)

    result = test_one_snapshot(model, env, coords, scores)
    save_route_csv(df, result, args)
    visualize(df, result, args)


if __name__ == '__main__':
    main()
