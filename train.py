"""
Scenario-based training for the Seoul PM10 ST-GNN.

Graph modes:
    static         : bidirectional edges within distance threshold (default)
    climatological : directed edges — only i→j if mean training-period wind
                     at station i blows toward j
    soft_dynamic   : same edges as static, but edge features are zeroed out
                     at each timestep when wind blows against that edge

Usage:
    python3 train.py --scenario S1_transport_pm10
    python3 train.py --scenario S3_transport_pm10_pollutants --graph_mode climatological
    python3 train.py --scenario S1_transport_pm10 --graph_mode soft_dynamic --epochs 100
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_scenario_split, STGNNScenarioDataset
from graph_builder import (
    build_static_graph,
    build_climatological_graph,
    compute_all_dynamic_edge_features,
    get_full_edge_features,
)
from model import STGNNModel

GRAPH_MODES  = ("static", "climatological", "soft_dynamic")
GRAPH_DIR    = "graphs"
ALL_WINDOWS  = (12, 24, 48)

ALL_SCENARIOS = [
    "S1_transport_pm10",
    "S2_transport_pm10_pm10mask",
    "S3_transport_pm10_pollutants",
    "S4_transport_pm10_pollutants_pm10mask",
    "S5_transport_pm10_pollutants_allmask",
    "S6_transport_pm10_pollutants_summarymask",
    "S7_transport_pm10_weather",
    "S8_transport_pm10_rain",
    "S9_transport_pm10_weather_rain",
    "S10_transport_all_summarymask",
]


# ── Hyper-parameters ──────────────────────────────────────────────────────────

WINDOW       = 24
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
THRESHOLD_KM = 10.0
GAT_HIDDEN   = 64
GRU_HIDDEN   = 64
NUM_HEADS    = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_or_build_graph(
    graph_mode: str,
    coords,
    train_nodes: np.ndarray,
    threshold_km: float,
):
    """Load pre-built graph from graphs/ if available, otherwise build on the fly.

    soft_dynamic shares the static graph structure, so it reads from graphs/static/.
    Falls back to building when prepare_graphs.py has not been run yet.
    """
    # soft_dynamic reuses the static graph structure
    graph_key = "static" if graph_mode == "soft_dynamic" else graph_mode
    graph_path = os.path.join(GRAPH_DIR, graph_key)
    files = {
        "edge_index":    os.path.join(graph_path, "edge_index.npy"),
        "static_attr":   os.path.join(graph_path, "static_attr.npy"),
        "edge_bearings": os.path.join(graph_path, "edge_bearings.npy"),
    }

    if all(os.path.exists(p) for p in files.values()):
        edge_index    = np.load(files["edge_index"])
        static_attr   = np.load(files["static_attr"])
        edge_bearings = np.load(files["edge_bearings"])
        print(f"  Graph loaded from {graph_path}/  (E={edge_index.shape[1]})")
        return edge_index, static_attr, edge_bearings

    print(f"  Pre-built graph not found → building on the fly")
    print(f"  (Run prepare_graphs.py once to cache graphs)")
    if graph_mode == "climatological":
        return build_climatological_graph(coords, train_nodes, threshold_km)
    return build_static_graph(coords, threshold_km)


def minmax_normalize(arr: np.ndarray, a_min: np.ndarray, a_max: np.ndarray) -> np.ndarray:
    return (arr - a_min) / (a_max - a_min + 1e-8)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss downweighting imputed points.

    mask: [B, N] — 1 where imputed, 0 where real measurement.
    Loss weight = 1 - mask, so imputed points contribute 0.
    """
    weight = (1.0 - mask).unsqueeze(-1)        # [B, N, 1]
    sq_err = (pred - target) ** 2              # [B, N, 1]
    denom = weight.sum() + 1e-8
    return (sq_err * weight).sum() / denom


def evaluate(
    model: STGNNModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    pm10_min: float,
    pm10_max: float,
):
    """MAE and RMSE in original PM10 scale (µg/m³), evaluated on all points."""
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for node_w, edge_w, target, _ in loader:
            node_w  = node_w.to(device)
            edge_w  = edge_w.to(device)
            target  = target.to(device)

            pred, _ = model(node_w, edge_index, edge_w)
            pred_np   = pred.cpu().numpy()   * (pm10_max - pm10_min) + pm10_min
            target_np = target.cpu().numpy() * (pm10_max - pm10_min) + pm10_min
            all_pred.append(pred_np)
            all_true.append(target_np)

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    mae  = float(np.mean(np.abs(all_pred - all_true)))
    rmse = float(math.sqrt(np.mean((all_pred - all_true) ** 2)))
    return mae, rmse


def save_loss_history(history, out_dir):
    """Save per-epoch train/validation loss as CSV and PNG."""
    csv_path = os.path.join(out_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  Loss history CSV saved: {csv_path}")
        print("  matplotlib not installed; skipped loss curve PNG.")
        return

    png_path = os.path.join(out_dir, "loss_curve.png")
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE loss")
    ax.set_title("Training Loss Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"  Loss history CSV saved: {csv_path}")
    print(f"  Loss curve PNG saved: {png_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    scenario_name: str,
    graph_mode: str = "static",
    epochs: int = EPOCHS,
    window: int = WINDOW,
):
    assert graph_mode in GRAPH_MODES, f"graph_mode must be one of {GRAPH_MODES}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output directory: checkpoints/window_{W}/{scenario}/{graph_mode}/ ──
    out_dir = os.path.join("checkpoints", f"window_{window}", scenario_name, graph_mode)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, "best_model.pt")

    print(f"Device: {device} | Scenario: {scenario_name} | Graph mode: {graph_mode}")
    print(f"  Checkpoint: {checkpoint}")

    # ── 1. Load scenario CSV and split ───────────────────────────────────
    (train_nodes, val_nodes, test_nodes,
     train_mask, val_mask, test_mask,
     coords, feature_cols) = load_scenario_split(scenario_name)

    T_tr, N, F = train_nodes.shape
    print(f"  Stations: {N}  |  Features ({F}): {feature_cols}")
    print(f"  Train T: {T_tr}  Val T: {val_nodes.shape[0]}  Test T: {test_nodes.shape[0]}")
    print(f"  Loss mask: {'yes' if train_mask is not None else 'no'}")

    # ── 2. Load or build graph ────────────────────────────────────────────
    edge_index_np, static_attr_np, edge_bearings_np = load_or_build_graph(
        graph_mode, coords, train_nodes, THRESHOLD_KM
    )
    E = edge_index_np.shape[1]
    print(f"  Edges: {E}")

    # ── 3. Normalise node features (train stats only) ─────────────────────
    feat_min = train_nodes.min(axis=(0, 1), keepdims=True)  # [1, 1, F]
    feat_max = train_nodes.max(axis=(0, 1), keepdims=True)

    train_norm = minmax_normalize(train_nodes, feat_min, feat_max)
    val_norm   = minmax_normalize(val_nodes,   feat_min, feat_max)
    test_norm  = minmax_normalize(test_nodes,  feat_min, feat_max)

    pm10_idx = feature_cols.index("PM10")
    pm10_min = float(feat_min[0, 0, pm10_idx])
    pm10_max = float(feat_max[0, 0, pm10_idx])

    # ── 4. Dynamic edge features (computed from RAW wind values) ──────────
    def build_full_edges(raw_arr: np.ndarray) -> np.ndarray:
        T = raw_arr.shape[0]
        dyn = compute_all_dynamic_edge_features(
            edge_index_np, raw_arr, edge_bearings_np
        )                                                      # [T, E, 2]
        static_rep = np.broadcast_to(
            static_attr_np[None], (T, E, 3)
        ).copy()                                               # [T, E, 3]
        return get_full_edge_features(static_rep, dyn)         # [T, E, 5]

    train_edges = build_full_edges(train_nodes)
    val_edges   = build_full_edges(val_nodes)
    test_edges  = build_full_edges(test_nodes)

    # ── 5. Soft-dynamic edge active mask ──────────────────────────────────
    # wind_alignment is at index 3 of the 5-dim edge feature vector.
    # active[t, e] = 1 when wind at src blows toward dst, else 0.
    if graph_mode == "soft_dynamic":
        def active_mask(edges_np: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(
                (edges_np[:, :, 3] > 0).astype(np.float32)   # [T, E]
            )
        train_active = active_mask(train_edges)
        val_active   = active_mask(val_edges)
        test_active  = active_mask(test_edges)
    else:
        train_active = val_active = test_active = None

    # ── 6. Datasets & DataLoaders ─────────────────────────────────────────
    edge_index_t = torch.from_numpy(edge_index_np).long().to(device)

    def to_t(arr):
        return torch.from_numpy(arr) if arr is not None else None

    def make_ds(nodes, edges, loss_mask, edge_active):
        return STGNNScenarioDataset(
            node_features=to_t(nodes),
            edge_index=edge_index_t.cpu(),
            edge_features_all=to_t(edges),
            window=window,
            mask=to_t(loss_mask),
            pm10_idx=pm10_idx,
            edge_active_mask=edge_active,
        )

    train_ds = make_ds(train_norm, train_edges, train_mask, train_active)
    val_ds   = make_ds(val_norm,   val_edges,   val_mask,   val_active)
    test_ds  = make_ds(test_norm,  test_edges,  test_mask,  test_active)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Samples — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── 7. Model ──────────────────────────────────────────────────────────
    model = STGNNModel(
        node_dim=F,
        edge_dim=5,
        gat_hidden=GAT_HIDDEN,
        gru_hidden=GRU_HIDDEN,
        num_heads=NUM_HEADS,
        num_nodes=N,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")
    loss_history = []

    # ── 8. Training loop ──────────────────────────────────────────────────
    t_start = time.time()
    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", leave=True, unit="ep")

    for epoch in epoch_bar:
        model.train()
        train_loss, nb = 0.0, 0

        for node_w, edge_w, target, mask in train_loader:
            node_w  = node_w.to(device)
            edge_w  = edge_w.to(device)
            target  = target.to(device)
            mask    = mask.to(device)

            optimizer.zero_grad()
            pred, _ = model(node_w, edge_index_t, edge_w)
            loss = masked_mse(pred, target, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            nb += 1

        model.eval()
        val_loss, vb = 0.0, 0
        with torch.no_grad():
            for node_w, edge_w, target, mask in val_loader:
                node_w  = node_w.to(device)
                edge_w  = edge_w.to(device)
                target  = target.to(device)
                mask    = mask.to(device)
                pred, _ = model(node_w, edge_index_t, edge_w)
                val_loss += masked_mse(pred, target, mask).item()
                vb += 1

        avg_train = train_loss / nb
        avg_val = val_loss / vb
        loss_history.append({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
        })
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), checkpoint)

        epoch_bar.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}", best=f"{best_val:.4f}")

    elapsed = time.time() - t_start
    save_loss_history(loss_history, out_dir)
    print(f"\nBest Val Loss: {best_val:.6f}  |  학습 시간: {elapsed/60:.1f}분")

    # ── 9. Test evaluation ────────────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    mae, rmse = evaluate(model, test_loader, edge_index_t, device, pm10_min, pm10_max)
    print(f"Test MAE:  {mae:.2f} µg/m³")
    print(f"Test RMSE: {rmse:.2f} µg/m³")

    # 테스트 지표 저장
    import json
    metrics = {
        "scenario": scenario_name, "graph_mode": graph_mode,
        "mae": mae, "rmse": rmse,
        "best_val_loss": best_val, "elapsed_min": elapsed / 60,
        "epochs": epochs, "window": window,
        "n_features": F, "n_nodes": N, "n_edges": E,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_all(
    scenarios: list = None,
    graph_modes: list = None,
    windows: list = None,
    epochs: int = EPOCHS,
):
    """Run all window × scenario × graph_mode combinations sequentially."""
    scenarios   = scenarios   or ALL_SCENARIOS
    graph_modes = graph_modes or list(GRAPH_MODES)
    windows     = windows     or list(ALL_WINDOWS)

    combos = [(w, s, g) for w in windows for s in scenarios for g in graph_modes]
    total  = len(combos)

    print(
        f"총 {total}개 실험 시작 "
        f"({len(windows)}개 윈도우 × {len(scenarios)}개 시나리오 × {len(graph_modes)}개 그래프 모드)\n"
    )

    results = []
    combo_bar = tqdm(combos, desc="실험 전체", unit="exp")

    for window, scenario, graph_mode in combo_bar:
        combo_bar.set_description(f"w={window} | {scenario} | {graph_mode}")
        try:
            result = main(scenario, graph_mode, epochs, window)
            results.append(result)
        except Exception as e:
            tqdm.write(f"[ERROR] w={window} {scenario} × {graph_mode}: {e}")
            results.append({"window": window, "scenario": scenario,
                            "graph_mode": graph_mode,
                            "mae": None, "rmse": None, "elapsed_min": None})

    # 전체 결과 요약
    W = 6
    print("\n" + "=" * 85)
    print(f"{'윈도우':>{W}} {'시나리오':<45} {'모드':<16} {'MAE':>7} {'RMSE':>7} {'시간(분)':>8}")
    print("-" * 85)
    for r in results:
        mae_s  = f"{r['mae']:.2f}"  if r["mae"]  is not None else "ERROR"
        rmse_s = f"{r['rmse']:.2f}" if r["rmse"] is not None else "ERROR"
        time_s = f"{r['elapsed_min']:.1f}" if r["elapsed_min"] is not None else "-"
        print(f"{r.get('window', '-'):>{W}} {r['scenario']:<45} {r['graph_mode']:<16} "
              f"{mae_s:>7} {rmse_s:>7} {time_s:>8}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── 단일 실험 인자 ─────────────────────────────────────────────────────
    parser.add_argument(
        "--scenario", type=str, default="S1_transport_pm10",
        help="Scenario name (단일 실험 시 사용)"
    )
    parser.add_argument(
        "--graph_mode", type=str, default="static", choices=GRAPH_MODES,
        help="Graph mode (단일 실험 시 사용)"
    )

    # ── 전체 실험 인자 ─────────────────────────────────────────────────────
    parser.add_argument(
        "--all", action="store_true",
        help="모든 window × 시나리오 × 그래프 모드 조합 실행"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="--all 시 특정 시나리오만 선택"
    )
    parser.add_argument(
        "--graph_modes", nargs="+", default=None, choices=GRAPH_MODES,
        help="--all 시 특정 그래프 모드만 선택"
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=None,
        help=f"--all 시 특정 윈도우만 선택 (기본: {list(ALL_WINDOWS)})"
    )

    # ── 공통 인자 ─────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="단일 실험용 윈도우 크기")

    args = parser.parse_args()

    if args.all:
        run_all(args.scenarios, args.graph_modes, args.windows, args.epochs)
    else:
        main(args.scenario, args.graph_mode, args.epochs, args.window)

# # 단일 실험
# python3 train.py --scenario S1_transport_pm10 --graph_mode static --window 24

# # 전체 90개 (3 window × 10 scenario × 3 mode)
# python3 train.py --all

# # window만 ablation (30개)
# python3 train.py --all --scenario S1_transport_pm10 --graph_modes static

# # 특정 window만 선택
# python3 train.py --all --windows 12 24