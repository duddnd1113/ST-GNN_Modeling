"""
HiddenExtension V4 — Joint Fine-tuning 학습

Phase 1 (warmup):  ST-GNN frozen, Head 2/3만 학습 (30 epoch)
Phase 2 (joint):   ST-GNN GRU+GAT fine-tune + Head 2/3 (70 epoch)

실행:
    python3 train_joint.py
    python3 train_joint.py --phase1_only
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    STGNN_CKPT, STGNN_SCENARIO, STGNN_WINDOW, STGNN_PARAMS,
    SCENARIO_DIR, SPLIT_INFO, GRAPH_DIR,
    H_DIM, R_DIM, ATT_HIDDEN, N_HEADS_ATT, DROPOUT,
    PHASE1_EPOCHS, PHASE1_LR_HEAD,
    PHASE2_EPOCHS, PHASE2_LR_HEAD, PHASE2_LR_GRU, PHASE2_LR_GAT,
    LAMBDA_DIRECT, LAMBDA_SPATIAL,
    BATCH_SIZE, PATIENCE, WEIGHT_DECAY,
    GEO_CLUSTERS, CKPT_DIR, HIDDEN_DIR,
)
from joint_model import JointSpatialSTGNN, compute_joint_loss
from geo_loo import GeoLOOSampler


def load_stgnn_and_data():
    """기존 ST-GNN 모델 + 데이터 로드."""
    from pathlib import Path
    from model import STGNNModel
    from dataset import load_scenario_split, STGNNScenarioDataset
    from graph_builder import (build_static_graph, get_full_edge_features,
                               compute_all_dynamic_edge_features)

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    stgnn = STGNNModel(**STGNN_PARAMS)
    stgnn.load_state_dict(torch.load(STGNN_CKPT, map_location="cpu"))
    print(f"ST-GNN 로드 완료: {STGNN_CKPT}")

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    (train_nodes, val_nodes, test_nodes,
     train_mask, val_mask, test_mask,
     coords, feature_cols) = load_scenario_split(
        STGNN_SCENARIO,
        scenario_dir=Path(SCENARIO_DIR),
        split_info_path=Path(SPLIT_INFO),
    )

    # ── 그래프 ────────────────────────────────────────────────────────────
    edge_index, edge_attr_static, edge_bearings = build_static_graph(
        coords, threshold_km=10.0
    )
    E = edge_index.shape[1]

    def build_full_edges(raw_arr):
        T = raw_arr.shape[0]
        dyn = compute_all_dynamic_edge_features(edge_index, raw_arr, edge_bearings)
        static_rep = np.broadcast_to(edge_attr_static[None], (T, E, 3)).copy()
        return get_full_edge_features(static_rep, dyn)   # (T, E, 5)

    edge_feat_tr = build_full_edges(train_nodes)
    edge_feat_va = build_full_edges(val_nodes)

    def to_tensor(x):
        return torch.from_numpy(x) if x is not None else None

    train_ds = STGNNScenarioDataset(
        to_tensor(train_nodes), torch.from_numpy(edge_index.astype(np.int64)),
        to_tensor(edge_feat_tr),
        window=STGNN_WINDOW, mask=to_tensor(train_mask),
    )
    val_ds = STGNNScenarioDataset(
        to_tensor(val_nodes), torch.from_numpy(edge_index.astype(np.int64)),
        to_tensor(edge_feat_va),
        window=STGNN_WINDOW, mask=to_tensor(val_mask),
    )

    coords_tensor = torch.tensor(np.array(coords), dtype=torch.float32)
    return stgnn, train_ds, val_ds, edge_index, coords_tensor


def collate_fn(batch):
    node_w  = torch.stack([b[0] for b in batch])
    edge_w  = torch.stack([b[1] for b in batch])
    target  = torch.stack([b[2] for b in batch])
    mask    = torch.stack([b[3] for b in batch])
    return node_w, edge_w, target, mask


def evaluate(model, loader, device, coords, criterion):
    """Validation loop: forecast MAE + spatial MAE."""
    model.eval()
    sampler = GeoLOOSampler(GEO_CLUSTERS)
    d_preds, d_trues = [], []
    s_preds, s_trues = [], []

    with torch.no_grad():
        for step, (nf, ef, tgt, mask) in enumerate(loader):
            nf, ef, tgt, mask = nf.to(device), ef.to(device), tgt.to(device), mask.to(device)
            ei = loader.dataset.edge_index.to(device)

            cluster_name, target_idx = sampler.sample(step)
            out = model(nf, ei, ef, coords.to(device), target_idx, tgt)

            d_preds.append(out["pred_forecast"].squeeze(-1).cpu())
            d_trues.append(tgt.squeeze(-1).cpu())
            if out["pred_spatial"] is not None:
                s_preds.append(out["pred_spatial"].squeeze(-1).reshape(-1).cpu())
                s_trues.append(out["pm_tgt_spatial"].reshape(-1).cpu())

    d_pred = torch.cat(d_preds).numpy()
    d_true = torch.cat(d_trues).numpy()
    forecast_mae = float(np.mean(np.abs(d_pred - d_true)))

    spatial_mae = None
    if s_preds:
        s_pred = torch.cat(s_preds).numpy()
        s_true = torch.cat(s_trues).numpy()
        spatial_mae = float(np.mean(np.abs(s_pred - s_true)))

    return forecast_mae, spatial_mae


def run_phase(
    phase:       int,
    model:       JointSpatialSTGNN,
    train_loader, val_loader,
    device:      torch.device,
    coords:      torch.Tensor,
    epochs:      int,
    out_dir:     str,
    prev_best:   float = float("inf"),
):
    """단일 phase 학습 루프."""
    criterion = nn.MSELoss(reduction="none")
    sampler   = GeoLOOSampler(GEO_CLUSTERS)

    if phase == 1:
        print(f"\n{'='*60}\n  Phase 1: ST-GNN frozen, Head 2/3 학습  ({epochs} epochs)\n{'='*60}")
        model.freeze_stgnn()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR_HEAD, weight_decay=WEIGHT_DECAY,
        )
    else:
        print(f"\n{'='*60}\n  Phase 2: Joint Fine-tuning  ({epochs} epochs)\n{'='*60}")
        stgnn_groups = model.unfreeze_stgnn(PHASE2_LR_GRU, PHASE2_LR_GAT)
        head_group   = {"params": list(model.direct_head.parameters()) +
                                  list(model.spatial_head.parameters()),
                        "lr": PHASE2_LR_HEAD, "name": "heads"}
        optimizer = torch.optim.Adam(
            stgnn_groups + [head_group], weight_decay=WEIGHT_DECAY,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, verbose=True,
    )

    best_val  = prev_best
    no_improve = 0
    history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        cluster_name, target_idx = sampler.sample(epoch)

        tr_losses = {"total": [], "L_forecast": [], "L_direct": [], "L_spatial": []}
        t0 = time.time()

        for nf, ef, tgt, mask in train_loader:
            nf, ef, tgt, mask = nf.to(device), ef.to(device), tgt.to(device), mask.to(device)
            ei = train_loader.dataset.edge_index.to(device)

            out = model(nf, ei, ef, coords.to(device), target_idx, tgt)
            loss, loss_dict = compute_joint_loss(
                out, tgt, mask, LAMBDA_DIRECT, LAMBDA_SPATIAL, criterion,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in loss_dict.items():
                tr_losses[k].append(v)

        # Validation
        val_forecast_mae, val_spatial_mae = evaluate(
            model, val_loader, device, coords, criterion
        )
        scheduler.step(val_forecast_mae)

        tr_mean = {k: float(np.mean(v)) for k, v in tr_losses.items()}
        elapsed = time.time() - t0

        sp_str = f"{val_spatial_mae:.4f}" if val_spatial_mae is not None else "N/A"
        print(
            f"  [P{phase} E{epoch:3d}/{epochs}]  "
            f"tr_total={tr_mean['total']:.4f}  "
            f"tr_fc={tr_mean['L_forecast']:.4f}  "
            f"tr_sp={tr_mean['L_spatial']:.4f}  "
            f"val_fc_MAE={val_forecast_mae:.4f}  "
            f"val_sp_MAE={sp_str}  "
            f"cluster={cluster_name}  {elapsed:.1f}s"
        )

        history.append({
            "phase": phase, "epoch": epoch,
            "val_forecast_mae": val_forecast_mae,
            "val_spatial_mae":  val_spatial_mae,
            "cluster": cluster_name,
            **tr_mean,
        })

        if val_forecast_mae < best_val:
            best_val   = val_forecast_mae
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            print(f"    ★ 저장 (best val_forecast_mae={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                break

    return best_val, history


def main(phase1_only: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터 & 모델 로드 ──────────────────────────────────────────────────
    stgnn, train_ds, val_ds, edge_index, coords = load_stgnn_and_data()

    model = JointSpatialSTGNN(
        stgnn=stgnn, h_dim=H_DIM, r_dim=R_DIM,
        n_heads=N_HEADS_ATT, att_hidden=ATT_HIDDEN, dropout=DROPOUT,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # edge_index tensor를 dataset에 붙여두기 (run_phase에서 사용)
    ei_tensor = torch.from_numpy(edge_index.astype(np.int64))
    train_ds.edge_index = ei_tensor
    val_ds.edge_index   = ei_tensor

    os.makedirs(CKPT_DIR, exist_ok=True)
    all_history = []

    # ── Phase 1 ────────────────────────────────────────────────────────────
    best_val, h1 = run_phase(1, model, train_loader, val_loader,
                              device, coords, PHASE1_EPOCHS, CKPT_DIR)
    all_history.extend(h1)

    if not phase1_only:
        # ── Phase 2 ────────────────────────────────────────────────────────
        # Phase 1 best weight 로드 후 Phase 2 시작
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pt"),
                                          map_location=device))
        best_val, h2 = run_phase(2, model, train_loader, val_loader,
                                  device, coords, PHASE2_EPOCHS, CKPT_DIR,
                                  prev_best=best_val)
        all_history.extend(h2)

    # ── 최종 평가 & 저장 ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pt"),
                                      map_location=device))
    final_forecast_mae, final_spatial_mae = evaluate(
        model, val_loader, device, coords, nn.MSELoss(reduction="none")
    )

    # ST-GNN baseline 비교
    import glob
    stgnn_metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json"
    )
    stgnn_mae = None
    if os.path.exists(stgnn_metrics_path):
        with open(stgnn_metrics_path) as f:
            stgnn_mae = json.load(f)["mae"]

    result = {
        "best_val_forecast_mae": best_val,
        "final_val_forecast_mae": final_forecast_mae,
        "final_val_spatial_mae": final_spatial_mae,
        "stgnn_baseline_mae":   stgnn_mae,
        "lambda_direct":  LAMBDA_DIRECT,
        "lambda_spatial": LAMBDA_SPATIAL,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": 0 if phase1_only else PHASE2_EPOCHS,
        "r_dim": R_DIM, "n_heads": N_HEADS_ATT,
    }

    sp_str = f"{final_spatial_mae:.4f}" if final_spatial_mae is not None else "N/A"
    print(f"\n{'='*60}")
    print(f"  최종 결과")
    print(f"  Val forecast MAE : {final_forecast_mae:.4f}")
    print(f"  Val spatial  MAE : {sp_str}")
    if stgnn_mae:
        diff = final_forecast_mae - stgnn_mae
        sign = "↑개선" if diff < 0 else "↓악화"
        print(f"  ST-GNN baseline  : {stgnn_mae:.4f}  ({sign} {abs(diff):.4f})")
    print(f"{'='*60}")

    with open(os.path.join(CKPT_DIR, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump(all_history, f, indent=2)

    print(f"저장 완료 → {CKPT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_only", action="store_true")
    args = parser.parse_args()
    main(phase1_only=args.phase1_only)
