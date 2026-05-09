"""
JointHiddenExtensionModel V2 학습 루프.

단일 실험:
    python3 train.py
    python3 train.py --r_dim 16 --lam 0.2 --x_mode no_building

전체 ablation (퇴근 후 overnight):
    python3 train.py --run_all
    python3 train.py --run_all --gpu 1
"""
import os, sys, argparse, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    H_DIM, R_DIM, ATT_HIDDEN, DROPOUT, LAMBDA,
    LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, PATIENCE,
    X_MODE, LUR_MODE, ATTN_MODE,
)
from he_dataset import PseudoGridDataset, feature_names_for_mode
from model import JointHiddenExtensionModel

# ── Ablation 정의 ─────────────────────────────────────────────────────────────
# BASE: V1 best(r=8) + cross 강화(lam=0.3) + 9차원 피처
BASE = dict(r_dim=8, lam=0.5, x_mode='all', lur_mode='linear', attn_mode='full')

ABLATION_GROUPS = {
    'r_dim':     [dict(r_dim=d)     for d in [8, 16, 32]],
    'lambda':    [dict(lam=l)       for l in [0.3, 0.5, 0.7]],
    'x_mode':    [dict(x_mode=m)    for m in ['all', 'no_building', 'satellite', 'none']],
    'lur_mode':  [dict(lur_mode=m)  for m in ['linear', 'mlp']],
    'attn_mode': [dict(attn_mode=m) for m in ['full', 'spatial_only']],
}


def make_cfg(**kwargs):
    return {**BASE, **kwargs}


def cfg_to_dir(cfg):
    return (f"x{cfg['x_mode']}_attn{cfg['attn_mode']}_lur{cfg['lur_mode']}"
            f"_r{cfg['r_dim']}_lam{cfg['lam']:.1f}")


def all_ablation_cfgs():
    seen, cfgs = set(), []
    for group in ABLATION_GROUPS.values():
        for overrides in group:
            cfg = make_cfg(**overrides)
            key = cfg_to_dir(cfg)
            if key not in seen:
                seen.add(key)
                cfgs.append(cfg)
    return cfgs


# ── 학습 유틸 ─────────────────────────────────────────────────────────────────
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def compute_metrics(pred, true, x_dim):
    mae    = float(np.mean(np.abs(pred - true)))
    rmse   = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot)
    n      = len(true)
    adj_r2 = float(1 - (1 - r2) * (n - 1) / max(n - x_dim - 1, 1))
    return dict(mae=mae, rmse=rmse, r2=r2, adj_r2=adj_r2)


def run_epoch(model, loader, device, optimizer, criterion, lam, train=True):
    model.train() if train else model.eval()
    d_ls, c_ls, j_ls = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            h_tgt  = batch["h_target"].to(device)
            h_src  = batch["h_sources"].to(device)
            c_tgt  = batch["coords_target"].to(device)
            c_src  = batch["coords_sources"].to(device)
            X_tgt  = batch["X_target"].to(device)
            X_src  = batch["X_sources"].to(device)
            pm_tgt = batch["pm_target"].to(device)

            w_src  = batch["wind_sources"].to(device)
            pm_direct, pm_cross = model(h_src, c_tgt, c_src, X_tgt, X_src, w_src, h_tgt)
            ld = criterion(pm_direct, pm_tgt)
            lc = criterion(pm_cross,  pm_tgt)
            lj = lam * ld + (1 - lam) * lc

            if train:
                optimizer.zero_grad()
                lj.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            d_ls.append(ld.item()); c_ls.append(lc.item()); j_ls.append(lj.item())

    return dict(joint=np.mean(j_ls), direct=np.mean(d_ls), cross=np.mean(c_ls))


def evaluate_test(model, loader, device):
    model.eval()
    d_preds, c_preds, trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            pd_, pc = model(
                batch["h_sources"].to(device), batch["coords_target"].to(device),
                batch["coords_sources"].to(device), batch["X_target"].to(device),
                batch["X_sources"].to(device), batch["wind_sources"].to(device),
                batch["h_target"].to(device),
            )
            d_preds.append(pd_.cpu().numpy())
            c_preds.append(pc.cpu().numpy())
            trues.append(batch["pm_target"].numpy())
    return np.concatenate(d_preds), np.concatenate(c_preds), np.concatenate(trues)


def load_stgnn_metrics():
    path = os.path.join(os.path.dirname(__file__),
                        "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── LUR 해석 저장 ─────────────────────────────────────────────────────────────
def save_lur_interpretation(model, cfg, out_dir):
    """lur_mode='linear'일 때 β 계수 JSON + 막대 그래프 저장."""
    if cfg['lur_mode'] != 'linear' or cfg['x_mode'] == 'none':
        return

    feat_names = feature_names_for_mode(cfg['x_mode'])
    coefs = model.beta.weight.squeeze().detach().cpu().numpy()  # [x_dim]

    result = {n: round(float(c), 6) for n, c in zip(feat_names, coefs)}
    with open(os.path.join(out_dir, "lur_coefficients.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # β 계수 막대 그래프
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in result.values()]
    fig, ax = plt.subplots(figsize=(max(6, len(feat_names) * 0.9), 4))
    bars = ax.bar(list(result.keys()), list(result.values()), color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("LUR 계수 (β)", fontsize=10)
    ax.set_title(f"LUR 회귀계수\n[{cfg_to_dir(cfg)}]", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    for bar, val in zip(bars, result.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 * (1 if val >= 0 else -1),
                f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lur_coefficients.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_comparison(direct_m, cross_m, stgnn, cfg):
    w = 22
    print(f"\n{'':=<70}")
    print(f"  [{cfg_to_dir(cfg)}]")
    print(f"  {'모델':>{w}}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'Adj.R²':>7}")
    print(f"  {'-'*64}")
    if stgnn:
        print(f"  {'ST-GNN (baseline)':>{w}}  "
              f"{stgnn['mae']:>7.2f}  {stgnn['rmse']:>7.2f}  {'N/A':>7}  {'N/A':>7}")
    print(f"  {'Direct':>{w}}  "
          f"{direct_m['mae']:>7.2f}  {direct_m['rmse']:>7.2f}  "
          f"{direct_m['r2']:>7.4f}  {direct_m['adj_r2']:>7.4f}")
    print(f"  {'Cross-Attn (LOO)':>{w}}  "
          f"{cross_m['mae']:>7.2f}  {cross_m['rmse']:>7.2f}  "
          f"{cross_m['r2']:>7.4f}  {cross_m['adj_r2']:>7.4f}")
    print(f"{'':=<70}")


# ── 단일 실험 ─────────────────────────────────────────────────────────────────
def run(cfg: dict, device: torch.device):
    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints", cfg_to_dir(cfg))
    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"  [SKIP] 이미 완료: {cfg_to_dir(cfg)}")
        with open(os.path.join(out_dir, "metrics.json")) as f:
            return json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'─'*50}\n  실험: {cfg_to_dir(cfg)}")

    train_ds = PseudoGridDataset("train", x_mode=cfg['x_mode'])
    val_ds   = PseudoGridDataset("val",   x_mode=cfg['x_mode'],
                                 X_scaler=train_ds.X_scaler,
                                 wind_scaler=train_ds.wind_scaler)
    test_ds  = PseudoGridDataset("test",  x_mode=cfg['x_mode'],
                                 X_scaler=train_ds.X_scaler,
                                 wind_scaler=train_ds.wind_scaler)

    kw = dict(collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)

    x_dim = train_ds.x_dim
    print(f"  x_dim={x_dim}  r_dim={cfg['r_dim']}  λ={cfg['lam']}  "
          f"lur={cfg['lur_mode']}  attn={cfg['attn_mode']}")

    model = JointHiddenExtensionModel(
        h_dim=H_DIM, x_dim=x_dim, r_dim=cfg['r_dim'],
        att_hidden=ATT_HIDDEN, dropout=DROPOUT,
        lur_mode=cfg['lur_mode'], attn_mode=cfg['attn_mode'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    best_val, patience_cnt = float("inf"), 0
    history = []

    bar = tqdm(range(1, EPOCHS + 1), desc="Epoch", leave=False, unit="ep")
    for epoch in bar:
        tr = run_epoch(model, train_loader, device, optimizer, criterion, cfg['lam'], train=True)
        va = run_epoch(model, val_loader,   device, optimizer, criterion, cfg['lam'], train=False)
        history.append({"epoch": epoch,
                        **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})
        bar.set_postfix(tr=f"{tr['joint']:.3f}", va=f"{va['joint']:.3f}", best=f"{best_val:.3f}")

        if va["joint"] < best_val:
            best_val = va["joint"]
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                tqdm.write(f"  Early stopping @ epoch {epoch}")
                break

    # ── 평가 ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    d_pred, c_pred, true = evaluate_test(model, test_loader, device)
    direct_m = compute_metrics(d_pred, true, x_dim)
    cross_m  = compute_metrics(c_pred, true, x_dim)
    stgnn    = load_stgnn_metrics()
    print_comparison(direct_m, cross_m, stgnn, cfg)

    # ── LUR 해석 저장 ─────────────────────────────────────────────────────────
    save_lur_interpretation(model, cfg, out_dir)

    result = {**cfg, "x_dim": x_dim, "best_val": best_val,
              **{f"direct_{k}": v for k, v in direct_m.items()},
              **{f"cross_{k}":  v for k, v in cross_m.items()},
              "stgnn_mae": stgnn["mae"] if stgnn else None}

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return result


# ── 전체 Ablation 요약 ────────────────────────────────────────────────────────
def print_ablation_summary(results):
    sep = "=" * 80
    print(f"\n{sep}\n  V2 Ablation 전체 요약\n{sep}")
    print(f"  {'설정':<50}  {'Direct MAE':>10}  {'Cross MAE':>9}  {'Cross R²':>8}")
    print(f"  {'-'*76}")
    for r in sorted(results, key=lambda x: x.get('cross_mae', 9999)):
        name = cfg_to_dir(r)
        d  = r.get('direct_mae', float('nan'))
        c  = r.get('cross_mae',  float('nan'))
        r2 = r.get('cross_r2',   float('nan'))
        print(f"  {name:<50}  {d:>10.3f}  {c:>9.3f}  {r2:>8.4f}")
    print(sep)


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r_dim",     type=int,   default=BASE['r_dim'])
    parser.add_argument("--lam",       type=float, default=BASE['lam'])
    parser.add_argument("--x_mode",    type=str,   default=BASE['x_mode'],
                        choices=['all', 'no_building', 'satellite', 'static_only', 'none'])
    parser.add_argument("--lur_mode",  type=str,   default=BASE['lur_mode'],
                        choices=['linear', 'mlp'])
    parser.add_argument("--attn_mode", type=str,   default=BASE['attn_mode'],
                        choices=['full', 'spatial_only'])
    parser.add_argument("--run_all",   action="store_true",
                        help="전체 ablation 실험 순차 실행 (overnight)")
    parser.add_argument("--gpu",       type=int,   default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"Device: GPU {idx} — {torch.cuda.get_device_name(idx)}")
    else:
        print("Device: CPU")

    if args.run_all:
        cfgs = all_ablation_cfgs()
        print(f"\n총 {len(cfgs)}개 ablation 실험 실행 예정")
        for i, cfg in enumerate(cfgs, 1):
            print(f"\n[{i}/{len(cfgs)}]", end=" ")
        results = [run(cfg, device) for cfg in cfgs]
        print_ablation_summary(results)
    else:
        cfg = make_cfg(
            r_dim=args.r_dim, lam=args.lam,
            x_mode=args.x_mode, lur_mode=args.lur_mode, attn_mode=args.attn_mode,
        )
        run(cfg, device)
