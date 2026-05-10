"""
HiddenExtension V5 — 학습 및 Ablation

단일 실험:
    python3 train.py --exp V5-hier
전체 ablation:
    python3 train.py --run_all
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CKPT_DIR, H_DIM, N_STATION, LUR_DIM, TEMPORAL_NAMES,
    LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, PATIENCE, BIAS_L2, EXPERIMENTS,
)
from dataset import V5Dataset
from model import FixedEffectPMModel


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss   = np.sum((true - true.mean()) ** 2)
    r2   = float(1 - np.sum((pred - true) ** 2) / (ss + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2)


def run_epoch(model, loader, device, optimizer=None, bias_l2=0.01):
    train = optimizer is not None
    model.train() if train else model.eval()
    preds, trues = [], []
    total_loss = 0.0
    criterion  = nn.MSELoss()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for sta_idx, h, lur, temp, season, month, hour_bin, pm in loader:
            sta_idx  = sta_idx.to(device)
            h        = h.to(device)
            lur      = lur.to(device)
            temp     = temp.to(device)
            season   = season.to(device)
            month    = month.to(device)
            hour_bin = hour_bin.to(device)
            pm       = pm.to(device)

            pred = model(sta_idx, h, lur, temp, season, month, hour_bin)
            loss = criterion(pred, pm)
            # station residual 정규화
            loss = loss + bias_l2 * model.bias_regularization_loss()

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            preds.append(pred.detach().cpu().numpy())
            trues.append(pm.cpu().numpy())

    pred_all = np.concatenate(preds)
    true_all = np.concatenate(trues)
    metrics  = compute_metrics(pred_all, true_all)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def run_experiment(exp_cfg: tuple, verbose: bool = True) -> dict:
    exp_id, use_bias, fe_mode, use_hier, mlp_hidden, dropout = exp_cfg
    use_seasonal = fe_mode  # 하위 호환
    out_dir = os.path.join(CKPT_DIR, exp_id)

    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"  [SKIP] {exp_id}")
        return json.load(open(os.path.join(out_dir, "metrics.json")))

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {exp_id}")
        print(f"  bias={use_bias}  seasonal={use_seasonal}  hier={use_hier}")
        print(f"  mlp={mlp_hidden}  dropout={dropout}")

    # ── 데이터 ──────────────────────────────────────────────────────────
    train_ds = V5Dataset("train")
    val_ds   = V5Dataset("val",  h_scaler=train_ds.h_scaler,
                                  lur_scaler=train_ds.lur_scaler)
    test_ds  = V5Dataset("test", h_scaler=train_ds.h_scaler,
                                  lur_scaler=train_ds.lur_scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 모델 ────────────────────────────────────────────────────────────
    model = FixedEffectPMModel(
        n_stations=N_STATION, h_dim=H_DIM, lur_dim=LUR_DIM,
        temporal_dim=len(TEMPORAL_NAMES),
        mlp_hidden=mlp_hidden, dropout=dropout,
        use_bias=use_bias, use_seasonal_bias=fe_mode, use_hier_lur=use_hier,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, verbose=False
    )

    # ── 학습 ────────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    no_improve   = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, optimizer, BIAS_L2)
        va = run_epoch(model, val_loader,   device)
        scheduler.step(va["mae"])

        history.append({"epoch": epoch, "tr_mae": tr["mae"], "va_mae": va["mae"]})

        if va["mae"] < best_val_mae:
            best_val_mae = va["mae"]
            no_improve   = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}")
                break

        if verbose and epoch % 10 == 0:
            print(f"  E{epoch:3d}  tr_MAE={tr['mae']:.4f}  "
                  f"va_MAE={va['mae']:.4f}  best={best_val_mae:.4f}")

    # ── 최종 평가 ────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"),
                                      map_location=device))
    te = run_epoch(model, test_loader, device)

    stgnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json")
    stgnn_mae = json.load(open(stgnn_path))["mae"] if os.path.exists(stgnn_path) else None

    elapsed = (time.time() - t0) / 60
    result  = {
        "exp_id": exp_id, "use_bias": use_bias,
        "fe_mode": fe_mode, "use_seasonal": fe_mode,
        "use_hier": use_hier, "mlp_hidden": mlp_hidden,
        "val_mae":  best_val_mae,
        "test_mae": te["mae"], "test_rmse": te["rmse"], "test_r2": te["r2"],
        "stgnn_mae": stgnn_mae, "elapsed_min": round(elapsed, 2),
    }

    if verbose:
        diff = te["mae"] - stgnn_mae if stgnn_mae else 0
        sign = "↑개선" if diff < 0 else "↓악화"
        print(f"\n  결과: test_MAE={te['mae']:.4f}  R²={te['r2']:.4f}  "
              f"{sign} {abs(diff):.4f} vs baseline  ({elapsed:.1f}min)")

    json.dump(result,  open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)

    # ── gamma 계수 저장 (LUR 해석) ──────────────────────────────────────
    if use_hier:
        import pandas as pd
        from config import LUR_NAMES
        gamma_w = model.gamma.weight.squeeze().detach().cpu().numpy()
        gamma_b = float(model.gamma.bias.item())
        pd.DataFrame({
            "feature": LUR_NAMES + ["bias"],
            "gamma":   list(gamma_w) + [gamma_b],
        }).to_csv(os.path.join(out_dir, "gamma_lur.csv"), index=False)

        resid_w = model.residual.weight.squeeze().detach().cpu().numpy()
        from dataset import V5Dataset as _
        stations = np.load(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../HiddenExtension_V1/data/hidden_vectors/stations.npy"))
        pd.DataFrame({
            "station": stations, "residual_bias": resid_w,
        }).to_csv(os.path.join(out_dir, "station_residuals.csv"), index=False)

    return result


def run_all(verbose: bool = True):
    import pandas as pd
    results = [run_experiment(cfg, verbose) for cfg in EXPERIMENTS]
    df = pd.DataFrame(results).sort_values("test_mae")

    stgnn = df["stgnn_mae"].iloc[0]
    print(f"\n{'='*65}")
    print(f"  V5 전체 결과 요약  (ST-GNN baseline: {stgnn:.4f})")
    print(f"{'='*65}")
    print(f"  {'exp_id':<20} {'test_MAE':>9} {'vs_base':>9} {'R²':>7}")
    print(f"  {'─'*50}")
    for _, r in df.iterrows():
        d = r["test_mae"] - stgnn
        print(f"  {r['exp_id']:<20} {r['test_mae']:>9.4f} "
              f"{'↑' if d<0 else '↓'}{abs(d):>7.4f}  {r['test_r2']:>7.4f}")

    df.to_csv(os.path.join(CKPT_DIR, "results_summary.csv"), index=False)
    print(f"\n  저장 → {CKPT_DIR}/results_summary.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     type=str,  default=None)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all()
    elif args.exp:
        cfg_map = {e[0]: e for e in EXPERIMENTS}
        assert args.exp in cfg_map, f"가능한 실험: {list(cfg_map)}"
        run_experiment(cfg_map[args.exp])
    else:
        parser.print_help()
