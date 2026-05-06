"""
학습된 JointHiddenExtensionModel V2로 grid 단위 PM10 분포를 추론합니다.

출력:
    checkpoints/{cfg}/grid_pm10_predictions.npy   [T_test, G]
    checkpoints/{cfg}/grid_pm10_mean.csv
    checkpoints/{cfg}/grid_pm10_mean.png          서울 전역 시간평균 지도
    checkpoints/{cfg}/grid_pm10_snapshots.png     저/고농도 스냅샷 4장
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    H_DIM, R_DIM, ATT_HIDDEN, DROPOUT,
    HIDDEN_DIR, GRID_CSV_PATH,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH,
    TIME_IDX, STGNN_WINDOW,
    X_MODE, LUR_MODE, ATTN_MODE, LAMBDA,
)
from he_dataset import PseudoGridDataset
from model import JointHiddenExtensionModel

_CMAP       = "YlOrRd"
_S_SIZE     = 3
_WHO_ANNUAL = 15.0


def cfg_to_dir(r_dim, lam, x_mode, lur_mode, attn_mode):
    return f"x{x_mode}_attn{attn_mode}_lur{lur_mode}_r{r_dim}_lam{lam:.1f}"


def _scatter_map(ax, lon, lat, values, vmin, vmax, title, cbar_label):
    sc = ax.scatter(lon, lat, c=values, cmap=_CMAP,
                    vmin=vmin, vmax=vmax, s=_S_SIZE, linewidths=0, alpha=0.9)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    return sc


def _plot_mean_map(grid_basic, mean_pm, out_dir, cfg_label):
    lon, lat = grid_basic["lon"].values, grid_basic["lat"].values
    vmin, vmax = float(mean_pm.min()), float(np.percentile(mean_pm, 99))

    fig, ax = plt.subplots(figsize=(8, 7))
    _scatter_map(ax, lon, lat, mean_pm, vmin, vmax,
                 title=f"서울 PM10 시간평균 분포\n[{cfg_label}]",
                 cbar_label="PM10 (μg/m³)")
    ax.scatter([], [], c="none", label=f"WHO 연평균 기준: {_WHO_ANNUAL} μg/m³")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "grid_pm10_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def _plot_snapshots(grid_basic, all_pm, target_global_idx, out_dir):
    lon, lat = grid_basic["lon"].values, grid_basic["lat"].values
    hourly_mean = all_pm.mean(axis=1)
    pcts   = [10, 50, 90, 99]
    titles = ["저농도 (p10)", "중간 (p50)", "고농도 (p90)", "최고 (p99)"]
    idxs   = [int(np.argmin(np.abs(hourly_mean - np.percentile(hourly_mean, p)))) for p in pcts]
    vmin, vmax = float(all_pm.min()), float(np.percentile(all_pm, 99))

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("서울 PM10 농도 스냅샷 (테스트 기간)", fontsize=12, fontweight="bold")
    for ax, idx, title in zip(axes.flat, idxs, titles):
        _scatter_map(ax, lon, lat, all_pm[idx], vmin, vmax,
                     title=f"{title}  [t={target_global_idx[idx]}]",
                     cbar_label="PM10 (μg/m³)")
    fig.tight_layout()
    path = os.path.join(out_dir, "grid_pm10_snapshots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def run(r_dim: int = R_DIM, lam: float = LAMBDA,
        x_mode: str = X_MODE, lur_mode: str = LUR_MODE, attn_mode: str = ATTN_MODE):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = cfg_to_dir(r_dim, lam, x_mode, lur_mode, attn_mode)
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", ckpt_dir, "best_model.pt")

    train_ds = PseudoGridDataset("train", x_mode=x_mode)
    X_scaler = train_ds.X_scaler
    x_dim    = train_ds.X.shape[-1]

    model = JointHiddenExtensionModel(
        h_dim=H_DIM, x_dim=x_dim, r_dim=r_dim,
        att_hidden=ATT_HIDDEN, dropout=DROPOUT,
        lur_mode=lur_mode, attn_mode=attn_mode,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    coords_sta  = train_ds.coords.to(device)
    sta_to_grid = np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy"))
    h_test      = torch.from_numpy(np.load(os.path.join(HIDDEN_DIR, "h_test.npy")))

    grid_basic  = pd.read_csv(GRID_CSV_PATH)
    grid_coords = torch.from_numpy(
        grid_basic[["lat", "lon"]].values.astype(np.float32)
    ).to(device)
    G = len(grid_basic)

    ndvi_all = np.load(NDVI_PATH)   # [T_all, G]
    ibi_all  = np.load(IBI_PATH)    # [T_all, G]
    lc_all   = np.load(LC_PATH)     # [G, 4]
    bldg_all = np.load(BLDG_PATH)   # [G, 3]

    # 측정소 정적 피처
    lc_sta   = lc_all[sta_to_grid]    # [N, 4]
    bldg_sta = bldg_all[sta_to_grid]  # [N, 3]

    from he_dataset import X_SLICE
    sl = X_SLICE[x_mode]

    time_idx_test      = np.load(TIME_IDX["test"])
    target_global_idx  = time_idx_test[STGNN_WINDOW:]
    T = len(target_global_idx)
    print(f"  Grid: {G}개  |  Test 타임스텝: {T}개")

    all_pm = np.zeros((T, G), dtype=np.float32)

    def _build_X(ndvi_t, ibi_t, lc, bldg, scaler, sl_):
        X_full = np.concatenate([
            ndvi_t[:, None], ibi_t[:, None], lc, bldg
        ], axis=-1).astype(np.float32)            # [N or G, 9]
        X_sl = X_full[:, sl_] if sl_ is not None else X_full[:, :0]
        if scaler is not None and X_sl.shape[-1] > 0:
            X_sl = scaler.transform(X_sl).astype(np.float32)
        return torch.from_numpy(X_sl)

    with torch.no_grad():
        for i, global_t in enumerate(tqdm(target_global_idx, desc="Grid inference")):
            h_t = h_test[i].to(device)

            X_sta_t = _build_X(
                ndvi_all[global_t][sta_to_grid], ibi_all[global_t][sta_to_grid],
                lc_sta, bldg_sta, X_scaler, sl
            ).to(device)

            X_grid_t = _build_X(
                ndvi_all[global_t], ibi_all[global_t],
                lc_all, bldg_all, X_scaler, sl
            ).to(device)

            pred = model(
                h_t.unsqueeze(0).expand(G, -1, -1),
                grid_coords,
                coords_sta.unsqueeze(0).expand(G, -1, -1),
                X_grid_t,
                X_sta_t.unsqueeze(0).expand(G, -1, -1),
            )
            all_pm[i] = pred.cpu().numpy()

    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints", ckpt_dir)
    np.save(os.path.join(out_dir, "grid_pm10_predictions.npy"), all_pm)

    mean_pm = all_pm.mean(axis=0)
    pd.DataFrame({
        "CELL_ID":   grid_basic["CELL_ID"],
        "lat":       grid_basic["lat"],
        "lon":       grid_basic["lon"],
        "pm10_mean": mean_pm,
    }).to_csv(os.path.join(out_dir, "grid_pm10_mean.csv"), index=False)

    _plot_mean_map(grid_basic, mean_pm, out_dir, ckpt_dir)
    _plot_snapshots(grid_basic, all_pm, target_global_idx, out_dir)

    print(f"\n  저장 완료 → {out_dir}/")
    print(f"  grid_pm10_predictions.npy  {all_pm.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--r_dim",     type=int,   default=R_DIM)
    parser.add_argument("--lam",       type=float, default=LAMBDA)
    parser.add_argument("--x_mode",    type=str,   default=X_MODE,
                        choices=['all', 'no_building', 'satellite', 'static_only', 'none'])
    parser.add_argument("--lur_mode",  type=str,   default=LUR_MODE,
                        choices=['linear', 'mlp'])
    parser.add_argument("--attn_mode", type=str,   default=ATTN_MODE,
                        choices=['full', 'spatial_only'])
    args = parser.parse_args()
    run(r_dim=args.r_dim, lam=args.lam,
        x_mode=args.x_mode, lur_mode=args.lur_mode, attn_mode=args.attn_mode)
