"""
Station Hold-out 검증 — "그리드 성능" 추정

목적:
  학습한 모델로 한 번도 본 적 없는 station을 예측.
  이 MAE가 실제 grid 추론에서 기대할 수 있는 성능의 추정치.

방법:
  1. HOLDOUT_CLUSTERS의 각 cluster를 순서대로 hold-out
  2. 해당 station의 h를 나머지 station IDW로 근사
  3. dynamic(h_idw) + γ(LUR) 적용 (u_i=0, station bias 없음)
  4. 실측 PM과 비교

실행:
  python3 holdout_eval.py --exp V5-monthly
  python3 holdout_eval.py --run_all
"""
import os, sys, json, argparse

# HiddenExtension_V5가 최우선 — root model.py가 아닌 V5 model.py 로드
_V5_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_V5_DIR, ".."))
_V3_DIR   = os.path.join(_ROOT_DIR, "HiddenExtension_V3")

for _mod in ["model", "config", "dataset"]:   # 캐시 제거
    sys.modules.pop(_mod, None)
for _p in [_ROOT_DIR, _V3_DIR, _V5_DIR]:
    if _p in sys.path: sys.path.remove(_p)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _V3_DIR)
sys.path.insert(0, _V5_DIR)   # 최우선

import numpy as np
import pandas as pd
import torch

from config import (
    CKPT_DIR, H_DIM, N_STATION, LUR_DIM, TEMPORAL_NAMES,
    HIDDEN_DIR, HOLDOUT_CLUSTERS, EXPERIMENTS,
)
from model import FixedEffectPMModel
from dataset import V5Dataset, get_season, get_hour_bin
from grid_hidden import GridHiddenGenerator


def compute_metrics(pred, true):
    mae  = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    r2   = float(1 - np.sum((pred - true)**2) / (np.sum((true - true.mean())**2) + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2)


def run_holdout(exp_id: str, verbose: bool = True) -> dict:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(CKPT_DIR, exp_id)

    # ── 모델 & 설정 로드 ──────────────────────────────────────────────────
    with open(os.path.join(out_dir, "metrics.json")) as f:
        cfg = json.load(f)

    fe_mode = cfg.get("fe_mode", cfg.get("use_seasonal", None))
    model = FixedEffectPMModel(
        n_stations=N_STATION, h_dim=H_DIM, lur_dim=LUR_DIM,
        temporal_dim=len(TEMPORAL_NAMES),
        mlp_hidden=cfg["mlp_hidden"],
        use_bias=cfg["use_bias"],
        use_seasonal_bias=fe_mode,
        use_hier_lur=cfg["use_hier"],
    )
    model.load_state_dict(torch.load(
        os.path.join(out_dir, "best_model.pt"), map_location=device))
    model.to(device).eval()

    # ── 데이터 ────────────────────────────────────────────────────────────
    train_ds = V5Dataset("train")
    test_ds  = V5Dataset("test",
                          h_scaler=train_ds.h_scaler,
                          lur_scaler=train_ds.lur_scaler)

    h_test   = np.load(os.path.join(HIDDEN_DIR, "h_test.npy"))   # (T, N, 64)
    pm_test  = np.load(os.path.join(HIDDEN_DIR, "pm_test.npy"))  # (T, N)
    coords   = np.load(os.path.join(HIDDEN_DIR, "coords.npy"))   # (N, 2)
    T, N, d  = h_test.shape

    # temporal 정보 (test split 기준)
    from config import TIME_IDX, STGNN_WINDOW
    import importlib; ds_mod = importlib.import_module("dataset")
    ts_all   = np.load(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../HiddenExtension_V1/../HiddenExtension_V1/data/hidden_vectors/../../../"
        "ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/격자 기본/timestamps_all.npy"
    ), allow_pickle=True)

    gdir  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/격자 기본"
    ts    = np.load(os.path.join(gdir, "timestamps_all.npy"))
    tidx  = np.load(TIME_IDX["test"])
    ts_test = ts[tidx[STGNN_WINDOW:]]

    # LUR (test split, station 위치)
    lur_norm = test_ds.lur[:N].numpy()   # (N, 9) - 첫 N개 = 한 타임스텝의 station LUR
    lur_t    = torch.from_numpy(lur_norm).to(device)

    # ── 클러스터별 hold-out 평가 ──────────────────────────────────────────
    cluster_results = {}
    all_preds, all_trues = [], []

    for cluster_name, holdout_idx in HOLDOUT_CLUSTERS.items():
        ctx_mask = np.ones(N, dtype=bool)
        ctx_mask[holdout_idx] = False
        ctx_stations = np.where(ctx_mask)[0]

        # IDW 생성기: context station → holdout station 위치로 보간
        coords_ctx  = coords[ctx_mask]
        coords_hold = coords[~ctx_mask]
        gen = GridHiddenGenerator(coords_ctx, coords_hold,
                                   method="wind", k=min(10, ctx_mask.sum()))
        gen.fit()

        ho_preds, ho_trues = [], []

        with torch.no_grad():
            for t in range(T):
                dt_t   = pd.to_datetime(ts_test[t])
                season = get_season(dt_t.month)
                month  = dt_t.month - 1
                hb     = get_hour_bin(dt_t.hour)

                # IDW hidden (wind 없이 거리 기반)
                h_ctx_t  = h_test[t][ctx_mask]      # (N_ctx, 64)
                h_hold_np = gen.transform_timestep(h_ctx_t,
                                                    np.zeros((len(ctx_stations), 2)))
                h_hold_t  = torch.from_numpy(
                    train_ds.h_scaler.transform(h_hold_np).astype(np.float32)
                ).to(device)

                # temporal
                from dataset import build_temporal_features
                temp_t = torch.from_numpy(
                    build_temporal_features(np.array([ts_test[t]]))[0]
                ).to(device)

                # LUR for holdout stations
                lur_hold = lur_t[holdout_idx]

                # 예측: dynamic + LUR bias (station-specific bias 없음, u=0)
                B = len(holdout_idx)
                temp_b = temp_t.unsqueeze(0).expand(B, -1)
                s_t = torch.full((B,), season, dtype=torch.long, device=device)
                m_t = torch.full((B,), month,  dtype=torch.long, device=device)
                h_t = torch.full((B,), hb,     dtype=torch.long, device=device)

                # sta_idx dummy (bias는 어차피 LUR 경로만)
                sta_dummy = torch.zeros(B, dtype=torch.long, device=device)

                # bias 없이 dynamic + LUR만
                dynamic = model.dynamic(
                    torch.cat([h_hold_t, temp_b], dim=-1))   # (B, 1)

                if model.use_hier_lur and hasattr(model, 'gamma'):
                    lur_bias = model.gamma(lur_hold)          # (B, 1)
                else:
                    lur_bias = torch.zeros(B, 1, device=device)

                temporal_bias = model.get_temporal_bias(
                    sta_dummy, s_t, m_t, h_t)                # (B, 1)

                pred_t = (dynamic + lur_bias + temporal_bias).squeeze(-1)
                ho_preds.append(pred_t.cpu().numpy())
                ho_trues.append(pm_test[t][holdout_idx])

        ho_pred = np.concatenate(ho_preds)
        ho_true = np.concatenate(ho_trues)
        m = compute_metrics(ho_pred, ho_true)
        cluster_results[cluster_name] = m
        all_preds.append(ho_pred)
        all_trues.append(ho_true)

        if verbose:
            print(f"  {cluster_name:<10} stations={holdout_idx}  "
                  f"MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

    # 전체 hold-out MAE
    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues))

    # station MAE (비교용)
    station_mae = json.load(open(
        os.path.join(out_dir, "metrics.json")))["test_mae"]

    stgnn_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json")
    stgnn_mae = json.load(open(stgnn_path))["mae"] if os.path.exists(stgnn_path) else None

    result = {
        "exp_id":            exp_id,
        "station_mae":       station_mae,
        "holdout_mae":       overall["mae"],
        "holdout_rmse":      overall["rmse"],
        "holdout_r2":        overall["r2"],
        "holdout_gap":       overall["mae"] - station_mae,
        "stgnn_baseline":    stgnn_mae,
        "cluster_results":   cluster_results,
    }

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  {exp_id}")
        print(f"  Station MAE (학습 위치): {station_mae:.4f}")
        print(f"  Hold-out MAE (미관측)  : {overall['mae']:.4f}  "
              f"(+{overall['mae']-station_mae:.4f})")
        if stgnn_mae:
            print(f"  ST-GNN baseline        : {stgnn_mae:.4f}")
        print(f"  → grid 추론 기대 MAE   : ~{overall['mae']:.4f}")

    json.dump(result, open(
        os.path.join(out_dir, "holdout_metrics.json"), "w"), indent=2)
    return result


def run_all(verbose: bool = True):
    results = []
    for cfg in EXPERIMENTS:
        exp_id = cfg[0]
        out_dir = os.path.join(CKPT_DIR, exp_id)
        if not os.path.exists(os.path.join(out_dir, "best_model.pt")):
            print(f"  [SKIP] {exp_id} — 학습 안 됨")
            continue
        print(f"\n[{exp_id}]")
        r = run_holdout(exp_id, verbose)
        results.append(r)

    if results:
        df = pd.DataFrame([{
            "exp_id": r["exp_id"],
            "station_MAE": r["station_mae"],
            "holdout_MAE": r["holdout_mae"],
            "gap": r["holdout_gap"],
            "holdout_R²": r["holdout_r2"],
        } for r in results]).sort_values("holdout_MAE")

        print(f"\n{'='*65}")
        print(f"  Hold-out 검증 결과 요약 (미관측 위치 기대 성능)")
        print(f"{'='*65}")
        print(df.to_string(index=False))

        stgnn_mae = results[0]["stgnn_baseline"]
        if stgnn_mae:
            print(f"\n  ST-GNN baseline: {stgnn_mae:.4f}")
        print(f"  해석: holdout_MAE ≈ 실제 grid 추론 MAE 기대치")

        df.to_csv(os.path.join(CKPT_DIR, "holdout_summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     type=str, default=None)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all()
    elif args.exp:
        print(f"\n[Hold-out 검증: {args.exp}]")
        run_holdout(args.exp)
    else:
        parser.print_help()
