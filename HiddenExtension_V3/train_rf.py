"""
HiddenExtension V3 — RF 학습 및 Ablation 실험

단일 실험:
    python3 train_rf.py --exp RF-HST

전체 ablation:
    python3 train_rf.py --run_all
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from config import (
    HIDDEN_DIR, GRID_DIR, CKPT_DIR, STGNN_WINDOW,
    NDVI_PATH, IBI_PATH, LC_PATH, BLDG_PATH, WIND_PATH, POPULATION_PATH,
    GRID_CSV_PATH, TIMESTAMPS_PATH, TIME_IDX,
    IDW_SIGMA_D, IDW_SIGMA_W, IDW_K,
    RF_PARAMS, EXPERIMENTS, TEMPORAL_COLS,
)
from grid_hidden import GridHiddenGenerator
from feature_builder import (
    FeatureBuilder, build_temporal_features, build_lur_features,
    load_split_data, flatten_station_data,
)


def load_grid_data():
    """Grid-level 공통 데이터 로드."""
    return {
        "ndvi":       np.load(NDVI_PATH),
        "ibi":        np.load(IBI_PATH),
        "lc":         np.load(LC_PATH),
        "bldg":       np.load(BLDG_PATH),
        "wind":       np.load(WIND_PATH),
        "population": np.load(POPULATION_PATH),
        "coords_sta": np.load(os.path.join(HIDDEN_DIR, "coords.npy")),
        "sta2grid":   np.load(os.path.join(HIDDEN_DIR, "station_to_grid_idx.npy")),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, x_dim: int = 0) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    n    = len(y_true)
    adj_r2 = float(1 - (1 - r2) * (n - 1) / max(n - x_dim - 1, 1))
    return dict(mae=mae, rmse=rmse, r2=r2, adj_r2=adj_r2)


def run_experiment(exp_cfg: tuple, grid_data: dict, verbose: bool = True) -> dict:
    """
    단일 실험 실행.

    exp_cfg: (exp_id, use_hidden, idw_method, pca_k, use_lur, use_population, use_temporal)
    """
    exp_id, use_hidden, idw_method, pca_k, use_lur, use_population, use_temporal = exp_cfg
    out_dir = os.path.join(CKPT_DIR, exp_id)

    if os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"  [SKIP] 이미 완료: {exp_id}")
        with open(os.path.join(out_dir, "metrics.json")) as f:
            return json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if verbose:
        print(f"\n{'─'*55}\n  실험: {exp_id}")
        print(f"  use_hidden={use_hidden}  idw={idw_method}  pca_k={pca_k}")
        print(f"  use_lur={use_lur}  use_pop={use_population}  use_temporal={use_temporal}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    train_data = load_split_data("train", STGNN_WINDOW)
    val_data   = load_split_data("val",   STGNN_WINDOW)
    test_data  = load_split_data("test",  STGNN_WINDOW)

    T_tr, N, d = train_data["h"].shape
    sta2grid    = grid_data["sta2grid"]

    # ── IDW 생성기 (hidden 사용 시) ──────────────────────────────────────────
    if use_hidden:
        grid_csv    = pd.read_csv(GRID_CSV_PATH)
        coords_grid = grid_csv[["lat", "lon"]].values.astype(np.float32)
        gen = GridHiddenGenerator(
            grid_data["coords_sta"], coords_grid,
            method=idw_method, sigma_d=IDW_SIGMA_D, sigma_w=IDW_SIGMA_W, k=IDW_K,
        )
        gen.fit()

    # ── LUR 피처 (station 위치 기준) ────────────────────────────────────────
    def get_lur_station(data):
        lur, lur_names = build_lur_features(
            ndvi=grid_data["ndvi"], ibi=grid_data["ibi"],
            lc=grid_data["lc"][:, :],    bldg=grid_data["bldg"],
            population=grid_data["population"] if use_population else None,
            target_global_idx=data["global_idx"],
        )
        # station 위치만 추출 (grid → station 인덱스)
        return lur[:, sta2grid, :], lur_names  # (T, N, n_lur)

    lur_tr, lur_names = get_lur_station(train_data)
    lur_val, _        = get_lur_station(val_data)
    lur_test, _       = get_lur_station(test_data)

    # ── Temporal 피처 ────────────────────────────────────────────────────────
    temp_tr   = build_temporal_features(train_data["timestamps"])  # (T_tr, 9)
    temp_val  = build_temporal_features(val_data["timestamps"])
    temp_test = build_temporal_features(test_data["timestamps"])

    # ── Flatten (T, N, *) → (T*N, *) ────────────────────────────────────────
    h_tr_f, lur_tr_f, temp_tr_f, pm_tr_f     = flatten_station_data(train_data, lur_tr,   temp_tr)
    h_val_f, lur_val_f, temp_val_f, pm_val_f = flatten_station_data(val_data,   lur_val,  temp_val)
    h_te_f, lur_te_f, temp_te_f, pm_te_f     = flatten_station_data(test_data,  lur_test, temp_test)

    # ── FeatureBuilder fit ───────────────────────────────────────────────────
    builder = FeatureBuilder(
        use_hidden=use_hidden, use_lur=use_lur,
        use_population=use_population, use_temporal=use_temporal,
        pca_k=pca_k,
    )
    builder.fit_scalers(
        h_flat   = h_tr_f   if use_hidden else np.zeros((1, d), dtype=np.float32),
        lur_flat = lur_tr_f if use_lur    else np.zeros((1, lur_tr_f.shape[1]), dtype=np.float32),
    )

    feat_names = builder.build_feature_names(
        h_dim=d, lur_names=lur_names if use_lur else []
    )

    # ── 피처 조합 ────────────────────────────────────────────────────────────
    X_tr  = builder.assemble(h_tr_f  if use_hidden else None,
                             lur_tr_f  if use_lur else None, temp_tr_f)
    X_val = builder.assemble(h_val_f if use_hidden else None,
                             lur_val_f if use_lur else None, temp_val_f)
    X_te  = builder.assemble(h_te_f  if use_hidden else None,
                             lur_te_f  if use_lur else None, temp_te_f)

    if verbose:
        print(f"  X_train: {X_tr.shape}  X_val: {X_val.shape}  X_test: {X_te.shape}")
        print(f"  피처 수: {len(feat_names)}")

    # ── RF 학습 ──────────────────────────────────────────────────────────────
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr, pm_tr_f)

    # ── 평가 ─────────────────────────────────────────────────────────────────
    pred_val  = rf.predict(X_val)
    pred_test = rf.predict(X_te)

    val_metrics  = compute_metrics(pm_val_f,  pred_val,  x_dim=X_val.shape[1])
    test_metrics = compute_metrics(pm_te_f,   pred_test, x_dim=X_te.shape[1])

    elapsed = (time.time() - t0) / 60

    # ST-GNN baseline
    stgnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../checkpoints/window_12/S3_transport_pm10_pollutants/static/metrics.json")
    stgnn_mae = None
    if os.path.exists(stgnn_path):
        with open(stgnn_path) as f:
            stgnn_mae = json.load(f)["mae"]

    result = {
        "exp_id":        exp_id,
        "use_hidden":    use_hidden,
        "idw_method":    idw_method,
        "pca_k":         pca_k,
        "use_lur":       use_lur,
        "use_population": use_population,
        "use_temporal":  use_temporal,
        "n_features":    len(feat_names),
        "n_train":       len(pm_tr_f),
        # val
        "val_mae":    val_metrics["mae"],
        "val_rmse":   val_metrics["rmse"],
        "val_r2":     val_metrics["r2"],
        # test
        "test_mae":   test_metrics["mae"],
        "test_rmse":  test_metrics["rmse"],
        "test_r2":    test_metrics["r2"],
        "test_adj_r2": test_metrics["adj_r2"],
        # 비교
        "stgnn_mae":  stgnn_mae,
        "elapsed_min": round(elapsed, 2),
    }

    if verbose:
        print(f"\n  {'모델':<22} {'MAE':>7}  {'RMSE':>7}  {'R²':>7}")
        print(f"  {'─'*48}")
        if stgnn_mae:
            print(f"  {'ST-GNN baseline':<22} {stgnn_mae:>7.4f}")
        print(f"  {'Val ' + exp_id:<22} {val_metrics['mae']:>7.4f}  "
              f"{val_metrics['rmse']:>7.4f}  {val_metrics['r2']:>7.4f}")
        print(f"  {'Test ' + exp_id:<22} {test_metrics['mae']:>7.4f}  "
              f"{test_metrics['rmse']:>7.4f}  {test_metrics['r2']:>7.4f}")
        print(f"  elapsed: {elapsed:.1f}min")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    joblib.dump(rf,      os.path.join(out_dir, "rf_model.pkl"))
    joblib.dump(builder, os.path.join(out_dir, "feature_builder.pkl"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feat_names[:len(rf.feature_importances_)],
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    if verbose:
        print(f"\n  Top-10 feature importance:")
        for _, row in fi.head(10).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")

    return result


def run_all(verbose: bool = True):
    """전체 ablation 실험 실행."""
    grid_data = load_grid_data()
    all_results = []

    for cfg in EXPERIMENTS:
        result = run_experiment(cfg, grid_data, verbose=verbose)
        all_results.append(result)

    # 요약 출력
    df = pd.DataFrame(all_results).sort_values("test_mae")
    print(f"\n{'='*70}")
    print("  전체 실험 결과 요약 (test_mae 순)")
    stgnn = df["stgnn_mae"].iloc[0]
    print(f"  ST-GNN baseline: {stgnn:.4f}")
    print(f"{'='*70}")
    print(f"  {'exp_id':<22} {'test_MAE':>10} {'vs_baseline':>12} {'test_R²':>8}")
    print(f"  {'─'*55}")
    for _, row in df.iterrows():
        diff = row["test_mae"] - stgnn
        sign = "↑" if diff < 0 else "↓"
        print(f"  {row['exp_id']:<22} {row['test_mae']:>10.4f} "
              f"{sign}{abs(diff):>10.4f}  {row['test_r2']:>8.4f}")

    df.to_csv(os.path.join(CKPT_DIR, "results_summary.csv"), index=False)
    print(f"\n  결과 저장 → {CKPT_DIR}/results_summary.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     type=str, default=None, help="단일 실험 ID")
    parser.add_argument("--run_all", action="store_true",    help="전체 ablation 실행")
    args = parser.parse_args()

    if args.run_all:
        run_all()
    elif args.exp:
        cfg_map = {e[0]: e for e in EXPERIMENTS}
        if args.exp not in cfg_map:
            print(f"사용 가능한 실험: {list(cfg_map.keys())}")
        else:
            grid_data = load_grid_data()
            run_experiment(cfg_map[args.exp], grid_data)
    else:
        parser.print_help()
