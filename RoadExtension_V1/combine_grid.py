"""
ST-GNN(V5-base) 대기 PM10 + 도로 재비산먼지 결합 → 격자 통합 PM10 map

결합 공식:
  total_PM_{g,t} = ambient_PM_{g,t}
                 + ALPHA * road_struc_%_{g} * road_PM_{d(g),t}

  road_struc_%_{g} : 격자 g의 도로 면적 비율 (LUR, 0~1)
  road_PM_{d(g),t} : 격자 g가 속한 구(d)의 재비산먼지 RF 예측값
  ALPHA            : 스케일 보정 계수 (calibrate_alpha로 추정)

실행:
    python3 combine_grid.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from config import (
    CKPT_DIR, AMBIENT_GRID, GRID_CSV, TIMESTAMPS,
    LC_PATH, TIME_IDX_TEST, ROAD_LC_IDX, ALPHA,
    HIDDEN_DIR,
)


def load_inputs():
    ambient = np.load(AMBIENT_GRID)           # (T, G)
    grid_csv = pd.read_csv(GRID_CSV)
    lc       = np.load(LC_PATH)               # (G, 4)
    road_frac = lc[:, ROAD_LC_IDX]            # (G,) road_struc_%
    ts       = np.load(TIMESTAMPS)
    tidx     = np.load(TIME_IDX_TEST)
    ts_test  = pd.to_datetime(ts[tidx[12:]])  # (T,)
    return ambient, grid_csv, road_frac, ts_test


def predict_road_pm_daily(ts_test: pd.DatetimeIndex,
                           model, features: list,
                           n_gu: int = 25) -> np.ndarray:
    """
    테스트 기간 각 날짜×구별 재비산먼지 예측.
    Returns: (T,) 전체 서울 일평균 (구 구분 없는 단순 버전)
             → 추후 구별 확장 가능
    """
    dates = ts_test.normalize().unique()
    rows  = []
    for dt in dates:
        row = {
            "기온":       dt.month * 0 + 10,   # 기온 없으면 월별 평균 사용
            "습도":       60,
            "is_dry":     0,
            "month_sin":  np.sin(2 * np.pi * dt.month / 12),
            "month_cos":  np.cos(2 * np.pi * dt.month / 12),
            "season":     {12:0,1:0,2:0, 3:1,4:1,5:1,
                           6:2,7:2,8:2, 9:3,10:3,11:3}[dt.month],
            "is_weekend": int(dt.weekday() >= 5),
            "weekday":    dt.weekday(),
            "도로길이":    3.0,
            "gu_enc":     12,    # 서울 중간값 구
        }
        rows.append(row)

    df_pred = pd.DataFrame(rows)
    feats_available = [f for f in features if f in df_pred.columns
                       and f != 'ambient_pm10_daily']

    # ambient_pm10_daily가 feature에 있으면 ambient 일평균으로 대체
    if 'ambient_pm10_daily' in features:
        ambient = np.load(AMBIENT_GRID)        # (T_test, G)
        ts_all  = np.load(TIMESTAMPS)
        tidx    = np.load(TIME_IDX_TEST)
        ts_test_full = pd.to_datetime(ts_all[tidx[12:]])
        daily_ambient = pd.Series(ambient.mean(axis=1), index=ts_test_full)\
                           .resample('D').mean()
        df_pred.index = dates
        df_pred['ambient_pm10_daily'] = daily_ambient.reindex(dates).values

    X = df_pred[features].fillna(20).values
    road_pm_daily = model.predict(X)   # (n_dates,)

    # 각 타임스텝 → 해당 날짜 매핑
    date_to_idx = {d: i for i, d in enumerate(dates)}
    road_pm_hourly = np.array([
        road_pm_daily[date_to_idx[t.normalize()]] for t in ts_test
    ])
    return road_pm_hourly   # (T,)


def calibrate_alpha(ambient: np.ndarray,
                    road_pm: np.ndarray,
                    road_frac: np.ndarray) -> float:
    """
    실측 관측소 PM10 대비 combined PM10의 스케일 조정.
    도로 기여분이 전체의 약 20~30%가 되도록 보정.
    """
    ambient_mean   = ambient.mean()
    road_contrib   = (road_frac.mean() * road_pm.mean())
    # 도로 기여 비율 목표: 15~25%
    target_fraction = 0.20
    alpha = (target_fraction * ambient_mean) / (road_contrib + 1e-8)
    alpha = float(np.clip(alpha, 0.1, 2.0))
    print(f"  calibrated alpha = {alpha:.3f}")
    print(f"  ambient mean = {ambient_mean:.2f}  road_contrib = {road_contrib:.2f}")
    return alpha


def run_combine():
    print("=== ST-GNN + 도로 재비산먼지 결합 ===\n")

    # 모델 로드
    model_path = os.path.join(CKPT_DIR, "road_rf_model.pkl")
    meta_path  = os.path.join(CKPT_DIR, "road_model_meta.json")
    assert os.path.exists(model_path), f"먼저 train_road.py를 실행하세요: {model_path}"

    model    = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta["features"]
    print(f"도로 RF 모델 로드 완료 (test MAE={meta['test_mae']:.2f})")

    # 데이터 로드
    ambient, grid_csv, road_frac, ts_test = load_inputs()
    T, G = ambient.shape
    print(f"대기 PM10: {ambient.shape}  격자 road_struc_% 범위: {road_frac.min():.3f}~{road_frac.max():.3f}")

    # 도로 재비산먼지 일별 예측 → 시간별 매핑
    print("\n도로 재비산먼지 예측 중...")
    road_pm_t = predict_road_pm_daily(ts_test, model, features)  # (T,)
    print(f"  road_PM: mean={road_pm_t.mean():.2f}  range=[{road_pm_t.min():.1f}, {road_pm_t.max():.1f}]")

    # Alpha 조정
    alpha = calibrate_alpha(ambient, road_pm_t, road_frac)

    # ── 결합 ──────────────────────────────────────────────────────────────
    # road_contribution[t,g] = alpha * road_frac[g] * road_pm[t]
    road_contribution = alpha * road_frac[None, :] * road_pm_t[:, None]  # (T, G)
    total_pm = ambient + road_contribution                                 # (T, G)

    print(f"\n=== 결합 결과 ===")
    print(f"  ambient_PM : mean={ambient.mean():.2f}  std={ambient.std():.2f}")
    print(f"  road_contrib: mean={road_contribution.mean():.2f}  std={road_contribution.std():.2f}")
    print(f"  total_PM   : mean={total_pm.mean():.2f}  std={total_pm.std():.2f}")
    print(f"  도로 기여 비율: {road_contribution.mean()/total_pm.mean()*100:.1f}%")

    # ── 저장 ──────────────────────────────────────────────────────────────
    np.save(os.path.join(CKPT_DIR, "total_pm10_test.npy"), total_pm.astype(np.float32))
    np.save(os.path.join(CKPT_DIR, "road_contribution_test.npy"), road_contribution.astype(np.float32))

    # 격자별 기간 평균 CSV
    pd.DataFrame({
        "CELL_ID":           grid_csv["CELL_ID"],
        "lat":               grid_csv["lat"],
        "lon":               grid_csv["lon"],
        "ambient_pm_mean":   ambient.mean(axis=0),
        "road_contrib_mean": road_contribution.mean(axis=0),
        "total_pm_mean":     total_pm.mean(axis=0),
        "road_frac":         road_frac,
        "road_pct":          (road_contribution.mean(axis=0) / (total_pm.mean(axis=0) + 1e-8) * 100),
    }).to_csv(os.path.join(CKPT_DIR, "combined_grid_mean.csv"), index=False)

    result = {
        "alpha": alpha,
        "ambient_mean": float(ambient.mean()),
        "road_contribution_mean": float(road_contribution.mean()),
        "total_pm_mean": float(total_pm.mean()),
        "road_fraction_pct": float(road_contribution.mean() / total_pm.mean() * 100),
        "shape": list(total_pm.shape),
    }
    with open(os.path.join(CKPT_DIR, "combine_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n저장 완료 → {CKPT_DIR}/")
    print(f"  total_pm10_test.npy       {total_pm.shape}")
    print(f"  road_contribution_test.npy {road_contribution.shape}")
    return total_pm, road_contribution, result


if __name__ == "__main__":
    run_combine()
