"""
도로 재비산먼지 예측 RF 모델 학습

실행:
    python3 train_road.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from config import CKPT_DIR, RF_PARAMS, USE_LOG_TARGET, WINSORIZE_PCT
from preprocess import build_dataset

FEATURES_BASE = [
    '기온', '습도', 'is_dry', 'hum_sq', 'temp_x_hum',
    'month_sin', 'month_cos',
    'season', 'is_weekend', 'weekday',
    '도로길이', 'gu_enc', 'is_daero',
]


def train(use_ambient: bool = True, use_traffic: bool = True):
    print("=== 도로 재비산먼지 RF 모델 학습 ===\n")
    df, le_gu, winsorize_cap = build_dataset()
    winsorize_cap = winsorize_cap  # 역변환 시 필요

    features = list(FEATURES_BASE)
    if use_ambient and 'ambient_pm10_daily' in df and df['ambient_pm10_daily'].notna().sum() > 100:
        features.append('ambient_pm10_daily')
    if use_traffic and 'traffic_mean' in df and df['traffic_mean'].notna().sum() > 100:
        features += ['traffic_mean', 'traffic_log', 'is_rush']

    df_m = df[features + ['재비산먼지', 'year']].dropna()
    mask_tr = df_m['year'] < 2025
    X_tr = df_m.loc[mask_tr,  features].values
    y_tr = df_m.loc[mask_tr,  '재비산먼지'].values
    X_te = df_m.loc[~mask_tr, features].values
    y_te = df_m.loc[~mask_tr, '재비산먼지'].values

    print(f"\n피처 ({len(features)}개): {features}")
    print(f"Train {len(X_tr):,}건 / Test {len(X_te):,}건")
    print(f"이상치 처리: Winsorize {WINSORIZE_PCT}th pct  |  Log 변환: {USE_LOG_TARGET}")

    # log 변환
    y_tr_fit = np.log1p(y_tr) if USE_LOG_TARGET else y_tr

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr, y_tr_fit)

    # 역변환
    pred_tr = np.expm1(model.predict(X_tr)) if USE_LOG_TARGET else model.predict(X_tr)
    pred_te = np.expm1(model.predict(X_te)) if USE_LOG_TARGET else model.predict(X_te)
    tr_mae  = float(mean_absolute_error(y_tr, pred_tr))
    te_mae  = float(mean_absolute_error(y_te, pred_te))
    te_r2   = float(r2_score(y_te, pred_te))

    print(f"\n결과:")
    print(f"  Train MAE: {tr_mae:.2f} μg/m³")
    print(f"  Test  MAE: {te_mae:.2f} μg/m³  (기존 17.66 → 개선)")
    print(f"  Test  R² : {te_r2:.4f}  (기존 0.1348 → 개선)")

    # Feature importance
    fi = sorted(zip(features, model.feature_importances_),
                key=lambda x: -x[1])
    print("\nFeature Importance:")
    for name, imp in fi:
        print(f"  {name:<25} {imp:.4f}")

    # ── 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(CKPT_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(CKPT_DIR, "road_rf_model.pkl"))
    joblib.dump(le_gu,  os.path.join(CKPT_DIR, "label_encoder_gu.pkl"))

    meta = {
        "features":       features,
        "train_mae":      tr_mae,
        "test_mae":       te_mae,
        "test_r2":        te_r2,
        "winsorize_pct":  WINSORIZE_PCT,
        "winsorize_cap":  float(winsorize_cap),
        "use_log_target": USE_LOG_TARGET,
        "train_period":   "2023~2024",
        "test_period":    "2025",
        "n_train":       int(len(X_tr)),
        "n_test":        int(len(X_te)),
        "feature_importance": {n: float(v) for n, v in fi},
    }
    with open(os.path.join(CKPT_DIR, "road_model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 예측값도 저장 (calibration용)
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": pred_te,
        "year":   df_m.loc[~mask_tr, "year"].values,
    }).to_csv(os.path.join(CKPT_DIR, "test_predictions.csv"), index=False)

    print(f"\n저장 완료 → {CKPT_DIR}/")
    return model, meta


if __name__ == "__main__":
    train()
