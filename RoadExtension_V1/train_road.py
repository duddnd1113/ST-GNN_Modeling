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

from config import CKPT_DIR, RF_PARAMS
from preprocess import build_dataset

FEATURES_BASE = [
    '기온', '습도', 'is_dry',
    'month_sin', 'month_cos',
    'season', 'is_weekend', 'weekday',
    '도로길이', 'gu_enc',
]


def train(use_ambient: bool = True, use_traffic: bool = True):
    print("=== 도로 재비산먼지 RF 모델 학습 ===\n")
    df, le_gu = build_dataset()

    features = list(FEATURES_BASE)
    if use_ambient and df.get('ambient_pm10_daily', pd.Series()).notna().sum() > 100:
        features.append('ambient_pm10_daily')
    if use_traffic and df.get('traffic_mean', pd.Series()).notna().sum() > 100:
        features += ['traffic_mean', 'traffic_log', 'is_rush']

    df_m = df[features + ['재비산먼지', 'year']].dropna()
    mask_tr = df_m['year'] < 2025
    X_tr = df_m.loc[mask_tr,  features].values
    y_tr = df_m.loc[mask_tr,  '재비산먼지'].values
    X_te = df_m.loc[~mask_tr, features].values
    y_te = df_m.loc[~mask_tr, '재비산먼지'].values

    print(f"\n피처: {features}")
    print(f"Train {len(X_tr)}건 / Test {len(X_te)}건")

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    tr_mae  = float(mean_absolute_error(y_tr, pred_tr))
    te_mae  = float(mean_absolute_error(y_te, pred_te))
    te_r2   = float(r2_score(y_te, pred_te))

    print(f"\n결과:")
    print(f"  Train MAE: {tr_mae:.2f} μg/m³")
    print(f"  Test  MAE: {te_mae:.2f} μg/m³")
    print(f"  Test  R² : {te_r2:.4f}")

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
        "features":      features,
        "train_mae":     tr_mae,
        "test_mae":      te_mae,
        "test_r2":       te_r2,
        "train_period":  "2023~2024",
        "test_period":   "2025",
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
