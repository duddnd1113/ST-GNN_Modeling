"""
[Step 5] Ablation Study 학습

Ablation 구성 (피처 그룹):
  A_base     : temporal + weather
  B_lur      : + LUR
  C_traffic  : + 격자별 교통량
  D_ambient  : + V5 ambient PM10
  E_all      : 모든 피처
  F_interact : + 교호작용 피처 (겨울건조, hum×pm, traffic×road 등)

추가 개선:
  - 고농도 샘플 가중치 (road_pm 높을수록 더 높은 학습 가중치)
  - Tweedie 목적함수 비교 (우편향 분포 특화)
  - 모델 비교: LightGBM / RF / MLP

출력:
    checkpoints/ablation_results.json
    checkpoints/models/

실행:
    python3 step5_train.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

from config import (
    FEATURES_TRAIN_CSV, FEATURES_TEST_CSV,
    ABLATION_RESULTS, CKPT_DIR,
    LGBM_PARAMS, LGBM_TWEEDIE_PARAMS, RF_PARAMS, MLP_PARAMS,
    FEATURE_GROUPS, ABLATION_CONFIGS,
    USE_LOG_TARGET, SAMPLE_WEIGHT_POWER,
)


MODEL_DIR = os.path.join(CKPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def resolve_features(config_name: str, df: pd.DataFrame) -> list:
    """Ablation 구성에서 실제 컬럼 목록 반환 (NaN이 너무 많은 컬럼 제외)."""
    groups = ABLATION_CONFIGS[config_name]
    cols = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])

    # 실제 df에 존재하는 컬럼만
    existing = [c for c in cols if c in df.columns]

    # NaN이 50% 이상인 컬럼 제외 (해당 피처 미생성 시)
    ok = []
    for c in existing:
        nan_pct = df[c].isna().mean()
        if nan_pct < 0.5:
            ok.append(c)
        else:
            print("  [skip] {} NaN {:.0f}%".format(c, 100 * nan_pct))
    return ok


def evaluate(y_true, y_pred, log_target=USE_LOG_TARGET):
    """log 역변환 후 MAE / R² 계산."""
    if log_target:
        y_pred_inv = np.expm1(y_pred)
        y_true_inv = np.expm1(y_true)
    else:
        y_pred_inv = y_pred
        y_true_inv = y_true
    mae = float(mean_absolute_error(y_true_inv, y_pred_inv))
    r2  = float(r2_score(y_true_inv, y_pred_inv))
    return mae, r2


def make_sample_weights(y_raw: np.ndarray) -> np.ndarray:
    """
    고농도 샘플에 더 높은 학습 가중치 부여.
    weight = 1 + (road_pm / median) ^ SAMPLE_WEIGHT_POWER
    모델이 극단값을 과소예측하는 편향을 완화.
    """
    if SAMPLE_WEIGHT_POWER == 0:
        return None
    median_pm = float(np.median(y_raw[y_raw > 0]))
    weights = 1.0 + (y_raw / median_pm) ** SAMPLE_WEIGHT_POWER
    return weights.astype(np.float32)


def run_lgbm(X_tr, y_tr, X_te, y_te, exp_name, y_tr_raw=None, use_weight=True):
    """
    y_tr_raw: log 변환 전 원래 road_pm 값 (샘플 가중치 계산용)
    use_weight: 샘플 가중치 적용 여부
    """
    weights = None
    if use_weight and y_tr_raw is not None:
        weights = make_sample_weights(y_tr_raw)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        sample_weight=weights,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(80, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    tr_mae, tr_r2 = evaluate(y_tr, model.predict(X_tr))
    te_mae, te_r2 = evaluate(y_te, model.predict(X_te))
    joblib.dump(model, os.path.join(MODEL_DIR, "{}_lgbm.pkl".format(exp_name)))
    return model, tr_mae, tr_r2, te_mae, te_r2


def run_lgbm_tweedie(X_tr, y_tr_raw, X_te, y_te_raw, exp_name):
    """
    Tweedie 목적함수: log1p 변환 없이 원래 값으로 학습.
    우편향(heavy right tail) 분포에 특화된 목적함수.
    """
    weights = make_sample_weights(y_tr_raw)
    model = lgb.LGBMRegressor(**LGBM_TWEEDIE_PARAMS)
    model.fit(
        X_tr, y_tr_raw,
        sample_weight=weights,
        eval_set=[(X_te, y_te_raw)],
        callbacks=[lgb.early_stopping(80, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    tr_mae = float(mean_absolute_error(y_tr_raw, np.clip(pred_tr, 0, None)))
    tr_r2  = float(r2_score(y_tr_raw, np.clip(pred_tr, 0, None)))
    te_mae = float(mean_absolute_error(y_te_raw, np.clip(pred_te, 0, None)))
    te_r2  = float(r2_score(y_te_raw, np.clip(pred_te, 0, None)))
    joblib.dump(model, os.path.join(MODEL_DIR, "{}_tweedie.pkl".format(exp_name)))
    return model, tr_mae, tr_r2, te_mae, te_r2


def impute_nan(X_tr, X_te):
    """RF/MLP용 NaN을 train 중앙값으로 대체."""
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)
    return X_tr_i, X_te_i, imp


def run_rf(X_tr, y_tr, X_te, y_te, exp_name):
    X_tr_i, X_te_i, imp = impute_nan(X_tr, X_te)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr_i, y_tr)
    tr_mae, tr_r2 = evaluate(y_tr, model.predict(X_tr_i))
    te_mae, te_r2 = evaluate(y_te, model.predict(X_te_i))
    joblib.dump((imp, model), os.path.join(MODEL_DIR, "{}_rf.pkl".format(exp_name)))
    return model, tr_mae, tr_r2, te_mae, te_r2


def run_mlp(X_tr, y_tr, X_te, y_te, exp_name):
    X_tr_i, X_te_i, imp = impute_nan(X_tr, X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_te_s = scaler.transform(X_te_i)
    model = MLPRegressor(**MLP_PARAMS)
    model.fit(X_tr_s, y_tr)
    tr_mae, tr_r2 = evaluate(y_tr, model.predict(X_tr_s))
    te_mae, te_r2 = evaluate(y_te, model.predict(X_te_s))
    joblib.dump((imp, scaler, model), os.path.join(MODEL_DIR, "{}_mlp.pkl".format(exp_name)))
    return model, tr_mae, tr_r2, te_mae, te_r2


def main():
    print("=== [Step 5] Ablation Study ===\n")

    if not os.path.exists(FEATURES_TRAIN_CSV):
        raise FileNotFoundError("step4_build_features.py를 먼저 실행하세요")

    df_tr = pd.read_csv(FEATURES_TRAIN_CSV, parse_dates=["date"])
    df_te = pd.read_csv(FEATURES_TEST_CSV,  parse_dates=["date"])
    print("Train: {} 건 / Test: {} 건\n".format(len(df_tr), len(df_te)))

    # log 변환
    y_tr = np.log1p(df_tr["road_pm"].values) if USE_LOG_TARGET else df_tr["road_pm"].values
    y_te = np.log1p(df_te["road_pm"].values) if USE_LOG_TARGET else df_te["road_pm"].values

    results = {}

    # ── Phase 1: 피처 Ablation (LightGBM) ───────────────────────────────────
    print("=" * 60)
    print("Phase 1: 피처 Ablation (LightGBM)")
    print("=" * 60)

    y_tr_raw = df_tr["road_pm"].values  # 샘플 가중치 계산용 원본값
    y_te_raw = df_te["road_pm"].values

    for cfg_name in ABLATION_CONFIGS:
        print("\n[{}]".format(cfg_name))
        feats_tr = resolve_features(cfg_name, df_tr)
        feats_te = resolve_features(cfg_name, df_te)
        feats = [f for f in feats_tr if f in feats_te]
        print("  피처 수: {}  {}".format(len(feats), feats))

        X_tr = df_tr[feats].values
        X_te = df_te[feats].values

        _, tr_mae, tr_r2, te_mae, te_r2 = run_lgbm(
            X_tr, y_tr, X_te, y_te, cfg_name, y_tr_raw=y_tr_raw)

        results[cfg_name] = {
            "model": "LightGBM+weight",
            "features": feats,
            "n_features": len(feats),
            "train_mae": tr_mae, "train_r2": tr_r2,
            "test_mae":  te_mae, "test_r2":  te_r2,
        }
        print("  Train MAE: {:.2f}  R²: {:.4f}".format(tr_mae, tr_r2))
        print("  Test  MAE: {:.2f}  R²: {:.4f}".format(te_mae, te_r2))

    # ── Phase 2: 모델 Ablation (E_all 기준) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: 모델 비교 (E_all 피처)")
    print("=" * 60)

    feats_all_tr = resolve_features("E_all", df_tr)
    feats_all_te = resolve_features("E_all", df_te)
    feats_all = [f for f in feats_all_tr if f in feats_all_te]
    X_tr_all = df_tr[feats_all].values
    X_te_all = df_te[feats_all].values

    # RF
    print("\n[E_all_RF]")
    _, tr_mae, tr_r2, te_mae, te_r2 = run_rf(X_tr_all, y_tr, X_te_all, y_te, "E_all_RF")
    results["E_all_RF"] = {
        "model": "RandomForest",
        "features": feats_all,
        "n_features": len(feats_all),
        "train_mae": tr_mae, "train_r2": tr_r2,
        "test_mae":  te_mae, "test_r2":  te_r2,
    }
    print("  Train MAE: {:.2f}  R²: {:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE: {:.2f}  R²: {:.4f}".format(te_mae, te_r2))

    # MLP
    print("\n[E_all_MLP]")
    _, tr_mae, tr_r2, te_mae, te_r2 = run_mlp(X_tr_all, y_tr, X_te_all, y_te, "E_all_MLP")
    results["E_all_MLP"] = {
        "model": "MLP",
        "features": feats_all,
        "n_features": len(feats_all),
        "train_mae": tr_mae, "train_r2": tr_r2,
        "test_mae":  te_mae, "test_r2":  te_r2,
    }
    print("  Train MAE: {:.2f}  R²: {:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE: {:.2f}  R²: {:.4f}".format(te_mae, te_r2))

    # ── Phase 3: Tweedie (F_interact 피처 기준) ──────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Tweedie 목적함수 비교 (F_interact 피처)")
    print("=" * 60)

    feats_f_tr = resolve_features("F_interact", df_tr)
    feats_f_te = resolve_features("F_interact", df_te)
    feats_f = [ff for ff in feats_f_tr if ff in feats_f_te]
    X_tr_f = df_tr[feats_f].values
    X_te_f = df_te[feats_f].values

    print("\n[F_interact_Tweedie]  (log 변환 없이 원래 값으로 학습)")
    _, tr_mae, tr_r2, te_mae, te_r2 = run_lgbm_tweedie(
        X_tr_f, y_tr_raw, X_te_f, y_te_raw, "F_interact")
    results["F_interact_Tweedie"] = {
        "model": "LightGBM_Tweedie",
        "features": feats_f,
        "n_features": len(feats_f),
        "train_mae": tr_mae, "train_r2": tr_r2,
        "test_mae":  te_mae, "test_r2":  te_r2,
    }
    print("  Train MAE: {:.2f}  R²: {:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE: {:.2f}  R²: {:.4f}".format(te_mae, te_r2))

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    with open(ABLATION_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 결과 요약")
    print("=" * 60)
    header = "{:<20} {:>8} {:>8} {:>8} {:>8}".format(
        "실험", "Train MAE", "Train R²", "Test MAE", "Test R²")
    print(header)
    print("-" * 60)
    for name, r in results.items():
        row = "{:<20} {:>8.2f} {:>8.4f} {:>8.2f} {:>8.4f}".format(
            name, r["train_mae"], r["train_r2"], r["test_mae"], r["test_r2"])
        print(row)

    # 최고 성능 (test MAE 기준)
    best = min(results, key=lambda k: results[k]["test_mae"])
    print("\n최고 성능: {} (Test MAE={:.2f})".format(
        best, results[best]["test_mae"]))

    print("\n결과 저장: {}".format(ABLATION_RESULTS))
    return results


if __name__ == "__main__":
    main()
