"""
[Step 5] V3 종합 모델 비교

이상치 처리: 1-99 percentile clip + Box-Cox(lambda=-0.11) 변환

실험 모델:
  1. LightGBM_BC       — Box-Cox 타깃, D_ambient 피처 (baseline)
  2. LightGBM_Huber    — Huber loss (δ=8), Box-Cox 없음 (raw scale)
  3. LightGBM_Weighted — Box-Cox + 고농도 sample weight
  4. XGBoost_BC        — Box-Cox 타깃
  5. LinearSVR         — 전체 데이터, 선형 커널
  6. SVR_RBF           — 15k 서브샘플, RBF 커널
  7. TwoStage          — 시간 LightGBM + 공간 Ridge (2단계)

실행:
    python3 step5_train.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

from config import (
    FEATURES_SRC_TR, FEATURES_SRC_TE,
    CKPT_DIR, ABLATION_RESULTS,
    ALL_FEATS, TEMPORAL_STAGE_FEATS, SPATIAL_STAGE_FEATS,
    LGBM_BASE, LGBM_HUBER, XGBOOST_PARAMS,
    LSVR_PARAMS, SVR_RBF_PARAMS, SVR_SUBSAMPLE,
    RIDGE_SPATIAL, SAMPLE_WEIGHT_POWER,
    USE_BOXCOX,
)
from preprocess import RoadPMTransformer

MODEL_DIR = os.path.join(CKPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 유틸리티 ──────────────────────────────────────────────────────────────
def resolve_feats(names, df):
    return [f for f in names if f in df.columns and df[f].isna().mean() < 0.5]


def impute(X_tr, X_te):
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_tr), imp.transform(X_te), imp


def scale(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), sc


def evaluate(y_true_raw, y_pred_raw):
    y_pred_raw = np.clip(y_pred_raw, 0, None)
    return float(mean_absolute_error(y_true_raw, y_pred_raw)), \
           float(r2_score(y_true_raw, y_pred_raw))


def make_weights(y_raw):
    if SAMPLE_WEIGHT_POWER == 0:
        return None
    med = float(np.median(y_raw[y_raw > 0]))
    return (1.0 + (y_raw / med) ** SAMPLE_WEIGHT_POWER).astype(np.float32)


# ── 모델별 학습 함수 ──────────────────────────────────────────────────────
def run_lgbm_bc(X_tr, y_tr_bc, X_te, y_te_raw, tr, name, weights=None):
    """LightGBM with Box-Cox target."""
    import copy
    params = copy.deepcopy(LGBM_BASE)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr_bc,
              sample_weight=weights,
              eval_set=[(X_te, tr.transform(y_te_raw))],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(period=-1)])
    pred_te = tr.inverse_transform(model.predict(X_te))
    pred_tr = tr.inverse_transform(model.predict(X_tr))
    y_tr_raw = tr.inverse_transform(y_tr_bc)
    joblib.dump((tr, model), os.path.join(MODEL_DIR, "{}.pkl".format(name)))
    return model, evaluate(y_tr_raw, pred_tr), evaluate(y_te_raw, pred_te)


def run_lgbm_huber(X_tr, y_tr_raw, X_te, y_te_raw, name):
    """LightGBM Huber loss — 원래 scale에서 직접 학습 (clip만 적용)."""
    import copy
    params = copy.deepcopy(LGBM_HUBER)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr_raw,
              eval_set=[(X_te, y_te_raw)],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(period=-1)])
    pred_te = np.clip(model.predict(X_te), 0, None)
    pred_tr = np.clip(model.predict(X_tr), 0, None)
    joblib.dump(model, os.path.join(MODEL_DIR, "{}.pkl".format(name)))
    return model, evaluate(y_tr_raw, pred_tr), evaluate(y_te_raw, pred_te)


def run_xgboost_bc(X_tr, y_tr_bc, X_te, y_te_raw, tr, name):
    import copy
    params = copy.deepcopy(XGBOOST_PARAMS)
    # NaN 처리 (XGBoost는 NaN 자동 처리)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr_bc,
              eval_set=[(X_te, tr.transform(y_te_raw))],
              early_stopping_rounds=60,
              verbose=False)
    pred_te = tr.inverse_transform(model.predict(X_te))
    pred_tr = tr.inverse_transform(model.predict(X_tr))
    y_tr_raw = tr.inverse_transform(y_tr_bc)
    joblib.dump((tr, model), os.path.join(MODEL_DIR, "{}.pkl".format(name)))
    return model, evaluate(y_tr_raw, pred_tr), evaluate(y_te_raw, pred_te)


def run_linear_svr(X_tr, y_tr_bc, X_te, y_te_raw, tr, name):
    X_tr_i, X_te_i, imp = impute(X_tr, X_te)
    X_tr_s, X_te_s, sc  = scale(X_tr_i, X_te_i)
    model = LinearSVR(**LSVR_PARAMS)
    model.fit(X_tr_s, y_tr_bc)
    pred_te = tr.inverse_transform(model.predict(X_te_s))
    pred_tr = tr.inverse_transform(model.predict(X_tr_s))
    y_tr_raw = tr.inverse_transform(y_tr_bc)
    joblib.dump((tr, imp, sc, model), os.path.join(MODEL_DIR, "{}.pkl".format(name)))
    return model, evaluate(y_tr_raw, pred_tr), evaluate(y_te_raw, pred_te)


def run_svr_rbf(X_tr, y_tr_bc, X_te, y_te_raw, tr, name):
    """RBF SVR — 서브샘플 학습."""
    np.random.seed(42)
    n_sub = min(SVR_SUBSAMPLE, len(X_tr))
    idx   = np.random.choice(len(X_tr), size=n_sub, replace=False)
    X_sub, y_sub = X_tr[idx], y_tr_bc[idx]

    X_sub_i, X_te_i, imp = impute(X_sub, X_te)
    X_sub_s, X_te_s, sc  = scale(X_sub_i, X_te_i)

    model = SVR(**SVR_RBF_PARAMS)
    model.fit(X_sub_s, y_sub)
    pred_te = tr.inverse_transform(model.predict(X_te_s))
    pred_sub = tr.inverse_transform(model.predict(X_sub_s))
    y_sub_raw = tr.inverse_transform(y_sub)
    print("    (서브샘플 {}건으로 학습)".format(n_sub))
    joblib.dump((tr, imp, sc, model), os.path.join(MODEL_DIR, "{}.pkl".format(name)))
    return model, evaluate(y_sub_raw, pred_sub), evaluate(y_te_raw, pred_te)


def run_two_stage(df_tr, df_te, y_tr_bc, y_te_raw, tr, name):
    """
    2단계 시공간 분리 모델.
    Stage 1: 시간 LightGBM (날씨+시간 → 시간적 기준값 예측)
    Stage 2: 공간 Ridge (LUR+ambient → 공간 잔차 예측)
    """
    feats_t = resolve_feats(TEMPORAL_STAGE_FEATS, df_tr)
    feats_s = resolve_feats(SPATIAL_STAGE_FEATS, df_tr)

    X_tr_t = df_tr[feats_t].values
    X_te_t = df_te[feats_t].values
    X_tr_s = df_tr[feats_s].values
    X_te_s = df_te[feats_s].values

    # Stage 1: 시간 모델
    import copy
    params1 = copy.deepcopy(LGBM_BASE)
    m1 = lgb.LGBMRegressor(**params1)
    m1.fit(X_tr_t, y_tr_bc,
           eval_set=[(X_te_t, tr.transform(y_te_raw))],
           callbacks=[lgb.early_stopping(60, verbose=False),
                      lgb.log_evaluation(period=-1)])

    stage1_tr = m1.predict(X_tr_t)
    stage1_te = m1.predict(X_te_t)
    resid_tr  = y_tr_bc - stage1_tr

    print("    Stage1 피처 {}개: {}".format(len(feats_t), feats_t))
    print("    Stage2 피처 {}개: {}".format(len(feats_s), feats_s))
    print("    Stage1 잔차 std: {:.4f}  (전체 std: {:.4f})".format(
        resid_tr.std(), y_tr_bc.std()))

    # Stage 2: 공간 잔차 모델
    X_tr_si, X_te_si, imp2 = impute(X_tr_s, X_te_s)
    m2 = Ridge(**RIDGE_SPATIAL)
    m2.fit(X_tr_si, resid_tr)
    resid_te = m2.predict(X_te_si)
    resid_tr_pred = m2.predict(X_tr_si)

    # 최종 예측
    pred_te_bc = stage1_te + resid_te
    pred_tr_bc = stage1_tr + resid_tr_pred
    pred_te = tr.inverse_transform(pred_te_bc)
    pred_tr = tr.inverse_transform(pred_tr_bc)
    y_tr_raw = tr.inverse_transform(y_tr_bc)

    joblib.dump((tr, m1, imp2, m2), os.path.join(MODEL_DIR, "{}.pkl".format(name)))
    return (m1, m2), evaluate(y_tr_raw, pred_tr), evaluate(y_te_raw, pred_te)


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    print("=== [Step 5] V3 종합 모델 비교 ===\n")

    # 데이터 로드
    df_tr = pd.read_csv(FEATURES_SRC_TR, parse_dates=["date"])
    df_te = pd.read_csv(FEATURES_SRC_TE, parse_dates=["date"])
    print("Train: {:,}건  Test: {:,}건".format(len(df_tr), len(df_te)))

    # 피처 준비
    feats = resolve_feats(ALL_FEATS, df_tr)
    feats = [f for f in feats if f in df_te.columns]
    X_tr  = df_tr[feats].values
    X_te  = df_te[feats].values

    # 이상치 처리 + Box-Cox
    print("\n[전처리]")
    tr = RoadPMTransformer()
    y_tr_raw = df_tr["road_pm"].values
    y_te_raw = df_te["road_pm"].values
    y_tr_bc  = tr.fit_transform(y_tr_raw)
    y_te_bc  = tr.transform(y_te_raw)
    tr.summary()
    print("  피처 수: {}".format(len(feats)))

    results = {}

    # ── 1. LightGBM_BC (기준선) ──────────────────────────────────────────
    print("\n" + "="*60)
    print("[1] LightGBM_BC  (Box-Cox, D_ambient 피처)")
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_lgbm_bc(
        X_tr, y_tr_bc, X_te, y_te_raw, tr, "lgbm_bc")
    print("  Train MAE={:.4f}  R²={:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}".format(te_mae, te_r2))
    results["LightGBM_BC"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                                   test_mae=te_mae, test_r2=te_r2,
                                   note="Box-Cox 타깃, 기준선")

    # ── 2. LightGBM_Huber (Huber loss) ──────────────────────────────────
    print("\n[2] LightGBM_Huber  (Huber loss δ=8, clip만 적용)")
    y_tr_clip = np.clip(y_tr_raw, tr.p_low, tr.p_high)
    y_te_clip = np.clip(y_te_raw, tr.p_low, tr.p_high)
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_lgbm_huber(
        X_tr, y_tr_clip, X_te, y_te_clip, "lgbm_huber")
    # 최종 평가는 원본 스케일로
    te_mae_raw, te_r2_raw = evaluate(y_te_raw, np.clip(
        joblib.load(os.path.join(MODEL_DIR, "lgbm_huber.pkl")).predict(X_te), 0, None))
    print("  Train MAE={:.4f}  R²={:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}  (원본 스케일)".format(te_mae_raw, te_r2_raw))
    results["LightGBM_Huber"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                                      test_mae=te_mae_raw, test_r2=te_r2_raw,
                                      note="Huber loss δ=8, 극단치 완화")

    # ── 3. LightGBM_Weighted (BC + 고농도 가중치) ────────────────────────
    print("\n[3] LightGBM_Weighted  (Box-Cox + 고농도 sample weight)")
    weights = make_weights(y_tr_raw)
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_lgbm_bc(
        X_tr, y_tr_bc, X_te, y_te_raw, tr, "lgbm_weighted", weights=weights)
    print("  Train MAE={:.4f}  R²={:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}".format(te_mae, te_r2))
    results["LightGBM_Weighted"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                                         test_mae=te_mae, test_r2=te_r2,
                                         note="고농도 샘플 가중치 w=1+(pm/med)^0.5")

    # ── 4. XGBoost_BC ────────────────────────────────────────────────────
    print("\n[4] XGBoost_BC  (Box-Cox 타깃)")
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_xgboost_bc(
        X_tr, y_tr_bc, X_te, y_te_raw, tr, "xgboost_bc")
    print("  Train MAE={:.4f}  R²={:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}".format(te_mae, te_r2))
    results["XGBoost_BC"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                                  test_mae=te_mae, test_r2=te_r2,
                                  note="XGBoost, Box-Cox 타깃")

    # ── 5. LinearSVR ─────────────────────────────────────────────────────
    print("\n[5] LinearSVR  (전체 데이터, 선형 커널, Box-Cox)")
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_linear_svr(
        X_tr, y_tr_bc, X_te, y_te_raw, tr, "linear_svr")
    print("  Train MAE={:.4f}  R²={:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}".format(te_mae, te_r2))
    results["LinearSVR"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                                 test_mae=te_mae, test_r2=te_r2,
                                 note="선형 커널 SVR, 전체 데이터")

    # ── 6. SVR_RBF (서브샘플) ────────────────────────────────────────────
    print("\n[6] SVR_RBF  ({}건 서브샘플, RBF 커널, Box-Cox)".format(SVR_SUBSAMPLE))
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_svr_rbf(
        X_tr, y_tr_bc, X_te, y_te_raw, tr, "svr_rbf")
    print("  Train MAE={:.4f}  R²={:.4f}  (서브샘플 기준)".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}".format(te_mae, te_r2))
    results["SVR_RBF"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                               test_mae=te_mae, test_r2=te_r2,
                               note="RBF 커널, {}건 서브샘플".format(SVR_SUBSAMPLE))

    # ── 7. TwoStage (시간 LightGBM + 공간 Ridge) ─────────────────────────
    print("\n[7] TwoStage  (시간 LightGBM + 공간 Ridge, Box-Cox)")
    _, (tr_mae, tr_r2), (te_mae, te_r2) = run_two_stage(
        df_tr, df_te, y_tr_bc, y_te_raw, tr, "two_stage")
    print("  Train MAE={:.4f}  R²={:.4f}".format(tr_mae, tr_r2))
    print("  Test  MAE={:.4f}  R²={:.4f}".format(te_mae, te_r2))
    results["TwoStage"] = dict(train_mae=tr_mae, train_r2=tr_r2,
                                test_mae=te_mae, test_r2=te_r2,
                                note="시간 LGBM + 공간 Ridge 2단계")

    # ── 결과 저장 + 요약 ─────────────────────────────────────────────────
    meta = {
        "boxcox_lambda": tr.lam,
        "clip_low":      tr.p_low,
        "clip_high":     tr.p_high,
        "shift":         tr.shift,
        "n_train":       int(len(df_tr)),
        "n_test":        int(len(df_te)),
        "features":      feats,
        "results":       results,
    }
    with open(ABLATION_RESULTS, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("최종 결과 요약 (V2 D_ambient 기준: MAE=7.35  R²=0.314)")
    print("="*60)
    print("{:<22} {:>9} {:>8} {:>9} {:>8}".format(
        "모델", "Train MAE", "Train R²", "Test MAE", "Test R²"))
    print("-" * 60)
    for name, r in results.items():
        marker = " ★" if r["test_mae"] == min(v["test_mae"] for v in results.values()) else ""
        print("{:<22} {:>9.4f} {:>8.4f} {:>9.4f} {:>8.4f}{}".format(
            name, r["train_mae"], r["train_r2"], r["test_mae"], r["test_r2"], marker))

    best = min(results, key=lambda k: results[k]["test_mae"])
    print("\n최고 성능: {} (Test MAE={:.4f})".format(best, results[best]["test_mae"]))
    print("저장 완료: {}".format(ABLATION_RESULTS))
    return results


if __name__ == "__main__":
    main()
