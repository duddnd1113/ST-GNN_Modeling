"""
[Step 6] Ablation 결과 분석 및 Feature Importance

출력:
    checkpoints/feature_importance_E_all.csv
    checkpoints/evaluation_report.txt

실행:
    python3 step6_evaluate.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from config import (
    FEATURES_TRAIN_CSV, FEATURES_TEST_CSV,
    ABLATION_RESULTS, CKPT_DIR, USE_LOG_TARGET,
    FEATURE_GROUPS, ABLATION_CONFIGS,
)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


MODEL_DIR = os.path.join(CKPT_DIR, "models")


def load_results():
    with open(ABLATION_RESULTS, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_importance_lgbm(exp_name="E_all"):
    """LightGBM feature importance 추출."""
    model_path = os.path.join(MODEL_DIR, "{}_lgbm.pkl".format(exp_name))
    if not os.path.exists(model_path):
        print("  {} 모델 없음".format(model_path))
        return None

    model = joblib.load(model_path)
    results = load_results()
    feats = results[exp_name]["features"]

    fi = list(zip(feats, model.feature_importances_))
    fi = sorted(fi, key=lambda x: -x[1])
    df_fi = pd.DataFrame(fi, columns=["feature", "importance"])
    out_path = os.path.join(CKPT_DIR, "feature_importance_{}.csv".format(exp_name))
    df_fi.to_csv(out_path, index=False)
    return df_fi


def detailed_evaluation(exp_name="E_all"):
    """오차 분포, 분위수별 성능 등 상세 분석."""
    model_path = os.path.join(MODEL_DIR, "{}_lgbm.pkl".format(exp_name))
    if not os.path.exists(model_path):
        return

    model = joblib.load(model_path)
    results = load_results()
    feats = results[exp_name]["features"]

    df_te = pd.read_csv(FEATURES_TEST_CSV, parse_dates=["date"])
    feats = [f for f in feats if f in df_te.columns]
    X_te = df_te[feats].values
    y_te = df_te["road_pm"].values

    pred = model.predict(X_te)
    if USE_LOG_TARGET:
        pred = np.expm1(pred)

    df_eval = pd.DataFrame({"y_true": y_te, "y_pred": pred})
    df_eval["error"] = df_eval["y_pred"] - df_eval["y_true"]
    df_eval["abs_error"] = df_eval["error"].abs()

    # 분위수별 분석
    pct_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    pct_bins = [0, 25, 50, 75, 100]
    quantiles = np.percentile(y_te, pct_bins)
    df_eval["pm_bin"] = pd.cut(df_eval["y_true"], bins=quantiles, labels=pct_labels,
                                include_lowest=True, duplicates="drop")

    print("\n=== {} 상세 분석 ===".format(exp_name))
    print("\nPM10 분위수별 MAE:")
    for label, grp in df_eval.groupby("pm_bin"):
        if len(grp) > 0:
            print("  {} (n={}): MAE={:.2f}".format(
                label, len(grp), grp["abs_error"].mean()))

    print("\n오차 분포:")
    print("  bias (mean error): {:.2f}".format(df_eval["error"].mean()))
    print("  std  (error std) : {:.2f}".format(df_eval["error"].std()))
    print("  RMSE             : {:.2f}".format(np.sqrt(mean_squared_error(y_te, pred))))

    # 구별 성능 (지역명 있는 경우)
    if "지역명" in df_te.columns:
        df_te["y_pred"] = pred
        gu_perf = df_te.groupby("지역명").apply(
            lambda g: pd.Series({
                "MAE": mean_absolute_error(g["road_pm"], g["y_pred"]),
                "n": len(g),
            })).reset_index()
        print("\n구별 MAE:")
        print(gu_perf.sort_values("MAE").to_string(index=False))

    out_path = os.path.join(CKPT_DIR, "eval_detail_{}.csv".format(exp_name))
    df_eval.to_csv(out_path, index=False)
    return df_eval


def generate_report():
    """평가 보고서 텍스트 파일 생성."""
    results = load_results()
    report_path = os.path.join(CKPT_DIR, "evaluation_report.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("RoadExtension V2 — Ablation Study 결과 보고서")
    lines.append("=" * 70)
    lines.append("")

    # Phase 1: 피처 Ablation
    lines.append("[ Phase 1: 피처 Ablation (LightGBM) ]")
    lines.append("{:<20} {:>6} {:>10} {:>8} {:>10} {:>8}".format(
        "실험", "피처수", "Train MAE", "Train R²", "Test MAE", "Test R²"))
    lines.append("-" * 70)

    ablation_names = list(ABLATION_CONFIGS.keys())
    for name in ablation_names:
        if name not in results:
            continue
        r = results[name]
        lines.append("{:<20} {:>6} {:>10.2f} {:>8.4f} {:>10.2f} {:>8.4f}".format(
            name, r["n_features"],
            r["train_mae"], r["train_r2"],
            r["test_mae"],  r["test_r2"]))

    lines.append("")
    lines.append("[ Phase 2: 모델 비교 (E_all 피처) ]")
    lines.append("{:<20} {:>10} {:>8} {:>10} {:>8}".format(
        "모델", "Train MAE", "Train R²", "Test MAE", "Test R²"))
    lines.append("-" * 70)

    for name in ["E_all", "E_all_RF", "E_all_MLP"]:
        if name not in results:
            continue
        r = results[name]
        model_label = "{} ({})".format(name, r["model"])
        lines.append("{:<20} {:>10.2f} {:>8.4f} {:>10.2f} {:>8.4f}".format(
            model_label, r["train_mae"], r["train_r2"],
            r["test_mae"], r["test_r2"]))

    lines.append("")
    # 최고 성능
    best = min(results, key=lambda k: results[k]["test_mae"])
    lines.append("최고 성능 실험: {} (Test MAE={:.2f}, R²={:.4f})".format(
        best, results[best]["test_mae"], results[best]["test_r2"]))

    lines.append("")
    lines.append("[ 피처 그룹별 기여 분석 ]")
    # A→B, B→C, B→D 개선량
    pairs = [
        ("A_base", "B_lur", "LUR 추가 효과"),
        ("B_lur", "C_traffic", "격자 교통량 추가 효과"),
        ("B_lur", "D_ambient", "V5 ambient PM 추가 효과"),
        ("D_ambient", "E_all", "교통량+ambient 모두 추가 효과"),
    ]
    for prev, curr, label in pairs:
        if prev in results and curr in results:
            delta = results[prev]["test_mae"] - results[curr]["test_mae"]
            lines.append("  {}: {:.2f} μg/m³ ({})".format(
                label, delta, "개선" if delta > 0 else "악화"))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\n보고서 저장: {}".format(report_path))


def main():
    print("=== [Step 6] 결과 분석 ===\n")

    if not os.path.exists(ABLATION_RESULTS):
        raise FileNotFoundError("step5_train.py를 먼저 실행하세요")

    # Feature Importance
    print("Feature Importance (E_all LightGBM):")
    fi_df = feature_importance_lgbm("E_all")
    if fi_df is not None:
        print(fi_df.head(15).to_string(index=False))

    # 상세 분석
    detailed_evaluation("E_all")

    # 보고서 생성
    print("\n보고서 생성...")
    generate_report()


if __name__ == "__main__":
    main()
