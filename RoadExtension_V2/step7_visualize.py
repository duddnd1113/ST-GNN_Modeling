"""
[Step 7] 시각화

1. Ablation 비교 막대 그래프 (MAE / R²)
2. 피처 중요도 (E_all LightGBM)
3. 격자 단위 road PM10 공간 분포 (예측 vs 실측)
4. 오차 분포

실행:
    python3 step7_visualize.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from config import (
    ABLATION_RESULTS, FEATURES_TRAIN_CSV, FEATURES_TEST_CSV,
    CKPT_DIR, GRID_CSV, ABLATION_CONFIGS, USE_LOG_TARGET,
)

# 한글 폰트 설정
import matplotlib.font_manager as fm
_KR_FONTS = ["NanumGothic", "NanumBarunGothic", "Malgun Gothic",
             "AppleGothic", "DejaVu Sans"]
_all_fonts = [f.name for f in fm.fontManager.ttflist]
for _f in _KR_FONTS:
    if any(_f.lower() in x.lower() for x in _all_fonts):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

MODEL_DIR = os.path.join(CKPT_DIR, "models")


# ── Plot 1: Ablation 비교 ─────────────────────────────────────────────────────
def plot_ablation_comparison(results):
    ablation_keys = list(ABLATION_CONFIGS.keys())
    model_keys    = ["E_all_RF", "E_all_MLP"]
    all_keys      = ablation_keys + model_keys
    present       = [k for k in all_keys if k in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
              "#795548", "#607D8B"]

    # MAE
    ax = axes[0]
    te_maes = [results[k]["test_mae"] for k in present]
    tr_maes = [results[k]["train_mae"] for k in present]
    x = np.arange(len(present))
    w = 0.35
    bars1 = ax.bar(x - w/2, tr_maes, w, label="Train MAE",
                   color=[c + "88" for c in colors[:len(present)]])
    bars2 = ax.bar(x + w/2, te_maes, w, label="Test MAE",
                   color=colors[:len(present)])
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAE (μg/m³)")
    ax.set_title("Ablation Study — MAE Comparison")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars2, te_maes):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, "{:.1f}".format(v),
                ha="center", va="bottom", fontsize=7)

    # R²
    ax = axes[1]
    te_r2s = [results[k]["test_r2"] for k in present]
    bars = ax.bar(present, te_r2s, color=colors[:len(present)], alpha=0.85)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Ablation Study — R² Comparison")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, te_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, max(v + 0.01, 0.01),
                "{:.3f}".format(v), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = os.path.join(CKPT_DIR, "ablation_comparison.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print("저장: {}".format(out))


# ── Plot 2: Feature Importance ────────────────────────────────────────────────
def plot_feature_importance():
    fi_path = os.path.join(CKPT_DIR, "feature_importance_E_all.csv")
    if not os.path.exists(fi_path):
        return

    df = pd.read_csv(fi_path)
    top = df.head(20)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#2196F3", alpha=0.85)
    ax.set_xlabel("Feature Importance (LightGBM gain)")
    ax.set_title("Top 20 Feature Importance (E_all LightGBM)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(CKPT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print("저장: {}".format(out))


# ── Plot 3: 공간 분포 (격자별 예측 평균) ─────────────────────────────────────
def plot_spatial_distribution():
    model_path = os.path.join(MODEL_DIR, "E_all_lgbm.pkl")
    if not os.path.exists(model_path):
        return
    if not os.path.exists(FEATURES_TEST_CSV):
        return

    model = joblib.load(model_path)
    df_te = pd.read_csv(FEATURES_TEST_CSV, parse_dates=["date"])

    with open(ABLATION_RESULTS, "r") as f:
        results = json.load(f)
    feats = results["E_all"]["features"]

    # 유효 피처만
    feats = [f for f in feats if f in df_te.columns]
    X_te = df_te[feats].values
    pred = model.predict(X_te)
    if USE_LOG_TARGET:
        pred = np.expm1(pred)
    df_te["road_pm_pred"] = pred

    # 격자별 평균 (예측 / 실측)
    cell_agg = df_te.groupby("CELL_ID").agg(
        pm_pred=("road_pm_pred", "mean"),
        pm_true=("road_pm",      "mean"),
    ).reset_index()

    # Grid 좌표
    grid_df = pd.read_csv(GRID_CSV)
    cell_agg = cell_agg.merge(grid_df[["CELL_ID", "lat", "lon"]], on="CELL_ID", how="left")
    cell_agg = cell_agg.dropna(subset=["lat", "lon"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmin = 0
    vmax = max(cell_agg["pm_pred"].quantile(0.99),
               cell_agg["pm_true"].quantile(0.99))

    for ax, col, title in [
        (axes[0], "pm_true", "실측 Road PM10 (μg/m³)"),
        (axes[1], "pm_pred", "예측 Road PM10 (μg/m³)"),
    ]:
        sc = ax.scatter(cell_agg["lon"], cell_agg["lat"],
                        c=cell_agg[col], cmap="YlOrRd",
                        s=8, vmin=vmin, vmax=vmax, alpha=0.9)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(CKPT_DIR, "spatial_road_pm.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print("저장: {}".format(out))


# ── Plot 4: 예측 vs 실측 산점도 ──────────────────────────────────────────────
def plot_scatter():
    model_path = os.path.join(MODEL_DIR, "E_all_lgbm.pkl")
    if not os.path.exists(model_path) or not os.path.exists(FEATURES_TEST_CSV):
        return

    model = joblib.load(model_path)
    df_te = pd.read_csv(FEATURES_TEST_CSV, parse_dates=["date"])

    with open(ABLATION_RESULTS, "r") as f:
        results = json.load(f)
    feats = [f for f in results["E_all"]["features"] if f in df_te.columns]
    X_te = df_te[feats].values
    pred = model.predict(X_te)
    y_te = df_te["road_pm"].values
    if USE_LOG_TARGET:
        pred = np.expm1(pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 산점도
    ax = axes[0]
    ax.scatter(y_te, pred, s=5, alpha=0.4, color="#2196F3")
    lim = max(y_te.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Road PM10 (μg/m³)")
    ax.set_ylabel("Predicted Road PM10 (μg/m³)")
    ax.set_title("Prediction vs Actual (Test 2025)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 오차 히스토그램
    ax = axes[1]
    errors = pred - y_te
    ax.hist(errors, bins=50, color="#FF9800", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Prediction Error (pred - actual)")
    ax.set_ylabel("count")
    ax.set_title("Error Distribution (bias={:.2f})".format(errors.mean()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(CKPT_DIR, "prediction_scatter.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print("저장: {}".format(out))


def main():
    print("=== [Step 7] 시각화 ===\n")

    if not os.path.exists(ABLATION_RESULTS):
        raise FileNotFoundError("step5_train.py를 먼저 실행하세요")

    with open(ABLATION_RESULTS, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("1. Ablation 비교 그래프...")
    plot_ablation_comparison(results)

    print("2. Feature Importance...")
    plot_feature_importance()

    print("3. 공간 분포 지도...")
    plot_spatial_distribution()

    print("4. 예측 vs 실측 산점도...")
    plot_scatter()

    print("\n모든 그래프: {}".format(CKPT_DIR))


if __name__ == "__main__":
    main()
