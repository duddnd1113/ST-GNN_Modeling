"""
RoadExtension V3 — 설정
V2 D_ambient 기반, 새 이상치 처리 + 다양한 모델 비교
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

# ── V2 중간 산출물 재사용 (geocoding, V5 inference, features) ──────────────
V2_DIR          = os.path.join(ROOT_DIR, "RoadExtension_V2")
V2_CKPT         = os.path.join(V2_DIR, "checkpoints")
FEATURES_SRC_TR = os.path.join(V2_CKPT, "features_train.csv")   # V2 피처 (원본 road_pm 포함)
FEATURES_SRC_TE = os.path.join(V2_CKPT, "features_test.csv")

# ── V3 저장 경로 ───────────────────────────────────────────────────────────
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
ABLATION_RESULTS = os.path.join(CKPT_DIR, "ablation_results.json")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 이상치 처리 ─────────────────────────────────────────────────────────────
CLIP_LOW_PCT  = 1    # 하위 1 percentile clip
CLIP_HIGH_PCT = 99   # 상위 99 percentile clip

# Box-Cox 설정 (lambda는 학습 데이터에서 자동 탐색)
USE_BOXCOX    = True
BOXCOX_SHIFT  = 0.01   # 0값 방지용 shift

# ── D_ambient 피처 구성 (V2 최고 성능) ────────────────────────────────────
TEMPORAL_FEATS = [
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    "weekday", "is_weekend", "season",
]
WEATHER_FEATS = ["기온", "습도", "is_dry"]
LUR_FEATS = [
    "buildings", "greenspace", "road_struc", "river_zone",
    "ndvi", "ibi", "elev_mean", "sum_area", "sum_height",
]
AMBIENT_FEATS = ["ambient_pm10"]

# 시공간 분리 모델용 피처 분리
# Stage 1: 시간 모델 (공간 정보 없음 — 날씨+시간만)
TEMPORAL_STAGE_FEATS = TEMPORAL_FEATS + WEATHER_FEATS
# Stage 2: 공간 잔차 모델 (격자 특성 + ambient)
SPATIAL_STAGE_FEATS  = LUR_FEATS + AMBIENT_FEATS + ["traffic"]

ALL_FEATS = TEMPORAL_FEATS + WEATHER_FEATS + LUR_FEATS + AMBIENT_FEATS

# ── 모델 하이퍼파라미터 ────────────────────────────────────────────────────
LGBM_BASE = dict(
    n_estimators=500, learning_rate=0.05, max_depth=8,
    num_leaves=63, min_child_samples=10, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)

LGBM_HUBER = dict(
    **{k: v for k, v in LGBM_BASE.items()},
    objective="huber",
    alpha=8.0,   # δ=8: 오차 <8 μg/m³ → MSE, 그 이상 → MAE
)

XGBOOST_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=8,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbosity=0,
)

LSVR_PARAMS = dict(
    C=1.0, max_iter=5000, random_state=42,
)

SVR_RBF_PARAMS = dict(
    C=10.0, gamma="scale", epsilon=0.1,
)
SVR_SUBSAMPLE = 15000   # RBF SVR 서브샘플 크기

RIDGE_SPATIAL = dict(alpha=1.0)

# 고농도 샘플 가중치 (Weighted 실험용)
SAMPLE_WEIGHT_POWER = 0.5   # weight = 1 + (pm / median)^0.5
