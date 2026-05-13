"""
RoadExtension V2 — 격자 단위 도로 재비산먼지 예측
V1 대비 개선:
  - 격자별 교통량 (서울 평균 X)
  - LUR 변수 전부 포함
  - V5 ambient PM10을 입력 피처로 사용
  - 도로 측정값을 geocoding 기반 격자로 매핑
  - Ablation study 지원
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..")
LUR_BASE  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression"
GRID_DIR  = os.path.join(LUR_BASE, "격자 기본")

# ── 원시 데이터 경로 ──────────────────────────────────────────────────────────
ROAD_PM_FILES = [
    "/home/data/youngwoong/ST-GNN_Dataset/0-2. 도로 미세먼지/서울 비재산먼지_23년도.xlsx",
    "/home/data/youngwoong/ST-GNN_Dataset/0-2. 도로 미세먼지/서울 비재산먼지_24년도.xlsx",
    "/home/data/youngwoong/ST-GNN_Dataset/0-2. 도로 미세먼지/서울 비재산먼지_25년도.xlsx",
]
TRAFFIC_PARQUET = os.path.join(LUR_BASE, "교통량 데이터/grid_traffic_hourly.parquet")

# ── 격자 및 LUR 데이터 경로 ───────────────────────────────────────────────────
GRID_CSV         = os.path.join(GRID_DIR, "격자_250m_4326.csv")
GRID_LUR_CSV     = os.path.join(GRID_DIR, "격자_250m_4326_with_lur.csv")
TIMESTAMPS_PATH  = os.path.join(GRID_DIR, "timestamps_all.npy")
NDVI_PATH        = os.path.join(GRID_DIR, "ndvi_hourly.npy")
IBI_PATH         = os.path.join(GRID_DIR, "ibi_hourly.npy")
LC_PATH          = os.path.join(GRID_DIR, "landcover_static.npy")
BLDG_PATH        = os.path.join(GRID_DIR, "building_stats_static.npy")
WIND_PATH        = os.path.join(GRID_DIR, "wind_all.npy")
TIME_IDX = {
    "train": os.path.join(GRID_DIR, "time_idx_train.npy"),
    "val":   os.path.join(GRID_DIR, "time_idx_val.npy"),
    "test":  os.path.join(GRID_DIR, "time_idx_test.npy"),
}

# ── V5 hidden vector 경로 ─────────────────────────────────────────────────────
HIDDEN_DIR   = os.path.join(ROOT_DIR, "HiddenExtension_V1/data/hidden_vectors")
V5_CKPT_DIR  = os.path.join(ROOT_DIR, "HiddenExtension_V5/checkpoints/V5-base")
V5_GRID_PM = {  # V5 inference 결과 (step3에서 생성)
    "train": os.path.join(V5_CKPT_DIR, "grid_pm10_train.npy"),
    "val":   os.path.join(V5_CKPT_DIR, "grid_pm10_val.npy"),
    "test":  os.path.join(V5_CKPT_DIR, "grid_pm10_test.npy"),
}
STGNN_WINDOW = 12  # ST-GNN 입력 윈도우 크기

# ── 저장 경로 (checkpoints) ──────────────────────────────────────────────────
CKPT_DIR            = os.path.join(BASE_DIR, "checkpoints")
GEOCODE_CACHE       = os.path.join(CKPT_DIR, "road_geocoded.csv")
ROAD_TARGET_CSV     = os.path.join(CKPT_DIR, "road_pm_grid.csv")
FEATURES_TRAIN_CSV  = os.path.join(CKPT_DIR, "features_train.csv")
FEATURES_TEST_CSV   = os.path.join(CKPT_DIR, "features_test.csv")
ABLATION_RESULTS    = os.path.join(CKPT_DIR, "ablation_results.json")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 전처리 파라미터 ──────────────────────────────────────────────────────────
USE_LOG_TARGET   = True   # log1p(y) 변환
GEOCODE_RADIUS_M = 1000   # 도로 측정값 할당 반경 (미터)
MIN_ROAD_STRUC   = 5.0    # 격자 포함 최소 도로 면적 비율 (%)
EXCLUDE_STATUS   = ["매우나쁨"]  # 1차: 극단 상태 제외 (49건)

# 맥락적 이상치 탐지 — 그룹(지역명+계절)별 IQR 기반
# - k=1.5: 표준 IQR fence (Q3 + 1.5*IQR), 이상치를 그룹 중앙값으로 대체
# - 단순 percentile clip 대신 그룹별 분포를 기준으로 엄격하게 탐지
OUTLIER_GROUP    = ["지역명", "season"]  # 그룹 기준 컬럼
OUTLIER_IQR_K    = 1.5                  # IQR 배수 (1.5=표준, 높을수록 관대)

# ── 모델 하이퍼파라미터 ──────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

# Tweedie 목적함수 파라미터 (우편향 양수 분포에 특화)
# tweedie_variance_power: 1.0=Poisson, 2.0=Gamma, 1.5=중간
LGBM_TWEEDIE_PARAMS = dict(
    objective="tweedie",
    tweedie_variance_power=1.5,
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=63,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

# 고농도 샘플 가중치 계수
# 0으로 비활성화: 가중치가 과적합을 유발하는 것으로 확인
SAMPLE_WEIGHT_POWER = 0

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42,
)

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    learning_rate_init=1e-3,
)

# ── Ablation Study 피처 그룹 ──────────────────────────────────────────────────
FEATURE_GROUPS = {
    "temporal": [
        "month_sin", "month_cos",
        "hour_sin",  "hour_cos",
        "weekday", "is_weekend", "season",
    ],
    "weather": ["기온", "습도", "is_dry"],
    "lur": [
        "buildings", "greenspace", "road_struc", "river_zone",
        "ndvi", "ibi",
        "elev_mean", "sum_area", "sum_height",
    ],
    "traffic": ["traffic"],
    "ambient_pm": ["ambient_pm10"],
    # 교호작용 피처
    "interaction": [
        "cold_and_dry",   # 기온<5°C + 습도<50% (r=0.214, 겨울 극단 패턴)
        "traffic_x_road", # 교통량 × 도로 면적 비율 (실질적 교통 노출)
    ],
}

ABLATION_CONFIGS = {
    "A_base":      ["temporal", "weather"],
    "B_lur":       ["temporal", "weather", "lur"],
    "C_traffic":   ["temporal", "weather", "lur", "traffic"],
    "D_ambient":   ["temporal", "weather", "lur", "ambient_pm"],
    "E_all":       ["temporal", "weather", "lur", "traffic", "ambient_pm"],
    "F_interact":  ["temporal", "weather", "lur", "traffic", "ambient_pm", "interaction"],
}
