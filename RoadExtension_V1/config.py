"""
RoadExtension V1 — 도로 재비산먼지 예측 + ST-GNN 결합
"""
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..")
LUR_BASE  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression"
GRID_DIR  = os.path.join(LUR_BASE, "격자 기본")

# ── 원시 데이터 경로 ──────────────────────────────────────────────────────
TRAFFIC_PARQUET = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression/교통량 데이터/grid_traffic_hourly.parquet"
TRAFFIC_CACHE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "traffic_daily_hourly.csv")  # 사전 집계 캐시
ROAD_PM_FILES = [
    "/home/data/youngwoong/ST-GNN_Dataset/0-2. 도로 미세먼지/서울 비재산먼지_23년도.xlsx",
    "/home/data/youngwoong/ST-GNN_Dataset/0-2. 도로 미세먼지/서울 비재산먼지_24년도.xlsx",
    "/home/data/youngwoong/ST-GNN_Dataset/0-2. 도로 미세먼지/서울 비재산먼지_25년도.xlsx",
]
SCENARIO_DIR  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios"
SCENARIO_NAME = "S3_transport_pm10_pollutants"

# ── ST-GNN + V5 경로 ──────────────────────────────────────────────────────
HIDDEN_DIR    = os.path.join(ROOT_DIR, "HiddenExtension_V1/data/hidden_vectors")
AMBIENT_GRID  = os.path.join(ROOT_DIR, "HiddenExtension_V5/checkpoints/V5-base/grid_pm10_test.npy")
V5_MODEL_DIR  = os.path.join(ROOT_DIR, "HiddenExtension_V5/checkpoints/V5-base")

# ── LUR 격자 데이터 ───────────────────────────────────────────────────────
GRID_CSV      = os.path.join(GRID_DIR, "격자_250m_4326.csv")
TIMESTAMPS    = os.path.join(GRID_DIR, "timestamps_all.npy")
LC_PATH       = os.path.join(GRID_DIR, "landcover_static.npy")   # [G, 4]: buildings%, green%, road%, river%
BLDG_PATH     = os.path.join(GRID_DIR, "building_stats_static.npy")
TIME_IDX_TEST = os.path.join(GRID_DIR, "time_idx_test.npy")

# ── 저장 경로 ─────────────────────────────────────────────────────────────
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 모델 하이퍼파라미터 ───────────────────────────────────────────────────
RF_PARAMS = dict(n_estimators=200, max_depth=10,
                 min_samples_leaf=5, n_jobs=-1, random_state=42)

# ── 결합 파라미터 ─────────────────────────────────────────────────────────
# road_struc_% 컬럼 인덱스 (landcover_static.npy: buildings%, green%, road%, river%)
ROAD_LC_IDX = 2   # road_struc_%

# 재비산 기여 스케일 — 실측 데이터로 calibration
# total_PM = ambient_PM + ALPHA * road_struc_% * road_PM
ALPHA = 0.5        # 초기값, calibrate_alpha()로 자동 조정

# LC 컬럼명
LC_COLS = ["buildings_%", "greenspace_%", "road_struc_%", "river_zone_%"]
