"""
HiddenExtension V5 — Fixed Effect + Hierarchical LUR 설정

데이터 분석 기반 설계 원칙:
  - station 간 PM 차이가 시간 변동의 1/10에 불과 (between std 2.55 vs within 24.43)
  - 고정 효과의 station-level 설명력: ~0.5% → 예측 MAE 개선은 제한적
  - 단, 그리드 추론에서 LUR 기반 공간 기저 예측은 여전히 유의미
  - 핵심 개선 방향: station별 편향 보정 + seasonal 상호작용
"""
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..")
LUR_BASE  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression"
GRID_DIR  = os.path.join(LUR_BASE, "격자 기본")

HIDDEN_DIR = os.path.join(ROOT_DIR, "HiddenExtension_V1/data/hidden_vectors")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 격자 데이터 경로 ──────────────────────────────────────────────────────────
GRID_CSV    = os.path.join(GRID_DIR, "격자_250m_4326.csv")
TIMESTAMPS  = os.path.join(GRID_DIR, "timestamps_all.npy")
NDVI_PATH   = os.path.join(GRID_DIR, "ndvi_hourly.npy")
IBI_PATH    = os.path.join(GRID_DIR, "ibi_hourly.npy")
LC_PATH     = os.path.join(GRID_DIR, "landcover_static.npy")
BLDG_PATH   = os.path.join(GRID_DIR, "building_stats_static.npy")
WIND_PATH   = os.path.join(GRID_DIR, "wind_all.npy")
TIME_IDX    = {
    "train": os.path.join(GRID_DIR, "time_idx_train.npy"),
    "val":   os.path.join(GRID_DIR, "time_idx_val.npy"),
    "test":  os.path.join(GRID_DIR, "time_idx_test.npy"),
}
STGNN_WINDOW = 12

# ── 모델 구조 ─────────────────────────────────────────────────────────────────
H_DIM     = 64    # ST-GNN hidden dimension
N_STATION = 40
LUR_DIM   = 9     # NDVI, IBI, LC(4), bldg(3)

# LUR 피처 이름
LUR_NAMES = ["NDVI", "IBI",
             "buildings_%", "greenspace_%", "road_struc_%", "river_zone_%",
             "elev_mean", "sum_area", "sum_height"]

# Temporal 피처 이름
TEMPORAL_NAMES = ["hour_sin", "hour_cos",
                  "month_sin", "month_cos",
                  "doy_sin", "doy_cos",
                  "is_weekend", "is_winter", "is_spring"]

# ── Ablation 실험 정의 ────────────────────────────────────────────────────────
# (exp_id, use_bias, fe_mode, use_hier_lur, mlp_layers, dropout)
# fe_mode: None | 'season' | 'monthly' | 'monthly+hour'
EXPERIMENTS = [
    # 기존 실험
    ("V5-base",         False, None,           False, [64, 32], 0.1),
    ("V5-bias",         True,  None,           False, [64, 32], 0.1),
    ("V5-season",       True,  "season",       False, [64, 32], 0.1),
    ("V5-hier",         True,  "season",       True,  [64, 32], 0.1),
    # 신규: monthly FE (season → month 세분화)
    ("V5-monthly",      True,  "monthly",      True,  [64, 32], 0.1),
    # 신규: monthly + hour-bin FE
    ("V5-monthly-hour", True,  "monthly+hour", True,  [64, 32], 0.1),
]

# ── Hold-out 검증용 geographic cluster ────────────────────────────────────────
# 서울 전역에 고르게 분포한 8개 station을 2개씩 묶어 4개 cluster
# → "이 지역에 관측소가 없다면?" 시나리오 검증
HOLDOUT_CLUSTERS = {
    "north":   [11, 4],    # 노원, 강북 (북부)
    "south":   [0,  22],   # 강남, 송파 (남부)
    "west":    [5,  25],   # 강서, 양천 (서부)
    "east":    [2,  8],    # 강동, 광진 (동부)
    "central": [33, 31],   # 중구, 종로 (도심)
}

# ── 학습 하이퍼파라미터 ───────────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
BATCH_SIZE   = 1024
PATIENCE     = 15
# station bias 정규화 강도 (과적합 방지)
BIAS_L2      = 0.01
