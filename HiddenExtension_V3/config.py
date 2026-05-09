"""
HiddenExtension V3 설정
Wind-aware IDW + Random Forest 기반 spatial PM prediction
"""
import os

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..")
LUR_BASE  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression"
GRID_DIR  = os.path.join(LUR_BASE, "격자 기본")

HIDDEN_DIR = os.path.join(ROOT_DIR, "HiddenExtension_V1/data/hidden_vectors")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# 격자 데이터
GRID_CSV_PATH        = os.path.join(GRID_DIR, "격자_250m_4326.csv")
TIMESTAMPS_PATH      = os.path.join(GRID_DIR, "timestamps_all.npy")
NDVI_PATH            = os.path.join(GRID_DIR, "ndvi_hourly.npy")
IBI_PATH             = os.path.join(GRID_DIR, "ibi_hourly.npy")
LC_PATH              = os.path.join(GRID_DIR, "landcover_static.npy")
BLDG_PATH            = os.path.join(GRID_DIR, "building_stats_static.npy")
WIND_PATH            = os.path.join(GRID_DIR, "wind_all.npy")
POPULATION_PATH      = os.path.join(GRID_DIR, "population_hourly.npy")
TIME_IDX = {
    "train": os.path.join(GRID_DIR, "time_idx_train.npy"),
    "val":   os.path.join(GRID_DIR, "time_idx_val.npy"),
    "test":  os.path.join(GRID_DIR, "time_idx_test.npy"),
}

# ── ST-GNN 설정 ───────────────────────────────────────────────────────────────
STGNN_WINDOW = 12   # hidden vector는 window 이후 타임스텝부터

# ── 피처 이름 ─────────────────────────────────────────────────────────────────
HIDDEN_DIM   = 64
LUR_COLS     = ["NDVI", "IBI",
                "buildings_%", "greenspace_%", "road_struc_%", "river_zone_%",
                "elev_mean", "sum_area", "sum_height"]
BLDG_COLS    = ["elev_mean", "sum_area", "sum_height"]
LC_COLS      = ["buildings_%", "greenspace_%", "road_struc_%", "river_zone_%"]
TEMPORAL_COLS = ["hour_sin", "hour_cos",
                 "month_sin", "month_cos",
                 "doy_sin", "doy_cos",
                 "is_weekend",
                 "is_winter", "is_spring"]

# ── Wind-aware IDW 파라미터 ───────────────────────────────────────────────────
IDW_SIGMA_D = 0.05   # 거리 감쇠 (위경도 단위, ~5km)
IDW_SIGMA_W = 1.0    # 풍향 정렬 가중치
IDW_K       = 10     # 최근접 K개 station만 사용 (속도 향상)

# ── RF 하이퍼파라미터 ─────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":    200,
    "max_depth":       None,
    "min_samples_leaf": 5,
    "max_features":    "sqrt",
    "n_jobs":          -1,
    "random_state":    42,
}

# ── Ablation 실험 정의 ────────────────────────────────────────────────────────
# (exp_id, use_hidden, idw_method, pca_k, use_lur, use_population, use_temporal)
EXPERIMENTS = [
    # ── 베이스라인 ─────────────────────────────────────────────────────────
    ("RF-S",           False, None,      None, True,  False, False),
    ("RF-ST",          False, None,      None, True,  False, True),
    # ── Hidden vector 단독 ────────────────────────────────────────────────
    ("RF-H",           True,  "wind",    None, False, False, False),
    ("RF-HT",          True,  "wind",    None, False, False, True),
    # ── 조합 실험 ────────────────────────────────────────────────────────
    ("RF-HS",          True,  "wind",    None, True,  False, False),
    ("RF-HST",         True,  "wind",    None, True,  False, True),
    ("RF-HSTP",        True,  "wind",    None, True,  True,  True),
    # ── PCA 차원 축소 ────────────────────────────────────────────────────
    ("RF-PCA8-HST",    True,  "wind",    8,    True,  False, True),
    ("RF-PCA16-HST",   True,  "wind",    16,   True,  False, True),
    # ── IDW 방법 비교 ────────────────────────────────────────────────────
    ("RF-HST-IDW",     True,  "idw",     None, True,  False, True),
    ("RF-HST-Nearest", True,  "nearest", None, True,  False, True),
]
