"""
HiddenExtension 설정값.
"""
import os

LUR_BASE = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression"

# ── 경로 ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.join(LUR_BASE, "격자 기본")

STGNN_CHECKPOINT = os.path.join(
    BASE_DIR, "../checkpoints/window_12/S3_transport_pm10_pollutants/static/best_model.pt"
)
STGNN_SCENARIO = "S3_transport_pm10_pollutants"
STGNN_WINDOW   = 12

HIDDEN_DIR      = os.path.join(BASE_DIR, "data/hidden_vectors")
NDVI_PATH       = os.path.join(GRID_DIR, "ndvi_hourly.npy")
IBI_PATH        = os.path.join(GRID_DIR, "ibi_hourly.npy")
LC_PATH         = os.path.join(GRID_DIR, "landcover_static.npy")
GRID_CSV_PATH   = os.path.join(GRID_DIR, "격자_250m_4326.csv")
TIMESTAMPS_PATH = os.path.join(GRID_DIR, "timestamps_all.npy")
TIME_IDX = {
    "train": os.path.join(GRID_DIR, "time_idx_train.npy"),
    "val":   os.path.join(GRID_DIR, "time_idx_val.npy"),
    "test":  os.path.join(GRID_DIR, "time_idx_test.npy"),
}

# ── 모델 구조 ────────────────────────────────────────────────────────────────
H_DIM      = 64   # ST-GNN hidden (고정)
X_DIM      = 6    # NDVI + IBI + buildings + greenspace + road_struc + river_zone
R_DIM      = 16   # hidden 압축 차원 (ablation: 8 / 16 / 32 / 64)
ATT_HIDDEN = 32   # attention MLP hidden size
DROPOUT    = 0.1

# ── Joint training ────────────────────────────────────────────────────────────
LAMBDA = 0.5   # direct path 가중치 (ablation: 0.3 / 0.5 / 0.7)

# ── Ablation 설정 ─────────────────────────────────────────────────────────────
# x_mode   : 'all' | 'satellite' | 'landcover' | 'none'
#             all        → NDVI + IBI + buildings + greenspace + road + river (X_DIM=6)
#             satellite  → NDVI + IBI only                                    (X_DIM=2)
#             landcover  → buildings + greenspace + road + river only          (X_DIM=4)
#             none       → 공간변수 없음                                       (X_DIM=0)
X_MODE = 'all'

# lur_mode : 'linear' | 'mlp'
#             linear → PM = β@X + θ@r  (해석 가능한 LUR 구조)
#             mlp    → PM = MLP([X, r]) (비선형 상한 확인용)
LUR_MODE = 'linear'

# attn_mode: 'full' | 'spatial_only'
#             full         → score = MLP([dist, sin, cos, X_target, X_src])
#             spatial_only → score = MLP([dist, sin, cos])  (X 없이 위치만)
ATTN_MODE = 'full'

# ── 학습 ────────────────────────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 200
BATCH_SIZE   = 512
PATIENCE     = 20
