"""
HiddenExtension V4 설정
ST-GNN + Multi-head Joint Fine-tuning
"""
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, "..")
LUR_BASE  = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/Land Use Regression"
GRID_DIR  = os.path.join(LUR_BASE, "격자 기본")

# ── 기존 ST-GNN ───────────────────────────────────────────────────────────────
STGNN_CKPT     = os.path.join(ROOT_DIR, "checkpoints/window_12/S3_transport_pm10_pollutants/static/best_model.pt")
STGNN_SCENARIO = "S3_transport_pm10_pollutants"
STGNN_WINDOW   = 12
SCENARIO_DIR   = "/home/data/youngwoong/ST-GNN_Dataset/Data_Preprocessed/ST-GNN/feature_scenarios"
SPLIT_INFO     = os.path.join(ROOT_DIR, "split_info.pkl")
GRAPH_DIR      = os.path.join(ROOT_DIR, "graphs")

# ── ST-GNN 모델 구조 (기존과 동일하게 유지) ────────────────────────────────
STGNN_PARAMS = dict(
    node_dim=9, edge_dim=5, gat_hidden=64, gru_hidden=64, num_heads=4, num_nodes=40
)
H_DIM = 64    # GRU hidden = hidden vector dimension

# ── 경로 ─────────────────────────────────────────────────────────────────────
HIDDEN_DIR  = os.path.join(ROOT_DIR, "HiddenExtension_V1/data/hidden_vectors")
GRID_CSV    = os.path.join(GRID_DIR, "격자_250m_4326.csv")
TIMESTAMPS  = os.path.join(GRID_DIR, "timestamps_all.npy")
NDVI_PATH   = os.path.join(GRID_DIR, "ndvi_hourly.npy")
IBI_PATH    = os.path.join(GRID_DIR, "ibi_hourly.npy")
LC_PATH     = os.path.join(GRID_DIR, "landcover_static.npy")
BLDG_PATH   = os.path.join(GRID_DIR, "building_stats_static.npy")
WIND_PATH   = os.path.join(GRID_DIR, "wind_all.npy")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 서울 Geographic Cluster (LOO 전략용) ────────────────────────────────────
# 인접 station끼리 묶어 한 cluster 전체를 mask → 공간 일반화 학습
GEO_CLUSTERS = {
    "north":    [4, 11, 12, 21, 30, 39],          # 강북·노원·도봉·성북·정릉로·화랑로
    "northeast":[2, 8, 34, 35, 38],               # 강동·광진·중랑·천호대로·홍릉로
    "east":     [14, 20, 32, 31, 36, 33],         # 동대문·성동·종로구·종로·청계천로·중구
    "south":    [0, 1, 13, 22, 19, 7, 10, 15, 16],# 강남·강남대로·도산대로·송파·서초·관악·금천·동작·동작대로
    "west":     [5, 6, 9, 23, 25, 26, 27],        # 강서·공항대로·구로·시흥대로·양천·영등포·영등포로
    "central":  [17, 18, 24, 28, 29, 37],         # 마포·서대문·신촌로·용산·은평·한강대로
    "corridor": [3, 36, 16, 27, 37],              # 간선도로 측정소
}

# ── Joint Training 하이퍼파라미터 ────────────────────────────────────────────
# Phase 1: ST-GNN frozen, head만 학습
PHASE1_EPOCHS   = 30
PHASE1_LR_HEAD  = 1e-3

# Phase 2: ST-GNN fine-tune + head joint 학습
PHASE2_EPOCHS       = 70
PHASE2_LR_HEAD      = 3e-4
PHASE2_LR_GRU       = 1e-4   # GRU: 조심스럽게
PHASE2_LR_GAT       = 5e-5   # GAT: 더 조심스럽게

# Loss 가중치 (L = L_forecast + λ1·L_direct + λ2·L_spatial)
LAMBDA_DIRECT  = 0.3
LAMBDA_SPATIAL = 0.5

# Spatial head 구조
ATT_HIDDEN  = 32
R_DIM       = 32    # hidden 압축 차원 (V2보다 크게)
DROPOUT     = 0.1
N_HEADS_ATT = 4     # multi-head attention

# 학습 공통
BATCH_SIZE  = 32
PATIENCE    = 15
WEIGHT_DECAY = 1e-4
