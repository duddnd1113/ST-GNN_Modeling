# HiddenExtension V4

## 개요

V1~V3의 핵심 실패 원인인 **"ST-GNN frozen으로 인한 hidden 표현력 한계"** 를 해결하기 위해,  
ST-GNN을 Multi-head Joint Loss와 함께 **end-to-end fine-tuning** 하는 네 번째 버전.

---

## V1~V3에서 얻은 교훈

| 버전 | 방법 | 결과 | 문제 |
|------|------|------|------|
| V1 | Cross-attn + 6D LUR (frozen) | MAE 2.6659 | baseline 악화 |
| V2 | Cross-attn + 9D LUR (frozen) | MAE 2.6105 | 0.15% 개선뿐 |
| V3 | Wind-IDW + RF (frozen) | MAE 3.1230 | baseline 대비 20% 악화 |

공통 원인: **ST-GNN frozen → hidden이 spatial extension에 최적화 안 됨**

V3의 Feature importance: hidden 97%+, LUR ~0.7% → LUR 자체는 의미 없음

---

## V4 핵심 아이디어

```
L_total = L_forecast + λ₁·L_direct + λ₂·L_spatial_LOO

세 loss가 모두 ST-GNN encoder로 역전파
→ hidden vector가 처음부터 공간 일반화를 위해 학습됨
```

### Geographic Cluster LOO (V1/V2 random LOO와의 차이)

```
V1/V2: 매 샘플마다 1개 station을 random mask
        → 항상 39개 context로 1개 예측
        → 실제 grid 추론과 동떨어진 조건

V4: 서울을 7개 구역으로 나눠 한 구역(5~9개 station) 통째로 mask
        → 한 구역에 관측소가 없는 상황 학습
        → 실제 grid 추론 조건에 훨씬 가까움
```

---

## 구조

```
입력 시퀀스
    ↓
ST-GNN Encoder (fine-tune, lr 차등)
  GAT  : lr = 5e-5  (가장 조심)
  GRU  : lr = 1e-4
  Head1: lr = 1e-4
    ↓ h[B, N, 64]
    ├─ Head 1 (Forecast)  : h → PM[t+1]      L_forecast
    ├─ Head 2 (Direct)    : h → PM[t+1]      L_direct × λ₁=0.3
    └─ Head 3 (Spatial)   : cross_attn(h_ctx, Fourier(coord_tgt)) → PM_tgt
                            Geographic LOO    L_spatial × λ₂=0.5
```

---

## 2-Phase 학습 전략

| Phase | ST-GNN | Heads | Epochs | 목적 |
|-------|--------|-------|--------|------|
| Phase 1 | Frozen | 학습 | 30 | Head warm-up, ST-GNN 보호 |
| Phase 2 | Fine-tune | 학습 | 70 | Joint 최적화 |

Catastrophic Forgetting 방지:
- Phase 1에서 Head를 충분히 학습
- Phase 2에서 ST-GNN에 작은 lr 적용
- L_forecast가 기존 forecasting 능력 유지

---

## Geographic Clusters (서울 7개 구역)

| Cluster | 포함 station | 개수 |
|---------|------------|------|
| north | 강북·노원·도봉·성북·정릉로·화랑로 | 6 |
| northeast | 강동·광진·중랑·천호대로·홍릉로 | 5 |
| east | 동대문·성동·종로구·종로·청계천로·중구 | 6 |
| south | 강남·강남대로·도산대로·송파·서초·관악·금천·동작·동작대로 | 9 |
| west | 강서·공항대로·구로·시흥대로·양천·영등포·영등포로 | 7 |
| central | 마포·서대문·신촌로·용산·은평·한강대로 | 6 |
| corridor | 간선도로 측정소 | 5 |

---

## 파일 구조

```
HiddenExtension_V4/
├── README.md
├── config.py           ← 경로, loss 가중치, geo cluster 정의
├── geo_loo.py          ← Geographic LOO sampler
├── joint_model.py      ← JointSpatialSTGNN + SpatialCrossAttention
├── train_joint.py      ← 2-phase joint training
├── inference_grid.py   ← 학습된 spatial head로 grid PM 추론
└── checkpoints/
    ├── best_model.pt
    ├── metrics.json
    └── history.json
```

---

## 실행

```bash
# 전체 학습 (Phase 1 + 2)
python3 train_joint.py

# Phase 1만 (Head warm-up 확인용)
python3 train_joint.py --phase1_only

# Grid 추론 (학습 완료 후)
python3 inference_grid.py --split test
```

---

## 기대 목표

```
ST-GNN baseline       MAE = 2.6144
HE V2 best            MAE = 2.6105  (frozen, 0.15% 개선)
V4 목표               MAE < 2.55    (joint, 2%+ 개선)
```

V4가 성공적이면: Geographic LOO cluster 수 조정 + λ 튜닝으로 추가 개선 가능
