# HiddenExtension V3

## 개요

V1/V2에서 cross-attention 기반 extension이 ST-GNN baseline 대비 유의미한 개선을 내지 못한 원인을 분석하고,  
**RF(Random Forest) 기반 nonlinear PM prediction**으로 전환한 세 번째 버전.

핵심 전환: "신경망으로 공간 확장" → "물리 기반 보간 + RF 예측"

---

## V1/V2에서 얻은 교훈

| 발견 | 의미 |
|------|------|
| xnone(피처 없음)이 최고 direct MAE | LUR 피처가 hidden vector에 이미 내포 |
| Cross path가 Direct보다 28% 나쁨 | Grid 추론용 cross path 학습이 어려움 |
| Overfitting 심함 (gap ~14) | 40개 station으로 공간 일반화 한계 |
| 18개 중 1개만 baseline 개선 | Cross-attention 구조 자체의 한계 |

→ 복잡한 end-to-end 학습보다 **단순하고 해석 가능한 구조**가 더 효과적일 수 있음

---

## V3 구조

```
[Stage 1] Grid Hidden Vector 생성 (학습 불필요)
  ST-GNN hidden(station) → Wind-aware IDW → Grid hidden

[Stage 2] RF 학습 (station level)
  X = [h_station + LUR features + temporal features]
  y = PM10
  → RandomForestRegressor.fit(X, y)

[Stage 3] Grid PM Prediction
  X_grid = [h_grid(IDW) + LUR(grid) + temporal]
  PM_grid = rf.predict(X_grid)
```

---

## 실험 설계

### 축 1: Grid Hidden 생성 방법
| 방법 | 설명 |
|------|------|
| IDW | Inverse Distance Weighting |
| Wind-IDW | 풍향/풍속 고려한 IDW (주요 실험) |
| Nearest | 가장 가까운 station hidden 사용 |

### 축 2: RF 입력 피처 조합
| 실험 ID | 피처 | 목적 |
|---------|------|------|
| RF-H | hidden only | hidden 단독 영향 |
| RF-S | spatial LUR only | LUR 단독 베이스라인 |
| RF-HS | hidden + spatial | 결합 효과 |
| RF-HT | hidden + temporal | 시간 패턴 추가 |
| RF-HST | hidden + spatial + temporal | 전체 조합 |
| RF-PCA-H | PCA(hidden) + spatial + temporal | 차원 축소 효과 |

### 축 3: Hidden 차원 축소
- Raw hidden (d=64)
- PCA (k=8, 16, 32)
- 설명 분산비 기준 k 결정

### 축 4: Temporal Feature
```python
temporal_features = {
    'hour_sin', 'hour_cos',       # 시간 순환 인코딩
    'month_sin', 'month_cos',     # 월 순환 인코딩
    'doy_sin', 'doy_cos',         # 연중일 순환 인코딩
    'is_weekend',                  # 주말 여부
    'is_winter', 'is_spring',      # 계절 더미
}
```

---

## 파일 구조 (예정)

```
HiddenExtension_V3/
├── README.md               # 이 파일
├── config.py               # 경로 및 하이퍼파라미터
├── grid_hidden.py          # Wind-aware IDW hidden 생성
├── feature_builder.py      # RF 입력 피처 조합
├── train_rf.py             # RF 학습 및 ablation
├── inference_grid.py       # Grid-level PM map 생성
├── result_analysis.ipynb   # 결과 분석 노트북
└── checkpoints/            # 학습된 RF 모델 및 결과
    └── {exp_name}/
        ├── rf_model.pkl
        ├── metrics.json
        └── feature_importance.csv
```

---

## 기대 효과

1. **Overfitting 제어**: RF의 n_estimators, max_depth, min_samples_leaf로 조절
2. **Feature importance**: 어떤 hidden 차원/LUR 변수가 중요한지 직접 확인
3. **빠른 실험 사이클**: 신경망 재학습 불필요
4. **해석 가능성**: SHAP 값으로 예측 근거 분석 가능

---

## 비교 기준

| 모델 | MAE (예상) | 비고 |
|------|-----------|------|
| ST-GNN baseline | 2.6144 | 기준점 |
| HE V1 best | 2.6659 | cross-attn, 6D feat |
| HE V2 best | 2.6105 | cross-attn, xnone |
| **V3 목표** | **< 2.55** | RF 기반 |

---

## 향후 V4 방향 (참고)

V3가 성공적이면 다음 단계 고려:
- ST-GNN + Spatial Extension **joint fine-tuning** (Geographic LOO 전략)
- Neural Process 기반 spatial generalization
- Temporal MAE로 hidden representation 품질 개선
