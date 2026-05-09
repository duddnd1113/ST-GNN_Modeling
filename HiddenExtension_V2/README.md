# HiddenExtension V2

## 개요

V1의 구조적 한계(공유 compressor, 6차원 피처)를 개선한 두 번째 버전.  
Direct/Cross path에 **독립 compressor** 적용, LUR 피처를 **9차원**으로 확장.  
18개 ablation 실험을 통해 각 설계 축의 영향 분석.

---

## V1 대비 변경사항

| 항목 | V1 | V2 |
|------|----|----|
| LUR 피처 차원 | 6 | **9** (+elevation, sum_area, sum_height) |
| Compressor | Direct/Cross 공유 | **Direct/Cross 독립** |
| base r_dim | 16 | **8** |
| base lam | 0.5 | **0.3** |
| x_mode 옵션 | all/satellite/landcover/none | **all/no_building/satellite/none** |
| 실험 수 | 11개 | **18개** |

---

## 구조

```
ST-GNN (frozen)
    ↓ h_target, h_sources

Direct path:
  compressor_direct(h_target) → r_direct → θ(r) + β(X) → pm_direct

Cross path (LOO):
  cross_attention(h_sources, coords, X, wind) → h_cross
  compressor_cross(h_cross) → r_cross → θ(r) + β(X) → pm_cross

L = λ · L_direct + (1-λ) · L_cross
```

**Grid inference 시**: Cross path만 사용 (h_target 없음)

---

## Ablation 축

| 축 | 옵션 |
|----|------|
| x_mode | all(9차원) / no_building(6차원) / satellite(2차원) / none(0차원) |
| attn_mode | full(위치+피처+풍향) / spatial_only(위치만) |
| lur_mode | linear / mlp |
| r_dim | 8 / 16 / 32 |
| lam | 0.2 / 0.3 / 0.5 / 0.7 |

---

## 전체 실험 결과 (18개, direct MAE 순)

**ST-GNN baseline MAE: 2.6144**

| 순위 | 실험 | direct MAE | vs baseline | cross MAE |
|------|------|-----------|-------------|-----------|
| 1 | xnone, full, linear, r8, lam0.3 | **2.6105** | **↑0.0039** | 3.3546 |
| 2 | xno_building, full, linear, r8, lam0.3 | 2.6446 | ↓0.0302 | 3.3300 |
| 3 | xsatellite, full, linear, r8, lam0.5 | 2.6455 | ↓0.0311 | 3.3080 |
| 4 | xall, full, linear, r16, lam0.3 | 2.6509 | ↓0.0365 | **3.2571** |
| ... | ... | ... | ... | ... |
| 18 | xall, full, mlp, r8, lam0.3 | 2.8510 | ↓0.2366 | 3.2831 |

→ **18개 중 1개만 baseline 개선**, 나머지 17개 악화

---

## 주요 발견

### 1. LUR 피처 없는 것(xnone)이 최고 성능
spatial feature를 추가할수록 오히려 성능 저하 → 피처가 ST-GNN hidden vector에 이미 내포된 정보를 추가하거나 noise를 유입

### 2. Cross path가 Direct path보다 현저히 나쁨
```
Direct MAE (best): 2.6105  ← 학습 시 사용
Cross  MAE (best): 3.2571  ← Grid 추론 시 사용 (28% 나쁨)
```
→ 실제 grid 추론 품질이 생각보다 낮을 수 있음

### 3. 심각한 Overfitting
```
train direct loss: ~28  (계속 감소)
val   direct loss: ~42  (flat, 개선 없음)
gap: ~14  → 전형적인 과적합
```

### 4. lam=0.3이 cross MAE 개선에 유리하지 않음
cross path를 더 강조(lam=0.3 → direct 가중치 낮음)해도 cross MAE는 큰 개선 없음

---

## 학습 설정

| 항목 | 값 |
|------|-----|
| Epochs | 200 (early stop patience=20) |
| LR | 1e-3 |
| Batch size | 512 |
| Optimizer | Adam (weight_decay=1e-4) |
| Loss | MSELoss |
| ST-GNN | frozen |

---

## 한계 및 분석

1. **40개 station으로 공간 일반화 한계**: 위치 다양성이 40개뿐
2. **Cross-attention이 spatial interpolation 역할 미흡**: LOO 구조가 grid 추론과 완전히 일치하지 않음
3. **ST-GNN frozen으로 인한 표현력 제약**: extension에 최적화된 hidden 생성 불가

---

## 산출물

| 파일 | 설명 |
|------|------|
| `checkpoints/{exp}/best_model.pt` | 학습된 모델 가중치 |
| `checkpoints/{exp}/metrics.json` | 테스트 MAE/RMSE/R² |
| `checkpoints/{exp}/history.json` | epoch별 학습 곡선 |
| `checkpoints/{exp}/lur_coefficients.json` | β 가중치 (linear, x≠none만) |
| `result_analysis.ipynb` | 전체 결과 분석 노트북 |
| `summarize_results.py` | 결과 요약 스크립트 |

---

## V3에서 개선 방향

V1/V2 모두 cross-attention 기반 extension이 baseline 대비 개선을 거의 못 했음.  
→ V3는 **RF(Random Forest) 기반** 접근으로 전환:

1. **Grid hidden 생성**: Wind-aware IDW interpolation
2. **PM 예측**: RF([grid hidden + LUR + temporal features])
3. **비교 실험**: Joint fine-tuning + Geographic LOO 전략

자세한 내용은 `HiddenExtension_V3/README.md` 참고.
