# HiddenExtension V5

## 개요

ST-GNN hidden vector 기반 PM10 예측을 **고정 효과(Fixed Effect) + 계층적 LUR Prior** 구조로 확장한 다섯 번째 버전.  
V1~V4 실험에서 얻은 교훈을 바탕으로 설계되었으며, **V5-base가 전체 버전 통틀어 그리드 추론 성능 최고**를 달성했다.

---

## 연구 배경 및 V1~V4 교훈

### 전체 버전 실험 흐름

| 버전 | 핵심 방법 | ST-GNN | station MAE | 비고 |
|------|----------|--------|------------|------|
| ST-GNN baseline | 직접 예측 | frozen | **2.6144** | 기준점 |
| V1 | Cross-attn + 6D LUR (공유 compressor) | frozen | 2.6659 | baseline 악화 |
| V2 | Cross-attn + 9D LUR (독립 compressor) | frozen | 2.6105 | 0.15% 개선 |
| V3 | Wind-IDW + RF | frozen | 3.1230 | RF 과적합 |
| V4 | Multi-loss Joint Fine-tuning | fine-tune | 4.1001 | 파국적 망각 |
| **V5** | **Fixed Effect + Hierarchical LUR** | **frozen** | **2.5813** | **전체 최고** |

### V1~V4에서 반복된 핵심 발견

```
1. LUR 공간 피처 → RF feature importance 0.7% (거의 무의미)
   → LUR은 "시간 변동 예측"이 아닌 "공간 기저 예측"에 써야 함

2. ST-GNN fine-tuning → forecasting 성능 파괴 (2.61 → 4.10)
   → ST-GNN은 frozen 상태가 최적

3. station-specific 파라미터 증가 → grid 일반화 감소
   → 단순한 모델이 오히려 grid 추론에 유리

4. 40개 station 한계:
   → between-station variance (std=2.55)가 within-station (std=24.43)의 1/10
   → 대부분의 variance는 시간 변동, 공간 차이는 작음
```

---

## 데이터 분석 기반 설계

### PM10 분산 분해

```
between-station std : 2.55 μg/m³  (station 간 평균 PM10 차이)
within-station std  : 24.43 μg/m³ (한 station의 시간 변동)
between/within 비율 : 0.10

→ 고정 효과의 이론적 설명력 한계: ~0.5%
→ 대부분의 variance는 이미 ST-GNN hidden이 설명
```

### LUR ~ station 효과 상관

```
LUR 변수 ~ station_mean 상관계수:
  NDVI        : r = -0.273
  road_struc% : r = +0.271
  elev_mean   : r = -0.287
  나머지       : |r| < 0.2

최대 r = 0.29 → LUR의 station 효과 예측력 매우 약함
```

### 월별·시간대별 패턴

```
월별 PM10 평균:
  최저 17.5 μg/m³ (9월) ~ 최고 51.9 μg/m³ (4월, 황사)
  → 월별 차이 34 μg/m³ >> station 간 차이 2.55 μg/m³

시간대별 평균:
  야간(0-6시): 31.2 / 오후(13-18시): 35.8 → 4.6 μg/m³ 차이

station monthly std: 10.4 μg/m³
  → station × 월 교호 효과 (seasonal FE로 포착 가능)
```

---

## 모델 구조

### FixedEffectPMModel

```
PM_{i,t} = dynamic(h_{i,t}, temporal_t)    ← 주 예측 (MLP)
          + bias_i                           ← station 편향 보정
          + temporal_FE_{i,t}               ← 시간 × station 교호

bias_i  = LUR_i @ γ + u_i                  ← 계층적 구조 (V5-hier만)
          γ  : LUR → 공간 기저 매핑 (학습)
          u_i: station 잔차 (L2 정규화)
```

### dynamic MLP 구조

```
입력: h_{i,t} (64차원) + temporal_t (9차원) = 73차원

temporal 피처:
  hour_sin, hour_cos    (하루 주기 sin/cos)
  month_sin, month_cos  (연 주기 sin/cos)
  doy_sin, doy_cos      (연중일 주기 sin/cos)
  is_weekend            (주말 여부)
  is_winter, is_spring  (계절 더미)

MLP:
  Linear(73 → 64) → ReLU → Dropout(0.1)
  Linear(64 → 32) → ReLU → Dropout(0.1)
  Linear(32 →  1)

파라미터 수: 6,784개
```

---

## Ablation 실험 전체 결과

### Station-level 성능

| 실험 | bias | FE 방식 | hier_LUR | test MAE | R² | vs baseline |
|------|------|---------|---------|---------|-----|------------|
| V5-base | ✗ | - | ✗ | 2.6036 | 0.8836 | ↑0.0108 |
| V5-bias | ✓ | - | ✗ | 2.6007 | 0.8839 | ↑0.0137 |
| V5-hier | ✓ | 4계절 | ✓ | 2.6007 | 0.8834 | ↑0.0137 |
| **V5-season** | ✓ | **4계절** | ✗ | **2.5813** | **0.8847** | **↑0.0331** |
| V5-monthly | ✓ | 12개월 | ✓ | 2.6117 | 0.8832 | ↑0.0027 |
| V5-monthly-hour | ✓ | 12개월+4시간 | ✓ | 2.6227 | 0.8819 | ↓0.0083 |
| V5-hier-deep | ✓ | 4계절 | ✓ | 2.6669 | 0.8816 | ↓0.0525 |

**ST-GNN baseline: 2.6144**

### Hold-out 성능 (그리드 추론 기대 성능)

서울 5개 권역 10개 station hold-out → IDW hidden + LUR만으로 예측

| 실험 | station MAE | holdout MAE | gap | holdout R² |
|------|------------|------------|-----|-----------|
| **V5-base** | 2.6036 | **3.1842** | +0.58 | **0.8246** |
| V5-monthly | 2.6117 | 3.5277 | +0.92 | 0.7884 |
| V5-monthly-hour | 2.6227 | 3.6168 | +0.99 | 0.7796 |
| V5-bias | 2.6007 | 3.7039 | +1.10 | 0.7730 |
| V5-season | 2.5813 | 3.7335 | +1.15 | 0.7710 |
| V5-hier | 2.6007 | 4.0388 | +1.44 | 0.7326 |

**권역별 hold-out MAE (V5-base):**
```
central (중구·종로): 2.86  ← 도심, 주변 station 밀도 높아 IDW 정확
north   (노원·강북): 2.99
south   (강남·송파): 3.04
west    (강서·양천): 3.43
east    (강동·광진): 3.60  ← 동부 외곽, station 밀도 낮아 오차 큼
```

---

## 핵심 발견: Station 성능과 Grid 성능의 역전

```
station MAE가 좋을수록 hold-out MAE가 나빠지는 역설:

V5-base   : station 2.6036  holdout 3.1842  gap +0.58  ← grid 최고
V5-season : station 2.5813  holdout 3.7335  gap +1.15  ← station 최고

이유:
  V5-base: station-specific 파라미터 없음
    → 학습·추론 모두 동일한 경로 (h_idw + temporal → MLP)
    → IDW hidden 품질이 충분하면 잘 작동

  V5-season: bias_i + seasonal_{i,s} (station 고유값 암기)
    → 학습 station의 고유 패턴 학습
    → grid 추론 시 bias=0, seasonal=전체평균으로 대체
    → 암기한 정보를 버리니 오히려 나빠짐
```

---

## 계층적 회귀(V5-hier)가 실패한 이유

이론적으로는 올바른 방향이나, 데이터가 뒷받침하지 못했다.

```
V5-hier 구조:
  α_i = LUR_i @ γ + u_i
  γ   : LUR → station baseline 매핑
  u_i : residual

실패 원인:
  1. LUR → station_effect 상관 r=0.29 (너무 약함)
     → γ(LUR_g)만으로는 grid baseline 예측 부정확
  
  2. u_i가 40개 station 과적합
     → grid 추론 시 u_g=0 → 학습 정보 손실 → holdout 악화
  
  3. V5-hier holdout MAE=4.04 → 전체 최악

계층적 회귀가 효과적이려면:
  - station 수 ≥ 200 (현재 40개)
  - LUR → station_effect 상관 ≥ 0.7 (현재 0.29)
  → 더 많은 station 데이터 확보 시 재시도 가치 있음
```

---

## 최종 모델: V5-base

### 용도별 추천 모델

| 용도 | 추천 모델 | MAE |
|------|---------|-----|
| station-level 성능 보고 | **V5-season** | 2.5813 |
| grid PM map 생성 (실제 활용) | **V5-base** | holdout ~3.18 |

### V5-base 전체 구조

```
[Stage 1] ST-GNN (완전 frozen)
  입력: (B, T=12, N=40, F=9)  — 관측소 센서 12시간
  GAT (공간) + GRU (시간)
  출력: h_{i,t} ∈ R^{64}
  → 사전 추출: h_train/val/test.npy  shape=(T, 40, 64)

[Stage 2] V5-base MLP (학습 대상, 6,784 파라미터)
  입력: [h_{i,t}(64) || temporal_t(9)] = 73차원
  MLP: 73 → 64 → 32 → 1
  출력: PM_{i,t}  (스칼라)

[Stage 3] Grid 추론 (학습 없음)
  h_{station,t} (40, 64)
      ↓ Wind-aware IDW (상위 k=10 station, 거리+풍향 가중치)
  h_{grid,t} (10125, 64)
      ↓ V5-base MLP (동일 가중치)
  PM_{grid,t} (10125,)  — 서울 전역 250m 격자
```

### 학습 데이터

```
기간: 2023-10-01 ~ 2025-07-09  (학습+검증)
테스트: 2025-07-10 ~ 2025-10-31
샘플: 12,788 timestep × 40 station = 511,520 샘플
```

---

## Grid 추론 결과

```
grid_pm10_test.npy  shape: (2732, 10125)
기간: 2025-07-10 ~ 2025-10-31 (2,732시간)
격자: 서울 전역 250m × 250m = 10,125개

PM10 통계:
  mean = 18.45 μg/m³
  std  = 9.46 μg/m³
  min  = 4.45 μg/m³
  max  = 68.89 μg/m³
```

---

## 성능 평가 한계

```
station MAE (2.6036):
  → 학습에 쓰인 40개 station 기준, 낙관적 수치

Hold-out MAE (3.1842):
  → 5개 권역 10개 station hold-out
  → 실제 grid 추론에서 기대할 수 있는 MAE 추정치

진정한 grid MAE:
  → 측정 불가 (grid에 ground truth 없음)
  → hold-out MAE를 상한 추정치로 사용
  → 권역별 차이 존재 (도심 2.86 vs 동부 외곽 3.60)
```

---

## 파일 구조

```
HiddenExtension_V5/
├── README.md
├── config.py              ← 경로, 모델 설정, HOLDOUT_CLUSTERS, EXPERIMENTS
├── dataset.py             ← V5Dataset (h + LUR + temporal + season + month + hour_bin)
├── model.py               ← FixedEffectPMModel (dynamic + bias + temporal FE + γ)
├── train.py               ← 학습 루프 + ablation 전체 실행
├── holdout_eval.py        ← Geographic hold-out 검증 (그리드 성능 추정)
├── inference_grid.py      ← Wind-IDW → grid hidden → grid PM10
├── visualize_grid.ipynb   ← 날짜/시간 선택형 시각화 노트북
└── checkpoints/
    ├── results_summary.csv      ← ablation 전체 결과
    ├── holdout_summary.csv      ← hold-out 검증 요약
    └── {exp_id}/
        ├── best_model.pt
        ├── metrics.json
        ├── history.json
        ├── holdout_metrics.json        ← hold-out 결과 (권역별)
        ├── gamma_lur.csv               ← LUR γ 계수 (V5-hier만)
        ├── grid_pm10_test.npy          ← (T, G) grid 예측값 (inference 후)
        └── grid_pm10_mean_test.csv     ← 격자별 기간 평균
```

---

## 실행

```bash
cd "/workspace/ST-GNN Modeling"

# 1. 전체 ablation 학습
python3 HiddenExtension_V5/train.py --run_all

# 2. 단일 실험
python3 HiddenExtension_V5/train.py --exp V5-base

# 3. Hold-out 검증
python3 HiddenExtension_V5/holdout_eval.py --run_all
python3 HiddenExtension_V5/holdout_eval.py --exp V5-base

# 4. Grid PM10 추론 (V5-base)
python3 HiddenExtension_V5/inference_grid.py --exp V5-base

# 5. 시각화 (Jupyter)
# visualize_grid.ipynb 실행 — 날짜/시간 선택 후 셀 실행
```

---

## 시각화 방법

`visualize_grid.ipynb`에서 2셀의 날짜/시간을 바꾸면 원하는 시점의 PM10 지도를 확인할 수 있다.

```python
TARGET_DATE = '2025-08-15'   # YYYY-MM-DD (테스트 기간 내)
TARGET_HOUR = 9              # 0~23
```

**시각화 모드:**

| 모드 | 설명 | 추천 상황 |
|------|------|----------|
| `percentile` | 해당 시점 p5~p95 기준 자동 조정 | 분포 차이 강조 (기본 권장) |
| `power` | 비선형 스케일 (γ=0.4) | 저농도 지역 세밀한 차이 |
| `deviation` | 공간 평균 편차 표시 | 상대적 고/저오염 지역 비교 |
| `absolute` | 절대값 고정 범위 | 시간별 비교 (동일 스케일 유지) |

---

## 최종 결론

### 수치 요약

```
               Station MAE    Hold-out MAE    R²(station)
ST-GNN base:    2.6144            -              -
V5-season:      2.5813          3.7335         0.8847   ← station 최고
V5-base:        2.6036          3.1842         0.8836   ← grid 최고
```

### 핵심 결론

1. **ST-GNN hidden vector가 PM10의 대부분을 설명**
   - 40개 station, between-station variance = 0.5% 수준
   - 어떤 추가 구조를 붙여도 획기적 개선은 어려움

2. **단순한 모델이 Grid 일반화에 유리**
   - station-specific 파라미터 추가 → station MAE ↓, holdout MAE ↑
   - V5-base (MLP only)가 holdout 최고: IDW hidden의 일반화가 핵심

3. **계층적 LUR은 이론적으로 옳지만 데이터 한계**
   - LUR-station 상관 r=0.29, station 수 40개로 부족
   - station 수 ≥ 200 환경에서 재시도 가치 있음

4. **최종 선택: V5-base**
   - station MAE 2.6036 (baseline 2.6144 대비 1.3% 개선)
   - holdout MAE 3.1842 (그리드 추론 기대값)
   - 6,784 파라미터로 경량, 해석 용이, 빠른 추론
   - 서울 전역 250m 격자 2,732시간 PM10 지도 생성 완료
