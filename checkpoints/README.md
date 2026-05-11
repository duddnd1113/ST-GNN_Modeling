# ST-GNN 실험 결과 분석

서울 PM10 예측을 위한 Spatio-Temporal GNN 전체 실험 결과 비교 분석.  
총 **78개 실험** — 10개 시나리오 × 3개 Graph Mode × 3개 Window 크기.

---

## 실험 설계 개요

### 3개 축

```
Window 크기:  12h / 24h / 48h  (과거 몇 시간을 입력으로 사용)
Graph Mode:  static / soft_dynamic / climatological
Scenario:    S1 ~ S10 (node feature 구성 변경)
```

### 모델 구조

```
입력: (Batch, Window, N=40, Feature)
  ↓
EdgeAwareGATConv × T  — 시간스텝별 공간 인코딩
  ↓
Per-node GRU          — 시간 패턴 인코딩
  ↓
h_i ∈ R^{64}          — station hidden vector
  ↓
Linear(64 → 1)        — PM10 예측
```

---

## 시나리오 정의

| 시나리오 | Node Feature | n_features | 특징 |
|----------|-------------|-----------|------|
| **S1** | PM10, 풍향, 풍속, u, v | 5 | 최소 구성 |
| **S2** | PM10(마스크), 풍향, 풍속, u, v | 5 | PM10 결측 시뮬레이션 |
| **S3** | PM10, SO2, CO, O3, NO2, 풍향, 풍속, u, v | 9 | **오염물질 전체 추가** |
| **S4** | PM10(마스크), SO2, CO, O3, NO2, 풍향, 풍속, u, v | 9 | S3 + PM10 마스크 |
| **S5** | PM10, SO2, CO, O3, NO2(마스크), 풍향, 풍속, u, v | 9 | 부분 오염물질 마스크 |
| **S6** | S3 + 요약 마스크 | 9 | 결측 패턴 요약 |
| **S7** | S3 + 기온, 습도, 기압 | 10 | 기상 추가 |
| **S8** | S3 + 강수 | 8 | 강수 추가 |
| **S9** | S3 + 기상 + 강수 | 13 | 기상 전체 |
| **S10** | S3 + 기상 + 강수 (요약 마스크) | 18 | 전체 피처 + 마스크 |

### Graph Mode 정의

| Mode | 설명 | Edge 수 |
|------|------|---------|
| **static** | 거리 10km 이내 양방향 edge, 고정 | 706 |
| **soft_dynamic** | static 구조 + 풍향 반대 edge 일시적으로 0 | 706 (동적) |
| **climatological** | 학습기간 평균 풍향 기준 단방향 edge만 | 396 |

---

## 전체 실험 결과

### Window=12 전체 (30개 실험)

| 순위 | 시나리오 | Graph Mode | MAE | RMSE | n_feat | 학습시간 |
|------|----------|-----------|-----|------|--------|---------|
| **1** | **S3_transport_pm10_pollutants** | **static** | **2.6144** | **3.7305** | 9 | 44min |
| 2 | S7_transport_pm10_weather | static | 2.6985 | 3.8029 | 10 | 32min |
| 3 | S2_transport_pm10_pm10mask | soft_dynamic | 2.7455 | 3.9452 | 5 | 45min |
| 4 | S5_transport_pm10_pollutants_allmask | soft_dynamic | 2.7533 | 3.9526 | 9 | 55min |
| 5 | S6_transport_pm10_pollutants_summarymask | static | 2.7672 | 3.8997 | 9 | 49min |
| 6 | S8_transport_pm10_rain | soft_dynamic | 2.7696 | 3.9734 | 8 | 46min |
| 7 | S4_transport_pm10_pollutants_pm10mask | static | 2.7810 | 3.9817 | 9 | 44min |
| 8 | S9_transport_pm10_weather_rain | static | 2.7953 | 3.9183 | 13 | 43min |
| ... | ... | ... | ... | ... | ... | ... |
| 21~30 | climatological 전체 | climatological | 3.19~3.55 | 4.84~5.21 | - | 33~43min |

### Window=24 전체 (Top 5)

| 순위 | 시나리오 | Graph Mode | MAE |
|------|----------|-----------|-----|
| 1 | S10_transport_all_summarymask | soft_dynamic | 2.6683 |
| 2 | S3_transport_pm10_pollutants | soft_dynamic | 2.7138 |
| 3 | S5_transport_pm10_pollutants_allmask | static | 2.7495 |
| 4 | S8_transport_pm10_rain | static | 2.7587 |
| 5 | S5_transport_pm10_pollutants_allmask | soft_dynamic | 2.7635 |

### Window=48 전체 (Top 5)

| 순위 | 시나리오 | Graph Mode | MAE |
|------|----------|-----------|-----|
| 1 | S5_transport_pm10_pollutants_allmask | static | 2.6311 |
| 2 | S3_transport_pm10_pollutants | static | 2.6611 |
| 3 | S10_transport_all_summarymask | soft_dynamic | 2.6703 |
| 4 | S8_transport_pm10_rain | static | 2.6707 |
| 5 | S2_transport_pm10_pm10mask | static | 2.7032 |

---

## 차원별 비교 분석

### 1. Window 크기 분석

```
Graph Mode 평균 MAE (시나리오 전체 평균):

             Window 12   Window 24   Window 48
static        2.775       2.848       2.772
soft_dynamic  2.861       2.829       2.835
climatological 3.299      3.317       3.441

결론:
  - Window 12가 static에서 가장 좋음 (2.775)
  - Window 48도 static에서 경쟁력 있음 (2.772)
  - 큰 window가 반드시 좋지 않음 — 12h면 충분
  - Window 크기보다 피처 구성이 더 중요
```

**해석:** PM10 예측에서 과거 12시간이 핵심 정보를 담고 있다.  
24~48시간 추가 입력은 노이즈를 증가시킬 수 있어 개선이 불규칙하다.

---

### 2. Graph Mode 분석

```
전체 78개 실험 Graph Mode별 요약:

Mode            평균 MAE    최저 MAE    특징
static           2.791      2.6144      안정적, 최고 성능 달성
soft_dynamic     2.842      2.6683      중간 수준, 일부 시나리오 유리
climatological   3.352      3.143       전반적으로 열등
```

**static이 best인 이유:**
```
static:
  거리 기반 양방향 edge → 모든 방향의 대기 이동 학습 가능
  학습 안정적 (edge 구조 고정)

soft_dynamic:
  실시간 풍향 반대 edge 마스킹 → 물리적 의미 있음
  하지만 마스킹으로 인한 정보 손실 가능

climatological:
  평균 풍향 기반 단방향 edge → edge 수 706 → 396 (44% 감소)
  희소한 그래프로 인한 정보 부족 → 성능 열등
```

**soft_dynamic이 W24/W48에서 경쟁력 있는 이유:**
```
긴 window에서 시간에 따른 풍향 변화가 누적됨
→ soft_dynamic의 동적 마스킹이 더 의미 있어짐
→ W12에서는 12시간의 풍향 변화가 평균화되어 효과 약함
```

---

### 3. 시나리오(피처) 분석

#### 오염물질 추가 효과

```
S1 (PM10만, 5 feat)          MAE = 2.812  (window 12, static)
S3 (PM10+오염물질, 9 feat)   MAE = 2.614  (window 12, static)
→ 오염물질 추가로 MAE 0.198 개선 (7.0% 개선)

이유:
  SO2, CO, O3, NO2는 PM10과 함께 이동하는 대기 오염물질
  → 이웃 station의 오염물질 패턴이 PM10 이동 예측에 도움
  → Edge attention에서 "어느 station에서 오염이 전달되는지" 더 잘 포착
```

#### 기상 변수 추가 효과

```
S3 (오염물질만)              MAE = 2.614
S7 (오염물질 + 기상)         MAE = 2.699
→ 기상 추가 시 MAE 오히려 악화 (+0.085)

S8 (오염물질 + 강수)         MAE = 2.770  (더 나빠짐)
S9 (오염물질 + 기상 + 강수)  MAE = 2.795

이유:
  기상 변수(기온, 습도)는 PM10과 상관이 간접적
  → 피처 차원 증가로 학습 난이도 상승
  → 풍향/풍속은 이미 기본 피처에 포함 (u, v 성분)
  → 추가 기상 정보의 marginal benefit < 학습 복잡도 증가
```

#### 마스킹 전략 분석

```
마스킹 = 특정 관측값이 결측일 때 0으로 대체 + 마스크 indicator 추가

S2 (PM10 마스크)     MAE = 2.746 ~ 2.806  (soft_dynamic/static)
S4 (PM10 마스크)     MAE = 2.781 ~ 2.816
S5 (전체 마스크)     MAE = 2.753 ~ 2.808
S6 (요약 마스크)     MAE = 2.767 ~ 2.994

결론:
  마스킹 전략은 기준(S3 MAE=2.614)보다 전반적으로 열등
  → 실측 데이터가 있을 때는 마스킹 불필요
  → 결측 상황을 가정한 robust 모델 학습 목적이라면 의미 있음
```

#### 피처 수와 성능의 관계

```
n_features →    5        9        10       13       18
최고 MAE  →  2.703    2.614    2.699    2.769    2.668

"피처가 많을수록 좋지 않다"
→ S3 (9 features)가 최적 sweet spot
→ 핵심 오염물질 5종 + 풍향풍속이 이미 충분한 정보
→ 그 이상은 오히려 학습 복잡도만 증가
```

---

## 최종 선택 모델 분석

### 선택: S3 / static / Window=12

```
시나리오: S3_transport_pm10_pollutants
  Node features: PM10, SO2, CO, O3, NO2, 풍향, 풍속, u, v (9차원)
  
Graph: static
  Edge 수: 706개 (양방향, 거리 10km 이내)
  
Window: 12시간

결과:
  MAE  = 2.6144 μg/m³  ← 전체 78개 실험 1위
  RMSE = 3.7305 μg/m³
  학습 시간: ~44분
  best_val_loss: 0.000145
```

### 최종 선택 근거

```
1위 선택의 핵심 이유:

① 오염물질 추가의 효과
   S1(PM10만) MAE=2.812 → S3(+오염물질) MAE=2.614
   → 가장 큰 개선 요인 (7.0% 개선)

② Static graph의 우위
   PM10 확산은 평균적으로 양방향 → static이 최적
   실시간 풍향보다 학습 안정성이 더 중요

③ Window 12가 충분
   PM10의 공간 전달 시간: 보통 수 시간 이내
   → 12시간 과거로 충분, 더 긴 window는 노이즈

④ 단순성과 성능의 균형
   9개 피처 (필수 정보만) → 학습 효율 최고
```

### 이 모델이 HiddenExtension V5에서 사용된 이유

```
HiddenExtension V1~V5는 S3/static/window=12의
hidden vector h_i (64차원)를 입력으로 사용.

이 hidden vector는:
  - 40개 관측소의 시공간 PM dynamics를 압축
  - 이웃 station의 오염물질 패턴까지 인코딩
  - 거리·풍향 기반 graph attention으로 공간 정보 통합

→ V5-base MLP가 이 hidden에서 PM10을 2.6036 MAE로 예측
   (ST-GNN 자체 예측 2.6144보다 개선)
```

---

## 실험 전체 요약 통계

```
전체 78개 실험:
  MAE 범위: 2.6144 ~ 3.5496 μg/m³
  중앙값:   2.831 μg/m³
  상위 25%: MAE < 2.771
  하위 25%: MAE > 3.110 (대부분 climatological)

Graph Mode별 MAE 평균:
  static:          2.791  (최고)
  soft_dynamic:    2.842
  climatological:  3.352  (최저, 26개 중 1위가 없음)

Window별 최고:
  12h: 2.6144  (S3/static)
  24h: 2.6683  (S10/soft_dynamic)
  48h: 2.6311  (S5/static)

시나리오별 최고:
  S3: 2.6144  ← 전체 1위
  S5: 2.6311
  S10: 2.6683
```

---

## 폴더 구조

```
checkpoints/
├── README.md               ← 이 파일
├── window_12/              ← 30개 실험
│   ├── S1_transport_pm10/
│   │   ├── static/
│   │   │   ├── best_model.pt
│   │   │   ├── metrics.json
│   │   │   └── loss_history.csv
│   │   ├── soft_dynamic/
│   │   └── climatological/
│   ├── S2 ~ S10/  (동일 구조)
│   └── ...
├── window_24/              ← 30개 실험 (동일 구조)
└── window_48/              ← 18개 실험 (동일 구조)

metrics.json 내용:
  scenario, graph_mode, mae, rmse
  best_val_loss, elapsed_min, epochs
  window, n_features, n_nodes, n_edges
```

---

## 결론

```
ST-GNN 실험의 핵심 결론:

1. 최적 구성: S3 / static / window=12
   → MAE = 2.6144 μg/m³ (전체 78개 중 1위)

2. 피처 구성이 가장 중요한 변수
   → 기본 교통 변수 + 핵심 오염물질 5종 (S3)이 최적
   → 기상/강수 추가는 오히려 성능 저하

3. Graph Mode: static > soft_dynamic >> climatological
   → 양방향 고정 그래프가 PM 확산 학습에 최적
   → climatological(단방향)은 정보 손실로 일관되게 열등

4. Window: 12시간이면 충분
   → PM10 공간 전달은 수 시간 내 완료
   → 더 긴 window는 marginal benefit 없음

5. 이 hidden vector가 V5까지 이어지는 기반
   → S3/static/w12의 h_i가 PM dynamics를 가장 잘 인코딩
   → V5-base에서 station MAE 2.6036 달성 (baseline 대비 개선)
```
