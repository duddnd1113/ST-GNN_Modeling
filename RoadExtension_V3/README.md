# RoadExtension V3

## 개요

V2 D_ambient 기반, **이상치 처리 재설계 + 7종 모델 종합 비교**.  
**최종 선택 모델: TwoStage (시간 LightGBM + 공간 Ridge)** — Test MAE=7.3383

---

## 최종 결론 먼저

```
최종 모델: TwoStage
Test MAE : 7.3383 μg/m³
Test R²  : 0.276

V1 (RF, 서울 평균): MAE=17.66  → V3: 58% 개선
```

**TwoStage를 최종 모델로 선택한 이유:**

| 기준 | LightGBM_BC (MAE 1위) | TwoStage (최종 선택) |
|------|---------------------|-------------------|
| Test MAE | **7.3326** (근소 우위) | 7.3383 (0.006 차이) |
| 격자 확장 | 측정 격자만 학습 → 외삽 불안정 | **LUR 피처로 10,125개 전체 예측** |
| 공간 해석 | 블랙박스 | **Ridge 계수로 공간 기여도 해석 가능** |
| 예측 연속성 | 불연속(트리 구조) | **선형 보간 → 공간 연속성 보장** |

MAE 차이(0.006)는 측정 오차 수준이므로, **격자 확장 능력이 결정적 차별점**.

---

## V2 vs V3 변경점

| 항목 | V2 | V3 |
|------|----|----|
| 이상치 처리 | 맥락적 IQR → 중앙값 대체 | **1-99 percentile clip** |
| 타깃 변환 | log1p | **Box-Cox (λ=-0.111, 데이터 최적화)** |
| 모델 | LightGBM 단일 | **7종 비교 → TwoStage 최종** |
| 공사 데이터 | 포함 (실험) | 미사용 |
| 피처 수 | 20개 | 21개 (Stage1: 10, Stage2: 11) |

---

## 이상치 처리 + 변환

### 1단계: 1-99 Percentile Clip
```
학습 데이터 기준:
  하위  1% (2.0 μg/m³) 미만 → 2.0 으로 대체
  상위 99% (78.0 μg/m³) 초과 → 78.0 으로 대체
```

### 2단계: Box-Cox 변환 (λ = -0.111)
```
y* = ((y + 0.01)^λ - 1) / λ      [λ = -0.111]
역변환: y = (y* × λ + 1)^(1/λ) - 0.01

변환 후: skewness = -0.003 (완벽한 대칭 분포)
```

**λ = -0.111 선택 근거:**
- 학습 데이터에서 scipy.stats.boxcox로 최적값 자동 탐색
- log1p(λ≈0)보다 이 데이터에 더 적합한 분포 압축
- 역변환 오차 < 0.000001 (수치 안정성 확인)

---

## 최종 모델: TwoStage 구조

### 핵심 아이디어

도로 재비산먼지는 **"언제(시간)"** 와 **"어디서(공간)"** 성분으로 분리 가능.

```
road_pm = 시간 성분 (날씨·계절로 결정)
        + 공간 잔차 (격자별 고정 특성으로 결정)
```

분산 분해 (ICC 분석):
- 구 간 분산(공간): 3.3% ← Stage2가 설명
- 구 내 분산(시간): 96.7% ← Stage1이 설명

### Stage 1: 시간 모델 (LightGBM)

```
입력 피처 (10개):
  기상: 기온, 습도, is_dry
  시간: month_sin, month_cos, hour_sin, hour_cos,
        weekday, is_weekend, season

목적: "오늘, 이 시간, 이 날씨에서 서울의 도로 PM10 기준값은?"
출력: Box-Cox 공간에서의 시간 기준값 (BC_temporal)
```

**Stage1 피처 중요도:**

| 피처 | 중요도 | 의미 |
|------|--------|------|
| 습도 | 1,098 | 건조할수록 재비산 증가 |
| 기온 | 996 | 저온 건조 조건 |
| hour_sin | 511 | 측정 시간대 |
| weekday | 480 | 요일별 교통 패턴 |
| month_sin/cos | 315/252 | 계절성 |

### Stage 2: 공간 잔차 모델 (Ridge Regression)

```
입력 피처 (11개):
  LUR 정적: buildings, greenspace, road_struc, river_zone,
             ndvi, ibi, elev_mean, sum_area, sum_height
  동적:      ambient_pm10 (V5 예측), traffic (격자별)

목적: "이 격자는 서울 평균 대비 얼마나 높은/낮은가?"
입력: Stage1 예측의 잔차 (road_pm_BC - BC_temporal)
출력: 공간 편차 (BC_spatial_deviation)
```

**Stage2 Ridge 계수:**

| 피처 | 계수 | 해석 |
|------|------|------|
| ndvi | +0.0839 | 식생 많은 격자 → 더 높은 도로 PM (도로 주변 흙먼지) |
| ibi | +0.0042 | 도심 불투수면 → 소폭 증가 |
| elev_mean | -0.0019 | 고지대 → 소폭 감소 |
| ambient_pm10 | +0.0008 | 배경 PM 높을수록 증가 |
| road_struc | -0.0007 | 도로 밀도 (약한 음의 관계) |

계수가 전반적으로 작음 → 공간 분산(3.3%) 자체가 작아 Stage2 기여도 제한적.  
그럼에도 Stage1 단독 7.40 → Stage1+2 **7.34** (0.06 MAE 개선).

### 예측 흐름

```
(date, hour, CELL_ID) → 피처 추출
    ↓
Stage1: f_temporal(기온, 습도, season, ...)
    = BC_temporal          ← Box-Cox 공간의 시간 성분
    ↓
Stage2: f_spatial(LUR, ambient_pm10, traffic, ...)
    = BC_deviation          ← 공간 잔차
    ↓
final_BC = BC_temporal + BC_deviation
    ↓
inverse_BoxCox(final_BC) → road_pm 예측값 (μg/m³)
    ↓
clip(road_pm, 0, ∞)       ← 음수 방지
```

### 격자 확장 적용

```
측정 없는 격자 (5,441개 추가) 예측 방법:

Stage1: 해당 날짜/시간의 기상 피처만 있으면 예측 가능
Stage2: 격자별 LUR 피처(정적)만 있으면 예측 가능
        → ambient_pm10은 V5 inference로 전체 격자에 생성됨
        → traffic은 grid_traffic_hourly.parquet에서 획득

∴ 모든 10,125개 격자에 대해 (T, G) 시계열 예측 가능
```

---

## 전체 실험 결과

### 7종 모델 비교

| 모델 | Train MAE | Test MAE | Test R² | 비고 |
|------|---------|---------|---------|------|
| **TwoStage** ✓ | 5.45 | **7.3383** | 0.276 | **최종 선택 (격자 확장)** |
| LightGBM_BC | 5.32 | 7.3326 | 0.286 | MAE 수치 최고 (0.006 차이) |
| XGBoost_BC | 5.78 | 7.3826 | 0.163 | — |
| LightGBM_Weighted | 5.10 | 7.4615 | **0.327** | R² 최고 |
| LinearSVR | 6.91 | 7.6156 | 0.153 | 선형 한계 |
| LightGBM_Huber | 5.38 | 7.7086 | 0.283 | Huber δ=8 |
| SVR_RBF | 4.06 | 9.7096 | -0.300 | 서브샘플 과적합 |
| V2 D_ambient (기준) | 5.57 | 7.3498 | 0.314 | V2 기준선 |

### TwoStage 상세 성능

**월별 MAE:**

| 월 | MAE | 건수 |
|----|-----|------|
| **1월** | **25.46** | 5,284 |
| **2월** | **18.65** | 5,066 |
| 3월 | 8.36 | 7,563 |
| 5월 | 5.78 | 5,520 |
| 7월 | 4.03 | 4,736 |
| 9월 | **4.04** | 12,453 |
| 11월 | 4.17 | 7,525 |

**PM 범위별 MAE:**

| 실측 범위 | 건수 | MAE | 예측 평균 | 실측 평균 |
|----------|------|-----|---------|---------|
| 0–5 μg/m³ | 21,583 | 5.54 | 9.0 | 3.5 |
| 5–10 | 23,954 | 2.80 | 9.9 | 7.6 |
| 10–20 | 17,739 | 5.29 | 13.4 | 14.5 |
| 20–40 | 8,041 | 11.19 | 18.6 | 28.5 |
| **>40** | **3,871** | **46.83** | **22.7** | **69.6** |

### 버전별 성능 추이

```
V1 (RF, 서울 평균 스칼라)          MAE=17.66  R²=0.135
V2 (LightGBM, IQR 처리)           MAE= 7.35  R²=0.314
V3 TwoStage (최종)                 MAE= 7.34  R²=0.276
────────────────────────────────────────────────
V1 대비 개선: -58%  │  V2 대비 개선: -0.2%
```

---

## 실험별 핵심 발견

| 실험 | 발견 |
|------|------|
| Box-Cox vs log1p | λ=-0.111 최적이지만 MAE 차이 0.017 (무의미) — 변환이 병목 아님 |
| Huber loss | clip과 결합 시 오히려 악화. clip 없이 전체 분포로 학습 시 유효 |
| Sample weight | MAE 악화, R² 개선 — 고농도 vs 저농도 예측 트레이드오프 |
| SVR_RBF | 서브샘플 과적합 심각. GPU cuML 없이 실용 불가 |
| TwoStage | Stage2 Ridge 기여 0.06 MAE. 격자 확장이 결정적 장점 |

---

## 성능 상한 분석

현재 설명 가능 분산: R²=0.276 → **미설명 72.4%**

| 미관측 변수 | 예상 기여 |
|------------|---------|
| 강수 이력 (마지막 비 이후 일수) | 높음 |
| 도로 청소 일정 | 높음 |
| 차량 종류 비율 (화물차 비중) | 중간 |
| 도로 포장 노후도 | 중간 |

---

## 실행

```bash
cd "/workspace/ST-GNN Modeling"

# V3 전체 모델 학습 (V2 피처 CSV 재사용, step1~4 불필요)
python3 RoadExtension_V3/step5_train.py

# 저장 모델 위치
# checkpoints/models/two_stage.pkl  ← 최종 모델
```

---

## 파일 구조

```
RoadExtension_V3/
├── config.py              ← 경로, 이상치 파라미터, 모델 하이퍼파라미터
├── preprocess.py          ← RoadPMTransformer (clip + Box-Cox fit/transform/inverse)
├── step5_train.py         ← 7종 모델 종합 비교 + 저장
├── step_construction.py   ← 공사 데이터 전처리 (V3 미사용)
├── README.md
└── checkpoints/
    ├── ablation_results.json     ← 전체 실험 결과 + Box-Cox 메타(λ, clip 범위)
    └── models/
        ├── two_stage.pkl         ← 최종 모델 (tr, m1_lgbm, imp2, m2_ridge)
        ├── lgbm_bc.pkl           ← LightGBM_BC (MAE 수치 최고)
        ├── lgbm_huber.pkl
        ├── lgbm_weighted.pkl
        ├── xgboost_bc.pkl
        ├── linear_svr.pkl
        └── svr_rbf.pkl
```

`two_stage.pkl` 구조: `(RoadPMTransformer, LGBMRegressor, SimpleImputer, Ridge)`

---

## 전체 프로젝트에서의 위치

```
ST-GNN (S3/static/w12)
  ↓ hidden vector h(T, N=40, d=64)
HiddenExtension V5-base
  ↓ ambient PM10 grid (T, G=10125)     ← 배경 농도 (시간×격자)
RoadExtension V3 — TwoStage
  ↓
  [Stage1] 시간 LightGBM
    입력: 기온, 습도, is_dry, season, month/hour 사이클, weekday
    → 서울 도로 PM 시간 기준값

  [Stage2] 공간 Ridge
    입력: LUR(건물/녹지/도로/하천), NDVI, IBI, 고도, 교통량, ambient_pm10
    → 격자별 공간 편차

  ↓
  격자별 도로 PM10 (T, G=10125)        ← 재비산 성분
  ↓
Combined PM10 grid (T, G=10125)        ← 최종 통합
  ↓
도로 청소 차량 경로 최적화
```
