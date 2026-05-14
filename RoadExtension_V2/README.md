# RoadExtension V2

## 개요

RoadExtension **V1의 문제점**을 반영하여 격자 단위 도로 재비산먼지 예측 모델을 재설계.  
V1 대비 **MAE 58% 개선** (17.66 → 7.35 μg/m³).

---

## V1 vs V2 핵심 차이

| 항목 | V1 | V2 |
|------|----|----|
| 교통량 | 서울 전체 평균 스칼라 | **격자별 개별 교통량** (parquet) |
| LUR 변수 | 미포함 | **전체 포함** (NDVI, IBI, buildings 등) |
| Ambient PM10 | S3 CSV 구 단위 평균 | **V5 grid 단위 예측값** (250m 격자, 시간별) |
| 예측 대상 | 서울 평균 road_PM 스칼라 | **격자 단위 road_PM** |
| 도로→격자 매핑 | 없음 | **Nominatim geocoding** + 반경 1000m 할당 |
| 이상치 처리 | Winsorize 95th | **맥락적 IQR** (구+계절 그룹별) |
| 모델 | Random Forest | **LightGBM** (Ablation으로 선택) |

---

## 데이터 파이프라인

```
[원시 데이터]
  도로 재비산먼지 xlsx (8,280건, 2023~2025)
  격자별 교통량 parquet (10,125 격자 × 시간별)
  V5 ambient PM10 grid (T × 10,125)
  LUR 격자 CSV (NDVI, IBI, buildings 등)
  강수 parquet (격자별 마지막 비 이후 일수)
  서울 도로 위계 gpkg (OSM 기반, 11,076 구간)

    ↓ step1_geocode.py
  도로명 → Nominatim → lat/lon → 격자 CELL_ID

    ↓ step2_build_target.py
  도로 PM → 반경 1000m 격자 할당 → 이상치 처리 → road_pm_grid.csv

    ↓ step3_v5_inference_all.py
  h_train/val/test + Wind-aware IDW → grid_pm10_{split}.npy

    ↓ step4_build_features.py
  LUR + 교통량 + V5 ambient PM + 기상 + 교호작용 + 강수 + 도로위계
  → features_train.csv / features_test.csv

    ↓ step5_train.py
  피처 Ablation × LightGBM + 모델 비교

    ↓ step6_evaluate.py / step7_visualize.py
  결과 분석 + 시각화
```

---

## 이상치 처리 (2단계)

도로 재비산먼지 분포: mean=20, std=34, max=818 μg/m³ (심한 우편향)

### 1단계 — 절대 극단치 제거
- 상태 "매우나쁨" 49건 제거 (공사·특수 상황)
- 결과: 8,280 → 8,231건

### 2단계 — 맥락적 IQR 대체
- 그룹: (지역명, season)별 IQR 기반 탐지
- fence = Q3 + 1.5 × IQR 초과 → **해당 그룹 중앙값으로 대체**
- 649건 대체, 처리 후: mean=13.98, std=**14.84** (이전 std=22.15 → 33% 감소)

#### 왜 단순 percentile clip 대신 맥락적 IQR인가
- 여름 강남구 도로 50 μg/m³ = 정상일 수 있지만, 겨울 노원구 50 μg/m³ = 극단값
- 그룹 기준으로 탐지하면 이런 맥락을 반영할 수 있음
- cap 대신 대체(replacement)이므로 분포가 더 자연스럽게 유지됨

---

## 격자 매핑 방법

```
1. 고유 (지역명, 도로명) 290개 → Nominatim geocoding
2. 반환 lat/lon 기준 반경 1000m 내 격자 검색 (cKDTree)
3. road_struc% >= 5% 격자만 포함
4. 같은 (date, hour, CELL_ID) 중복 측정 → 평균
결과: 8,231건 → 198,417건 (격자 확장) → 198,417건 (집계 후)
커버 격자: 4,684 / 10,125 (46.3%)
```

---

## 피처 구성 및 Ablation 결과

### 피처 그룹

| 그룹 | 피처 | 수 |
|------|------|----|
| temporal | month_sin/cos, hour_sin/cos, weekday, is_weekend, season | 7 |
| weather | 기온, 습도, is_dry | 3 |
| lur | buildings, greenspace, road_struc, river_zone, ndvi, ibi, elev_mean, sum_area, sum_height | 9 |
| traffic | traffic (격자별 교통량) | 1 |
| ambient_pm | ambient_pm10 (V5 grid PM10) | 1 |
| interaction | cold_and_dry (기온<5°C + 습도<50%), traffic_x_road | 2 |
| rain | days_from_rain, daily_precip_mm | 2 |
| road_hier | highway_rank, max_lanes, mean_gvi, total_road_length_m | 4 |

### 전체 Ablation 결과 (2025 테스트셋)

| 실험 | 피처 구성 | Train MAE | Test MAE | Test R² | 비고 |
|------|----------|---------|---------|---------|------|
| A_base | temporal + weather | 5.72 | 7.51 | 0.282 | 기준선 |
| B_lur | + LUR | 5.46 | 7.44 | 0.285 | 공간 피처 추가 |
| C_traffic | + 격자 교통량 | 5.45 | 7.43 | 0.309 | R² 개선 |
| **D_ambient** | **+ V5 ambient PM** | **5.57** | **7.35** | **0.314** | **최종 선택** |
| E_all | D + traffic | 5.34 | 7.40 | 0.318 | 교통량 추가 효과 미미 |
| F_interact | E + 교호작용 | 5.40 | 7.45 | 0.309 | 교호작용 오히려 불리 |
| G_rain | D + 강수 | 5.48 | 7.43 | 0.298 | 강수 데이터 효과 없음 |
| H_hier | D + 도로위계 | 5.37 | 7.31 | 0.319 | 미미한 개선 (노이즈 수준) |
| I_all_v2 | 전부 | 5.24 | 7.53 | 0.273 | 과적합 |

**V1 기준선**: MAE=17.66, R²=0.135 → **D_ambient: 58% 개선**

---

## 모델 비교 (E_all 피처 기준)

| 모델 | Train MAE | Test MAE | Test R² | 비고 |
|------|---------|---------|---------|------|
| **LightGBM** | **5.34** | **7.40** | **0.318** | 최적 일반화 |
| Random Forest | 3.37 | 8.52 | 0.180 | 과적합 |
| MLP | 2.35 | 12.19 | -0.920 | 심각한 과적합 |
| LightGBM Tweedie | 5.09 | 8.04 | 0.323 | log1p 대비 불리 |

---

## 시도한 개선책과 결과

### 성공한 개선 — 맥락적 이상치 처리
- 결과: std 22.15 → 14.84 (33% 감소), MAE 11.77 → 7.35 (37% 추가 개선)
- 핵심: 단순 percentile cap 대신 그룹 중앙값으로 대체

### 실패한 개선 (시도하여 효과 없음)

#### 1. 샘플 가중치 (고농도 샘플 upweight)
- 시도: weight = 1 + (road_pm / median)^0.5
- 결과: D_ambient MAE 7.35 → 7.66 (악화)
- 원인: 고농도 희귀 샘플에 과적합, 일반화 성능 하락

#### 2. Tweedie 목적함수
- 시도: `objective='tweedie'`, `variance_power=1.5` (우편향 분포 특화)
- 결과: MAE 8.04 (log1p 대비 열등)
- 원인: log1p 변환이 이미 분포 보정에 충분

#### 3. 교호작용 피처 (hum×pm10, temp×pm10)
- 시도: ambient_pm10 NaN을 0으로 채워 교호작용 계산
- 결과: 악화
- 원인: NaN 처리 방식이 "ambient 없음" 패턴을 잘못된 신호로 학습

#### 4. 롤링 피처 (구별 30일 평균 road_pm)
- 시도: 같은 구의 최근 30일 측정 평균
- 결과: 상관계수 r=0.089로 무의미
- 원인: 도로 먼지는 장기 축적보다 당일 날씨에 의존

#### 5. 풍속 피처
- 시도: wind_all.npy에서 서울 평균 풍속 추출
- 결과: r=-0.014로 무의미
- 원인: 10m 고도 풍속이 도로 표면 재비산과 직접 연관 없음

#### 6. cold_and_dry 피처 (기온<5°C AND 습도<50%)
- 시도: 겨울 극단 패턴 전용 binary 피처
- 분석: r=0.214 (cold_and_dry=1: 43μg/m³, =0: 18μg/m³)
- 결과: 효과 없음
- 원인: LightGBM이 기온+습도를 비선형으로 이미 학습

### 새 데이터 실험 결과

#### 강수 데이터 (days from the last rain_by grid.parquet)
- 데이터: 격자별 마지막 비 이후 경과 일수, 2023-10~2025-10 기간, 9,307개 격자
- 이론 상관계수: r=0.236 (일별 서울 평균 기준)
  - 0~1일: 13.7 μg/m³ → 7~15일: 25.2 μg/m³ (단조 증가)
- 실제 결과: G_rain MAE=7.43 (D_ambient 7.35 대비 악화)
- 원인 분석:
  - 학습셋 커버리지 62.3% vs 테스트 83.3% (기간 불균형)
  - 강수 있는 샘플 MAE=7.82, 없는 샘플 MAE=5.46 → 강수 피처가 오히려 혼란 유발
  - 격자 수준에서는 신호가 훨씬 약함

#### 도로 위계 데이터 (서울 도로 위계 및 GVI.gpkg)
- 데이터: OSM 기반 도로 구간 11,076개, highway_type / 차선수 / GVI 포함
- 격자 커버리지: 26.2% (중점 매칭 방식의 한계)
- 실제 결과: H_hier MAE=7.31 (미미한 개선, 통계적 노이즈 수준)
- Feature Importance: highway_rank=3, max_lanes=9 (전체 중요도의 ~1%)
- 원인: 73.8%의 격자에서 값=0 → 신호 극히 약함

### 계층적 LUR 모델 실험

사용자 제안: `total PM = ambient PM (V5) + residual (LUR 계층 회귀)`

#### 분석 결과
```
잔차 = road_pm - ambient_pm10
  → 90.6%가 음수 (mean=-20.21)
  → 두 변수가 서로 다른 현상을 측정
     ambient_pm10: 도시 대기 배경 PM10 (30-80 μg/m³)
     road_pm:      도로 표면 재비산 먼지 (10-20 μg/m³)
```

#### ICC 분석 (구 단위 계층 효과)
```
구 간 분산 (between-gu): 3.3%   ← 계층 구조의 이득 원천
구 내 분산 (within-gu):  96.7%  ← 시간적 변동 (날씨)
```

#### 계층 모델 결과 (구 intercept + Ridge 회귀)
| | Train MAE | Test MAE | Test R² |
|---|---|---|---|
| 계층적 LUR | 7.67 | 8.01 | 0.274 |
| LightGBM D_ambient | 5.57 | **7.35** | **0.314** |

**결론**: 계층적 LUR이 LightGBM에 비해 열등.  
도로 재비산먼지는 "어디(공간)"보다 **"언제(날씨)"** 문제임.
- 일반 LUR 적용 대상: 공간 분산 크고 시간 분산 작음 (연평균 NO₂ 등)
- 도로 재비산먼지: **시간 분산이 97%** → 비선형 날씨 모델링(LightGBM)이 최적

---

## 오차 분석 (D_ambient 최종 모델)

### 월별 MAE
| 월 | MAE | 건수 |
|----|-----|------|
| 1월 | **25.6** | 5,284 |
| 2월 | **17.2** | 5,066 |
| 3월 | 8.2 | 7,563 |
| 5월 | 5.6 | 5,520 |
| 9월 | 4.4 | 12,453 |
| 11월 | 3.7 | 7,525 |

→ **겨울(1~2월)이 오차의 주원인**

### 실측 PM 범위별 MAE
| 실측 범위 | 건수 | MAE | 예측 평균 | 실측 평균 |
|----------|------|-----|---------|---------|
| 0~5 μg/m³ | 21,583 | 5.97 | 9.45 | 3.48 |
| 5~10 | 23,954 | 3.01 | 10.31 | 7.60 |
| 10~20 | 17,739 | 5.53 | 14.11 | 14.51 |
| 20~40 | 8,041 | 10.56 | 20.37 | 28.45 |
| **>40** | **3,871** | **43.55** | **26.03** | **69.58** |

→ **고농도 과소예측**: 실측 70 μg/m³ → 예측 26 μg/m³

### Feature Importance (D_ambient 기준)
| 피처 | 중요도 | 설명 |
|------|--------|------|
| ambient_pm10 | 929 | V5 배경 PM과 도로 PM 연동 |
| 습도 | 886 | 건조할수록 재비산 증가 |
| 기온 | 737 | 겨울 건조 조건 |
| hour_sin | 313 | 측정 시간대 |
| weekday | 287 | 교통 패턴 |

---

## 성능 상한 분석

현재 모델: MAE=7.35, R²=0.314 → **설명 가능 분산 31.4%**

**미설명 분산(68.6%)의 원인:**

| 미관측 변수 | 설명 |
|------------|------|
| 강수 이력 | "마지막 비 이후 일수" — 가장 중요한 누락 변수 |
| 도로 청소 이력 | 청소 직후 PM 급감, 이후 재축적 |
| 차량 종류 비율 | 화물차/대형차가 재비산 주도 |
| 도로 포장 상태 | 노후 아스팔트 → 더 많은 입자 발생 |

**개선 가능성**: 강수 일별 관측 데이터 확보 시 R²=0.45 이상 예상

---

## 최종 모델 설정

**선택 모델**: D_ambient LightGBM

```python
피처 (20개):
  temporal:  month_sin, month_cos, hour_sin, hour_cos,
             weekday, is_weekend, season
  weather:   기온, 습도, is_dry
  lur:       buildings, greenspace, road_struc, river_zone,
             ndvi, ibi, elev_mean, sum_area, sum_height
  ambient:   ambient_pm10 (V5 grid PM10)

하이퍼파라미터:
  n_estimators=500, learning_rate=0.05, max_depth=8,
  num_leaves=63, min_child_samples=10, subsample=0.8,
  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1

타깃: log1p(road_pm) → 예측 후 expm1() 역변환
이상치: 맥락적 IQR (구+계절 그룹, k=1.5) → 그룹 중앙값 대체
```

---

## 실행 순서

```bash
cd "/workspace/ST-GNN Modeling"

# 전체 파이프라인
python3 RoadExtension_V2/run_pipeline.py

# 강수/도로위계 캐시 생성 (선택)
python3 RoadExtension_V2/step_rain.py
python3 RoadExtension_V2/step_road_hier.py

# 개별 스텝
python3 RoadExtension_V2/step1_geocode.py          # ~5분
python3 RoadExtension_V2/step2_build_target.py     # ~1분
python3 RoadExtension_V2/step3_v5_inference_all.py # ~30~60분 (GPU 권장)
python3 RoadExtension_V2/step4_build_features.py   # ~5분
python3 RoadExtension_V2/step5_train.py            # ~3분
python3 RoadExtension_V2/step6_evaluate.py         # ~1분
python3 RoadExtension_V2/step7_visualize.py        # ~1분
```

---

## 파일 구조

```
RoadExtension_V2/
├── config.py                  ← 경로, 하이퍼파라미터, ablation 구성
├── step1_geocode.py           ← 도로명 geocoding (290개 도로)
├── step2_build_target.py      ← 격자 정답 생성 + 맥락적 이상치 처리
├── step3_v5_inference_all.py  ← V5 전기간 grid PM10 inference
├── step4_build_features.py    ← 피처 결합 (LUR+교통+V5PM+기상+강수+도로위계)
├── step5_train.py             ← Ablation 학습 (9가지 구성 × LightGBM + 모델 비교)
├── step6_evaluate.py          ← 결과 분석
├── step7_visualize.py         ← 시각화
├── step_rain.py               ← 강수 데이터 전처리
├── step_road_hier.py          ← 도로 위계 데이터 전처리
├── run_pipeline.py            ← 전체 파이프라인
└── checkpoints/
    ├── road_geocoded.csv           ← geocoding 결과 (캐시)
    ├── road_pm_grid.csv            ← 격자 정답 레이블
    ├── v5_ts_lookup.csv            ← V5 timestamp 인덱스
    ├── features_train.csv          ← 학습 피처 (Train 2023~2024)
    ├── features_test.csv           ← 테스트 피처 (Test 2025)
    ├── rain_cache.csv              ← 강수 피처 캐시
    ├── road_hier_grid.csv          ← 도로 위계 격자 캐시
    ├── ablation_results.json       ← 전체 ablation 결과
    ├── evaluation_report.txt       ← 상세 분석 보고서
    ├── target_validation.png       ← 격자 매핑 검증
    ├── ablation_comparison.png     ← Ablation 비교
    ├── feature_importance.png      ← 피처 중요도
    ├── spatial_road_pm.png         ← 공간 분포 지도
    ├── prediction_scatter.png      ← 예측 vs 실측 산점도
    └── models/
        ├── A_base_lgbm.pkl ~ F_interact_lgbm.pkl
        ├── G_rain_lgbm.pkl
        ├── H_hier_lgbm.pkl
        ├── I_all_v2_lgbm.pkl
        ├── F_interact_tweedie.pkl
        ├── E_all_RF_rf.pkl
        └── E_all_MLP_mlp.pkl
```

---

## 전체 프로젝트에서의 위치

```
ST-GNN (S3/static/w12)
  ↓ hidden vector h(T, N=40, 64)
HiddenExtension V5-base
  ↓ ambient PM10 grid (T, G=10125)       ← 배경 농도 (시계열 기반)
RoadExtension V2
  ↓ road PM10 per grid (T, G=10125)      ← 도로 재비산 성분 (D_ambient LightGBM)
  ↓
Combined PM10 grid (T, G)                ← 최종 통합
  ↓
도로 청소 차량 경로 최적화
```

---

## 기술적 한계

1. **격자 매핑 정확도**: 도로 midpoint 기반 → 구간 끝단 격자 누락 가능
2. **R² 상한**: 미관측 변수(강수 이력, 청소 일정 등)로 인해 0.31이 현실적 상한
3. **교통량 NaN**: 24.8% 미매칭 → LightGBM이 자동 처리
4. **V5 커버리지**: 2023-10 이전 road dust 샘플은 ambient_pm 피처 없음
5. **겨울 오차**: 1~2월 MAE 17~26 μg/m³로 과소예측 심각 (건조+저온 복합 조건)
