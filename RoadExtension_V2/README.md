# RoadExtension V2

## 개요

RoadExtension **V1의 문제점**을 반영하여 격자 단위 도로 재비산먼지 예측 모델을 재설계.

---

## V1 vs V2 비교

| 항목 | V1 | V2 |
|------|----|----|
| 교통량 | 서울 전체 평균 (단일 스칼라) | **격자별 개별 교통량** (parquet, 10,125 格자) |
| LUR 변수 | 미포함 | **전체 포함** (NDVI, IBI, buildings, greenspace, road_struc, river_zone, 건물 통계) |
| Ambient PM10 | S3 CSV 구 단위 평균 | **V5 grid 단위 예측값** (250m 격자, 시간별) |
| 예측 대상 | 서울 평균 road_PM 스칼라 → 수동 결합 | **격자 단위 road_PM** (격자가 정답) |
| 도로→格자 매핑 | 없음 (구 수준 평균) | **Nominatim geocoding** → 반경 1000m 격자 할당 |
| 이상치 처리 | Winsorize 95th | Winsorize 99th + "매우나쁨" 상태 제외 |
| 모델 | Random Forest 단일 | **LightGBM** + RF + MLP (Ablation 비교) |
| 실험 설계 | 단일 실험 | **5가지 피처 Ablation** + 3가지 모델 비교 |

---

## 데이터 파이프라인

```
[원시 데이터]
  도로 재비산먼지 xlsx (8,280건, 2023~2025)
  格자별 교통량 parquet (10,125 格자 × 시간별)
  V5 ambient PM10 grid (T × 10,125)
  LUR 格자 CSV (NDVI, IBI, buildings 등)

    ↓ Step 1: Geocoding
  도로명 + 지역명 → Nominatim → lat/lon → 格자 CELL_ID

    ↓ Step 2: 格자 정답 생성
  (date, hour, 도로명) → 반경 1000m 格자에 도로 PM10 할당
  (同 格자 중복 → 평균, Winsorize 적용)

    ↓ Step 3: V5 全기간 inference
  h_train/val/test + Wind-aware IDW → grid_pm10_{split}.npy

    ↓ Step 4: 피처 결합
  LUR static + 格자 교통량 + V5 ambient PM + temporal + weather
  → features_train.csv / features_test.csv

    ↓ Step 5: Ablation 학습
  5가지 피처 구성 × LightGBM
  + RF / MLP 모델 비교

    ↓ Step 6: 평가 분석
  MAE, R², 구별 성능, 분위수별 오차

    ↓ Step 7: 시각화
  Ablation 비교 / Feature Importance / 공간 분포 지도
```

---

## 피처 구성 (Ablation)

| 그룹 | 피처 | 비고 |
|------|------|------|
| **temporal** | month_sin/cos, hour_sin/cos, weekday, is_weekend, season | 7개 |
| **weather** | 기온, 습도, is_dry | 3개 |
| **lur** | buildings, greenspace, road_struc, river_zone, ndvi, ibi, elev_mean, sum_area, sum_height | 9개 정적+동적 |
| **traffic** | traffic (格자별 교통량 합계) | 1개 |
| **ambient_pm** | ambient_pm10 (V5 grid PM10) | 1개 |

### Ablation 구성

| 실험 | 포함 피처 |
|------|----------|
| A_base | temporal + weather |
| B_lur | + LUR |
| C_traffic | + LUR + 格자 교통량 |
| D_ambient | + LUR + V5 ambient PM |
| **E_all** | 모든 피처 (best 기대) |

---

## 이상치 처리 설계

도로 재비산먼지 분포: mean=20, std=34, max=818 μg/m³ (심한 우편향)

```
1. 상태 "매우나쁨" 제외 (49건): 공사·특수 상황 → 모델 설명 불가
2. Winsorize 99th percentile: 극단 이상치 cap
3. log1p 변환: 학습 타깃 정규화
```

---

## 格자 매핑 방법

```
1. 고유 (지역명, 도로명) 290개 → "서울특별시 {구} {도로명}" Nominatim 쿼리
2. 반환된 lat/lon 기준 반경 1000m 内 格자 검색
3. road_struc% ≥ 5% 格자만 포함 (도로 있는 格자만)
4. 같은 格자에 같은 날 여러 측정값 → 평균
```

**한계**: 도로 전체 구간이 아닌 midpoint 기반 할당이라 구간 끝단 格자 누락 가능

---

## 학습/테스트 분할

- **Train**: 2023~2024 (traffic parquet 시작 이후, 즉 2023-10-01 이후)
- **Test**: 2025

---

## 실행 순서

```bash
cd "/workspace/ST-GNN Modeling"

# 전체 파이프라인 (권장)
python3 RoadExtension_V2/run_pipeline.py

# V5 inference 없이 빠르게 (ambient_pm 피처 미포함)
python3 RoadExtension_V2/run_pipeline.py --skip_v5

# 특정 스텝부터 재실행
python3 RoadExtension_V2/run_pipeline.py --from_step 5

# 개별 스텝 실행
python3 RoadExtension_V2/step1_geocode.py        # ~5분 (Nominatim 속도 제한)
python3 RoadExtension_V2/step2_build_target.py   # ~1분
python3 RoadExtension_V2/step3_v5_inference_all.py  # ~30~60분 (GPU 권장)
python3 RoadExtension_V2/step4_build_features.py # ~5분 (parquet 읽기)
python3 RoadExtension_V2/step5_train.py          # ~3분
python3 RoadExtension_V2/step6_evaluate.py       # ~1분
python3 RoadExtension_V2/step7_visualize.py      # ~1분
```

---

## 파일 구조

```
RoadExtension_V2/
├── config.py               ← 경로, 하이퍼파라미터, ablation 구성
├── step1_geocode.py        ← 도로명 geocoding (290개 도로)
├── step2_build_target.py   ← 格자 단위 정답 생성 + 검증
├── step3_v5_inference_all.py ← V5 전기간 grid PM10 inference
├── step4_build_features.py ← LUR + 교통 + V5 PM + temporal 결합
├── step5_train.py          ← Ablation study 학습
├── step6_evaluate.py       ← 결과 분석 + 보고서
├── step7_visualize.py      ← 시각화
├── run_pipeline.py         ← 전체 파이프라인 실행
├── README.md
└── checkpoints/
    ├── road_geocoded.csv           ← 도로 geocoding 결과 (캐시)
    ├── road_pm_grid.csv            ← 格자 단위 정답 레이블
    ├── v5_ts_lookup.csv            ← V5 timestamp 인덱스 lookup
    ├── features_train.csv          ← 학습 피처 행렬
    ├── features_test.csv           ← 테스트 피처 행렬
    ├── ablation_results.json       ← 전체 ablation 결과
    ├── evaluation_report.txt       ← 분석 보고서
    ├── target_validation.png       ← 格자 매핑 검증 그래프
    ├── ablation_comparison.png     ← Ablation 비교 그래프
    ├── feature_importance.png      ← 피처 중요도
    ├── spatial_road_pm.png         ← 공간 분포 지도
    ├── prediction_scatter.png      ← 예측 vs 실측 산점도
    └── models/
        ├── A_base_lgbm.pkl
        ├── ...
        ├── E_all_lgbm.pkl          ← 최종 모델 (기대 best)
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
  ↓ road PM10 per grid (T, G=10125)      ← 도로 재비산 성분
  ↓
Combined PM10 grid (T, G)                ← 최종 통합
  ↓
도로 청소 차량 경로 최적화
```

---

## 기술적 한계 및 향후 개선

1. **격자 매핑 정확도**: 도로 midpoint 기반 할당 → 정밀 도로 GIS 데이터로 개선 가능
2. **R² 한계**: V1에서도 확인된 바와 같이, road dust의 분산 대부분이 모델로 설명되지 않는 요인 (비공개 공사, 청소 이력 등)
3. **교통량 fallback**: parquet에 해당 格자-시간 데이터 없으면 NaN → tree 모델은 자동 처리
4. **V5 시간 범위**: 2023-10-01 이전 road dust는 ambient_pm 피처 없음 (ablation A/B/C에서 확인 가능)
