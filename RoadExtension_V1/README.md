# RoadExtension V1

## 개요

서울 도로 재비산먼지를 예측하고, **ST-GNN 기반 대기 PM10(V5-base)**과 결합해  
**격자 단위 통합 PM10 지도**를 생성하는 첫 번째 버전.

---

## 결합 원리

```
총 PM10_{g,t} = 대기 PM10_{g,t}                          [V5-base ST-GNN]
              + α × road_struc_%_{g} × road_PM_{t}        [도로 재비산 기여]

road_struc_%_{g} : 격자 g의 도로 면적 비율 (LUR 변수, 0~1)
road_PM_{t}      : RF 모델로 예측한 재비산먼지 (μg/m³)
α                : 스케일 보정 계수 (calibrate_alpha로 자동 추정)
```

**왜 road_struc_%를 가중치로 쓰는가:**
- 이미 LUR에 있는 격자별 도로 면적 비율 활용
- 도심 간선도로 격자 → road_struc_% 높음 → 재비산 기여 큼
- 공원/주거지역 → road_struc_% 낮음 → 기여 작음

---

## 데이터

| 데이터 | 출처 | 특성 |
|--------|------|------|
| 도로 재비산먼지 | `0-2. 도로 미세먼지/서울 비재산먼지_2*년도.xlsx` | 8,280건, 2023~2025, 비정기 |
| 대기 PM10 grid | `HiddenExtension_V5/checkpoints/V5-base/grid_pm10_test.npy` | (2732, 10125), 시간별 |
| road_struc_% | `격자 기본/landcover_static.npy` index=2 | (10125,), 격자별 도로 비율 |

---

## 도로 재비산먼지 데이터 특성

```
측정 방식: 측정차량이 도로를 주행하며 재비산먼지 농도 측정
건수:       8,280건 (2023~2025, 347개 날짜)
도로 수:    290개 도로명, 서울 25개 구
구별 연간:  ~110건 (비정기 현장조사)

주요 피처 (RF 입력):
  기온, 습도, is_dry     (기상 조건)
  month_sin/cos, season (계절성)
  is_weekend, weekday   (교통 패턴)
  도로길이, gu_enc       (도로 특성)
  ambient_pm10_daily    (당일 대기 PM10 — 메커니즘 연결)
```

---

## 파이프라인

```
[Step 1] 전처리
  xlsx 로드 → 정제 → 피처 엔지니어링 → 대기 PM10 연결
  preprocess.py

[Step 2] RF 모델 학습
  2023~2024 train / 2025 test
  train_road.py

[Step 3] 결합 → 격자 통합 PM10
  V5-base ambient PM + α × road_frac × road_PM
  combine_grid.py

[Step 4] 시각화
  3-panel: 대기 / 도로기여 / 통합
  visualize.ipynb
```

---

## 파일 구조

```
RoadExtension_V1/
├── README.md
├── config.py           ← 경로, 결합 파라미터
├── preprocess.py       ← 전처리 + 피처 엔지니어링
├── train_road.py       ← RF 모델 학습
├── combine_grid.py     ← 대기 PM + 도로 기여 결합
├── visualize.ipynb     ← 3-panel 비교 시각화
└── checkpoints/
    ├── road_rf_model.pkl          ← 학습된 RF 모델
    ├── road_model_meta.json       ← 성능 및 feature importance
    ├── total_pm10_test.npy        ← (T, G) 통합 PM10
    ├── road_contribution_test.npy ← (T, G) 도로 기여분
    ├── combine_result.json        ← 결합 통계
    └── combined_grid_mean.csv     ← 격자별 기간 평균
```

---

## 실행 순서

```bash
cd "/workspace/ST-GNN Modeling"

# 1. 전처리 확인
python3 RoadExtension_V1/preprocess.py

# 2. RF 모델 학습
python3 RoadExtension_V1/train_road.py

# 3. 결합 grid 생성
python3 RoadExtension_V1/combine_grid.py

# 4. 시각화 (Jupyter)
# visualize.ipynb 실행 — 날짜/시간 설정 후 셀 실행
```

---

## 전체 프로젝트에서의 위치

```
ST-GNN (S3/static/w12)
  ↓ hidden vector h(T, N, 64)
HiddenExtension V5-base
  ↓ ambient PM10 grid (T, G=10125)     ← 배경 농도
RoadExtension V1
  ↓ road resusp contribution (T, G)   ← 도로 국소 기여
  ↓
Combined PM10 grid (T, G)              ← 최종 통합 지도
  ↓
도로 청소 차량 경로 최적화 (routing-stgnn)
```
