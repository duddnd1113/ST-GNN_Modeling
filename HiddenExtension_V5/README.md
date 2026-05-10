# HiddenExtension V5

## 개요

**Fixed Effect + Hierarchical LUR Prior** 기반 PM10 예측 모델.

LUR 변수를 공간 기저 효과(spatial baseline)의 사전 분포로 사용하고,
시간 변동분은 ST-GNN hidden vector로 설명하는 **역할 분리 구조**.

---

## 데이터 분석 기반 설계 결정

V5 설계 전 실시한 PM10 분산 분해 결과:

```
station 간 평균 차이 (between): std = 2.55 μg/m³
시간 변동 (within):             std = 24.43 μg/m³
→ between/within 비율: 0.10

단순 station 고정 효과 설명력: ~0.5%
LUR ~ station_mean 최대 상관:  r = 0.29 (약함)

결론: station-level MAE 개선은 제한적 (0.1~0.2 μg/m³ 수준)
      하지만 그리드 추론에서는 여전히 의미 있음
```

---

## 구조

```
PM_{i,t} = dynamic(h_{i,t}, temporal_t)      ← 주 예측 (frozen hidden)
          + bias_i                             ← station bias 보정
          + seasonal_bias_{i,s}               ← station × 계절 교호 보정

bias_i  = LUR_i @ γ + u_i                    ← 계층적 구조
          γ : LUR → 공간 기저 매핑 (학습)
          u_i: station 잔차 (학습, L2 정규화)

그리드 추론:
  PM_{g,t} = dynamic(h_{g,t}, temporal_t)    ← IDW hidden
            + LUR_g @ γ                       ← 공간 bias (u_g = 0)
            + season_global_s                 ← global 계절 평균 사용
```

### 핵심 차별점

| 구성 요소 | 역할 | V1~V4 대비 |
|-----------|------|-----------|
| `dynamic` | h + temporal → PM deviation | MLP, 동일 역할 |
| `bias_i`  | station 체계적 오류 보정 | 새로 추가 |
| `γ`       | LUR → 공간 기저 (그리드 추론용) | LUR의 올바른 역할 정의 |
| `u_i`     | LUR이 못 설명하는 station 효과 | L2 정규화로 과적합 방지 |
| `seasonal_bias` | 계절 × station 교호 | 계절성 더 세밀히 포착 |

---

## 왜 LUR이 V3에서 실패했는가

```
V3 (RF): X = [h(97%) + LUR(0.7%) + temporal(2.2%)] → PM
          ↑ LUR을 h와 같은 레벨에서 temporal 예측에 사용
          → LUR은 정적이라 temporal 변동 설명 불가 → 0.7% 기여

V5:       X = {
            dynamic: h → PM deviation  (temporal 변동 담당)
            LUR     → α_g              (공간 baseline 담당, 그리드 추론용)
          }
          → LUR의 역할이 명확해짐
```

---

## 데이터 분석 심화 발견

```
월별 PM10 평균:
  최저: 17.5 μg/m³ (9월)  최고: 51.9 μg/m³ (4월, 황사)
  → 월별 차이 34 μg/m³  >>  station 간 차이 2.55 μg/m³

시간대별 평균:
  야간(0-6): 31.2  /  오후(13-18): 35.8  →  4.6 μg/m³ 차이

station monthly std: 10.4 μg/m³
  → station × 월 교호 효과가 계절 교호보다 훨씬 강함
  → monthly FE가 seasonal FE를 크게 개선할 것으로 예상
```

---

## Ablation 실험

| 실험 | bias | FE 방식 | hier_LUR | params | 결과 |
|------|------|---------|---------|--------|------|
| V5-base | ✗ | - | ✗ | 기본 | 2.6036 |
| V5-bias | ✓ | - | ✗ | +40 | 2.6007 |
| V5-season | ✓ | 4계절 | ✗ | +160 | **2.5813** |
| V5-hier | ✓ | 4계절 | ✓ | +160+LUR | 2.6007 |
| **V5-monthly** | ✓ | **12개월** | ✓ | +480 | 진행 중 |
| **V5-monthly-hour** | ✓ | **12개월+4시간대** | ✓ | +640 | 진행 중 |

---

## 실험 결과 (업데이트)

```
ST-GNN baseline:   MAE = 2.6144
V5-season:         MAE = 2.5813  ↑ 0.0331 (모든 버전 중 최고)
V5-monthly:        MAE = ?       (예상: 2.55~2.57)
V5-monthly-hour:   MAE = ?       (예상: 2.54~2.56)
```

---

## Hold-out 검증 (그리드 성능 추정)

### 왜 필요한가

```
station MAE (2.5813):
  학습에 사용한 40개 station에서 측정
  → 낙관적 수치, 그리드 성능 보장 안 됨

Hold-out MAE:
  5개 geographic cluster의 10개 station을 hold-out
  → IDW hidden + LUR bias만으로 예측 (station bias 없음)
  → 이 MAE ≈ 실제 grid 추론에서 기대할 수 있는 성능
```

### Hold-out 클러스터 (서울 5개 권역)

| Cluster | Station | 위치 |
|---------|---------|------|
| north | 노원구(11), 강북구(4) | 북부 |
| south | 강남구(0), 송파구(22) | 남부 |
| west | 강서구(5), 양천구(25) | 서부 |
| east | 강동구(2), 광진구(8) | 동부 |
| central | 중구(33), 종로(31) | 도심 |

### 해석 가이드

```
holdout_MAE ≈ station_MAE:
  → 공간 일반화 우수, grid 추론 신뢰 가능

holdout_MAE >> station_MAE:
  → IDW 보간 오류가 크거나, 공간 일반화 부족
  → grid 추론 결과는 참고 수준으로만 사용

일반적 기대:
  holdout_MAE ≈ station_MAE + 1~3 μg/m³
```

---

## 그리드 추론 전체 파이프라인

```
[학습] station h_i,t → FixedEffectPMModel → PM_{i,t}
              ↓
[추론] h_{station,t} (40개)
       Wind-aware IDW
       h_{grid,t} (10,125개)
              ↓
       dynamic(h_{g,t}, temporal_t)
     + γ(LUR_g)                     ← LUR로 공간 baseline
     + season/monthly global avg    ← 전체 평균 보정
       ──────────────────────────
       PM_{g,t}  (10,125개)
```

---

## 파일 구조

```
HiddenExtension_V5/
├── README.md
├── config.py           ← 경로, 모델 설정, ablation 정의
├── dataset.py          ← V5 전용 데이터셋 (station_idx + h + LUR + temporal)
├── model.py            ← FixedEffectPMModel (dynamic + bias + seasonal + γ)
├── train.py            ← 학습 루프 + ablation 실행
├── inference_grid.py   ← 그리드 PM10 추론 (IDW + FixedEffect)
└── checkpoints/
    └── {exp_id}/
        ├── best_model.pt
        ├── metrics.json
        ├── history.json
        ├── gamma_lur.csv        ← γ 계수 (LUR → 공간 기저 해석)
        └── station_residuals.csv ← u_i (station 잔차 해석)
```

---

## 실행

```bash
cd "/workspace/ST-GNN Modeling"

# 전체 ablation (신규 monthly 포함)
python3 HiddenExtension_V5/train.py --run_all

# 단일 실험
python3 HiddenExtension_V5/train.py --exp V5-monthly

# Hold-out 검증 (그리드 성능 추정)
python3 HiddenExtension_V5/holdout_eval.py --exp V5-monthly
python3 HiddenExtension_V5/holdout_eval.py --run_all

# 그리드 추론 (best 모델 기준)
python3 HiddenExtension_V5/inference_grid.py --exp V5-monthly
```

---

## V1~V5 전체 버전 비교

| 버전 | 핵심 방법 | ST-GNN | best MAE |
|------|----------|--------|---------|
| V1 | Cross-attn + 6D LUR (공유 compressor) | frozen | 2.6659 |
| V2 | Cross-attn + 9D LUR (독립 compressor) | frozen | **2.6105** |
| V3 | Wind-IDW + RF | frozen | 3.1230 |
| V4 | Multi-loss Joint Fine-tuning | fine-tune | 4.10 (퇴화) |
| **V5** | **Fixed Effect + Hierarchical LUR** | frozen | 미정 |

---

## 해석 가능한 출력물

학습 완료 후 `gamma_lur.csv`에서 어떤 LUR 변수가 PM 기저에 영향을 주는지 확인 가능:

```
feature       gamma
NDVI          -0.xxx  (녹지 → PM 낮음)
road_struc_%  +0.xxx  (도로 → PM 높음)
elev_mean     -0.xxx  (고도 → PM 낮음)
...
```

이를 통해 공간 PM 분포의 지역적 특성을 해석할 수 있습니다.
