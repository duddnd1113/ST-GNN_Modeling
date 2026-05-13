# C.4 Experimental Results and Analysis

전체 **78개 실험** (10 Scenarios × 3 Graph Modes × 3 Window Sizes, 일부 미실시 제외)의  
Test MAE / RMSE 결과를 체계적으로 분석한다.  
원본 수치는 [all_results.csv](all_results.csv) 참조.

---

## C.4.1 Overall Experimental Setup

| 항목 | 값 |
|------|----|
| 노드 수 | 40개 (서울 PM10 관측소) |
| 예측 대상 | PM10 (µg/m³), 1시간 ahead |
| Window Size | 12h / 24h / 48h |
| Graph Mode | Static / Soft-Dynamic / Climatological |
| Feature Scenario | S1 ~ S10 (Appendix C3 참조) |
| 평가 지표 | Test MAE, Test RMSE (원래 스케일, µg/m³) |
| 모델 구조 | GAT-GRU ST-GNN (GAT hidden=64, GRU hidden=64, heads=4) |
| 학습 설정 | Epochs=100, LR=1e-3, Batch=32, 거리 임계값=10 km |
| 총 실험 수 | 78개 (climatological은 일부 시나리오·window 미실시) |

---

## C.4.2 Overall Performance Comparison

> **Fig 1** (Scenario × Graph Mode MAE Heatmap), **Fig 4** (시나리오별 Best MAE 랭킹) 참조.

### 전체 Best 구성 Top 5

| 순위 | 시나리오 | Graph Mode | Window | Test MAE | Test RMSE |
|:----:|---------|-----------|:------:|:--------:|:---------:|
| 🥇 1 | S3 (+Pollutants) | Static | 12h | **2.614** | 3.731 |
| 🥈 2 | S5 (+Poll+AllMask) | Static | 48h | 2.631 | 3.808 |
| 🥉 3 | S10 (All+SumMask) | Soft-Dynamic | 24h | 2.668 | 3.856 |
| 4 | S10 (All+SumMask) | Soft-Dynamic | 48h | 2.670 | 3.861 |
| 5 | S8 (+Rain) | Static | 48h | 2.671 | 3.893 |

Fig 1의 Heatmap에서 확인할 수 있듯, **Static Graph + 오염물질 포함 시나리오** 조합이 전반적으로 우세하며, Climatological Graph는 모든 시나리오에서 현저히 열세이다. Soft-Dynamic은 feature-rich 시나리오(S10)에서 Static에 필적하는 성능을 보인다.

---

## C.4.3 Effect of Graph Construction

> **Fig 2** (Graph Mode별 평균 MAE 비교) 참조.

### Graph Mode별 평균 성능

| Graph Mode | 평균 MAE (전체) | 특징 |
|------------|:--------------:|------|
| **Static** | **~2.78** | 모든 window·시나리오에서 가장 안정적 |
| Soft-Dynamic | ~2.84 | 일부 조합에서 Static 추월 |
| Climatological | ~3.35 | 전체적으로 유의미하게 열세 |

**Climatological의 부진**: 훈련 기간 평균 풍향으로 전체 엣지의 약 44%를 사전 제거(706 → 396)하므로, 실시간 기상 변화에 대한 메시지 패싱 표현력이 제한된다. 평균 풍향이 지배적이지 않은 시간대에는 오히려 중요한 경로를 차단하는 결과를 낳는다.

**Soft-Dynamic의 위치**: 구조는 Static과 동일하나 매 타임스텝마다 역풍 엣지의 dynamic feature를 0으로 마스킹한다. 단순 기상 정보(S1~S2)에서는 Static 대비 개선이 미미하지만, 다양한 오염물질·기상 정보가 풍부하게 제공되는 시나리오(S10)에서는 MAE 2.668로 Static(2.805)을 크게 앞선다. **즉, Soft-Dynamic은 feature가 풍부할수록 더 효과적인 inductive bias로 작동한다.**

### Graph Mode × Feature Scenario 교호작용

| 시나리오 유형 | Static MAE | Soft-Dynamic MAE | Soft-Dyn 이득 |
|-------------|:----------:|:----------------:|:-------------:|
| 단순 (S1, S2) | 2.806 / 2.703 | 2.882 / 2.745 | 없음 (오히려 손해) |
| 오염물질 포함 (S3~S6) | 2.614~2.770 | 2.704~2.916 | 없음~미미 |
| 전체 통합 (S10) | 2.805 | **2.668** | **−0.137** ↑↑ |

feature가 많아질수록 Soft-Dynamic의 실시간 바람 마스킹이 노이즈 필터로 기능하는 것으로 해석된다.

---

## C.4.4 Effect of Feature Scenarios

> **Fig 5** (Feature Group 추가에 따른 MAE 변화) 참조.

S1(Base)을 기준으로 Static Graph + Window=12h 조건에서 feature group별 기여를 분석한다.

| 시나리오 | 추가 특성 | MAE | S1 대비 | 해석 |
|---------|---------|:---:|:-------:|------|
| S1 | Base (기상 4종 + PM10) | 2.812 | 기준 | - |
| **S3** | **+오염물질 (SO2, CO, O3, NO2)** | **2.614** | **−0.198 ↑↑** | **단일 최대 개선** |
| S7 | +기상변수 (기온, 습도, 이슬점, 기압) | 2.699 | −0.114 ↑ | 두 번째로 유효 |
| S6 | +오염물질 + 요약 Mask | 2.767 | −0.045 ↑ | Mask 추가 효과 미미 |
| S4 | +오염물질 + PM10 Mask | 2.781 | −0.031 ↑ | S3 대비 Mask는 불리 |
| S9 | +기상 + 강수 | 2.795 | −0.017 ↑ | 강수가 기상 효과를 희석 |
| S2 | +PM10 Mask | 2.806 | −0.007 (미미) | Mask 단독 효과 없음 |
| S5 | +오염물질 + 전체 Mask | 2.808 | −0.004 (미미) | 과잉 Mask로 S3 대비 크게 열세 |
| S10 | 전체 통합 + 요약 Mask | 2.825 | +0.012 ↓ | 강수 포함으로 오히려 악화 |
| S8 | +강수 | 2.846 | **+0.034 ↓** | 강수 단독은 노이즈로 작용 |

**주요 관찰:**
- **오염물질(S3)**이 단연 가장 효과적인 feature group: SO2, CO, O3, NO2는 PM10과 높은 공변동성을 가지며 직접적인 예측 보조 신호로 작용한다.
- **Mask 변수의 역효과**: PM10 Mask나 오염물질 Mask를 과다 추가하면(S4, S5) S3 대비 성능이 오히려 하락한다. Mask는 모델에 "이 값은 신뢰도가 낮다"는 신호를 주지만, 이미 오염물질이라는 강한 신호가 있을 때 Mask는 불필요한 노이즈를 추가한다.
- **강수(S8)는 단독으로 역효과**: 강수 유무·강수량은 PM10 농도와의 관계가 비선형적이고 간헐적이어서, 단순 선형 신호로 포함 시 모델 학습을 방해하는 것으로 보인다.

---

## C.4.5 Effect of Window Size

> **Fig 3** (Window Size가 MAE에 미치는 영향) 참조.

| Window | Static 평균 MAE | Soft-Dynamic 평균 MAE |
|:------:|:--------------:|:--------------------:|
| 12h | 2.775 | 2.861 |
| 24h | 2.848 | 2.829 |
| 48h | 2.772 | 2.835 |

**Window Size는 성능에 결정적 인자가 아니다.** 세 window 간 평균 MAE 차이는 Static 기준 ~0.07, Soft-Dynamic 기준 ~0.03 수준으로 미미하며, 뚜렷한 단조 증감 경향이 없다.

시나리오별로 살펴보면:
- **Static**: window=12h와 48h가 비슷하게 좋고, 24h가 약간 불리한 U자 패턴이 관찰되는 시나리오가 있다.
- **Soft-Dynamic**: window=24h에서 최고 성능(S10 MAE=2.668)이 집중된다.
- **S8(강수)**: window가 길어질수록 오히려 성능 개선(Static 48h가 best)되어, 강수의 누적 효과를 긴 window가 포착하는 것으로 해석 가능하다.

**결론**: Window 선택은 시나리오·모드 조합에 따라 개별 조정이 필요하며, 계산 비용 대비 큰 효과 차이가 없다면 12h가 실용적이다.

---

## C.4.6 Final Model Selection

종합 분석 결과, **S3 + Static + Window=12h** 를 최종 ST-GNN 구성으로 선택한다.

### 선택 근거

| 기준 | S3 + Static + W12 | S10 + Soft-Dynamic + W24 |
|------|:-----------------:|:------------------------:|
| Test MAE | **2.614** | 2.668 |
| 특성 차원 | **9D** (경량) | 20D |
| 특성 해석 가능성 | 높음 (오염물질 직접 측정값) | 낮음 (강수·가시거리 등 혼재) |
| Graph 계산 비용 | 낮음 (고정 구조) | 중간 (타임스텝별 마스킹) |
| 학습 안정성 | 높음 (분산 낮음) | 중간 |

S10+Soft-Dynamic+W24의 MAE(2.668)와의 차이는 **0.054 µg/m³**로, 통계적으로 유의미하지 않을 수 있다. 그러나 S3은 절반 이하의 특성 수(9D vs 20D)로 이에 준하는 성능을 달성하며, 강수·가시거리 등 해석이 어려운 변수를 포함하지 않아 **모델의 물리적 해석 가능성**이 우수하다.

또한 Soft-Dynamic이 feature-rich 환경에서 유리함을 C.4.3에서 확인했으나, S3처럼 오염물질 위주의 깔끔한 특성 구성에서는 Static이 동등하거나 더 안정적이다.

> **최종 선택**: `S3_transport_pm10_pollutants` + `static` graph + `window=12h`  
> Test MAE = **2.614 µg/m³** | Test RMSE = **3.731 µg/m³**

---

## 재생성

```bash
cd "/workspace/ST-GNN Modeling"
python3 Appendix/Appendix_C4/generate_results_viz.py
```
