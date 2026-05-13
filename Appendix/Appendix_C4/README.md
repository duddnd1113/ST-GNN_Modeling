# Appendix C4: ST-GNN Experimental Results and Analysis

전체 **78개 실험** (10 Scenarios × 3 Graph Modes × 3 Window Sizes, 일부 미실시 제외)의  
Test MAE / RMSE 결과를 시각화·분석한다.

---

## 실험 설정 요약

| 항목 | 값 |
|------|----|
| 노드 수 | 40개 (서울 PM10 관측소) |
| 예측 대상 | PM10 (µg/m³), 1시간 ahead |
| Window Size | 12h / 24h / 48h |
| Graph Mode | Static / Soft-Dynamic / Climatological |
| Feature Scenario | S1 ~ S10 |
| 평가 지표 | Test MAE, Test RMSE (원래 스케일, µg/m³) |
| 모델 | GAT-GRU ST-GNN (hidden=64, heads=4) |

---

## 시각화 파일 목록

| 파일 | 내용 |
|------|------|
| [fig1_heatmap_mae.png](fig1_heatmap_mae.png) | Scenario × Graph Mode 최소 MAE Heatmap |
| [fig2_graph_mode_bar.png](fig2_graph_mode_bar.png) | Graph Mode별 평균 MAE 비교 (window별) |
| [fig3_window_effect.png](fig3_window_effect.png) | Window Size가 MAE에 미치는 영향 (시나리오별) |
| [fig4_best_per_scenario.png](fig4_best_per_scenario.png) | 시나리오별 Best Configuration MAE 랭킹 |
| [fig5_scenario_group.png](fig5_scenario_group.png) | Feature Group 추가에 따른 MAE 변화 (S1 대비) |
| [all_results.csv](all_results.csv) | 전체 78개 실험 결과 원본 테이블 |

---

## 주요 발견

### 1. Graph Mode: Static이 전반적으로 최강 (Fig 1, Fig 2)

| Graph Mode | 평균 MAE (전체) | 특징 |
|------------|:--------------:|------|
| **Static** | **~2.78** | 모든 window, 시나리오에서 가장 안정적 |
| Soft-Dynamic | ~2.84 | 일부 시나리오(S10)에서 Static 추월 |
| Climatological | ~3.35 | 전체적으로 유의미하게 열세 |

- **Climatological의 부진 원인**: 훈련 기간 평균 풍향으로 절반에 가까운 엣지를 사전 제거하므로,  
  실시간 기상 변화에 대한 표현력이 제한됨.
- **Soft-Dynamic**: 구조는 Static과 동일하나 dynamic edge masking이  
  일부 시나리오에서 유용한 inductive bias로 작동 (특히 S10, window=24).

### 2. Feature Scenario: 오염물질 추가가 가장 효과적 (Fig 5)

S1(Base, MAE=2.812) 대비 Static + Window=12h 기준:

| 시나리오 | 추가 특성 | MAE | 변화 |
|---------|---------|-----|------|
| **S3** | +오염물질 (SO2, CO, O3, NO2) | **2.614** | **−0.198** ↑↑ |
| S7 | +기상변수 (기온, 습도, 기압) | 2.699 | −0.114 ↑ |
| S6 | +오염물질 + 요약 Mask | 2.767 | −0.045 ↑ |
| S4 | +오염물질 + PM10 Mask | 2.781 | −0.031 ↑ |
| S9 | +기상 + 강수 | 2.795 | −0.017 ↑ |
| S2 | +PM10 Mask | 2.806 | −0.007 (미미) |
| S5 | +오염물질 + 전체 Mask | 2.808 | −0.004 (미미) |
| S8 | +강수 | 2.846 | **+0.034 ↓** |
| S10 | 전체 통합 + 요약 Mask | 2.825 | +0.012 ↓ |

- **강수 단독(S8)은 오히려 성능 악화**: PM10 예측에서 강수 변수는  
  모델에 노이즈로 작용할 가능성이 있음.
- **과잉 Mask(S5)**는 S3 대비 크게 나빠짐: 불필요한 특성 노이즈 증가.
- **S10(전체 통합)**은 강수 포함으로 S3보다 열세이나,  
  Soft-Dynamic에서는 MAE=2.668로 전체 실험 중 2위.

### 3. 전체 Best 구성 Top 5

| 순위 | 시나리오 | Graph Mode | Window | Test MAE |
|:----:|---------|-----------|:------:|:--------:|
| 🥇 1 | S3 (+Pollutants) | Static | 12h | **2.614** |
| 🥈 2 | S5 (+Poll+AllMask) | Static | 48h | 2.631 |
| 🥉 3 | S10 (All+SumMask) | Soft-Dynamic | 24h | 2.668 |
| 4 | S8 (+Rain) | Static | 48h | 2.671 |
| 5 | S10 (All+SumMask) | Soft-Dynamic | 48h | 2.670 |

### 4. Window Size: 뚜렷한 단조 경향 없음 (Fig 3)

- Static 모드에서는 window=12h가 전반적으로 약간 유리하나 차이 미미.
- Soft-Dynamic은 window=24h에서 가장 좋은 결과가 집중.
- 일부 시나리오(S8, S9)에서는 긴 window가 오히려 역효과.
- **결론**: Window 선택은 시나리오·모드 조합에 따라 개별 튜닝 필요.

---

## 재생성 방법

```bash
cd "/workspace/ST-GNN Modeling"
python3 Appendix/Appendix_C4/generate_results_viz.py
```

`checkpoints/` 폴더의 모든 `metrics.json`을 자동 수집하여 시각화를 재생성한다.
