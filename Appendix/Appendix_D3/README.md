# D.3 Case Studies on High-PM Events

고농도 PM10 시기에 모델의 attention 패턴이 어떻게 달라지는지 분석한다.

---

## High-PM 이벤트 정의

| 항목 | 값 |
|------|----|
| 기준 변수 | 테스트셋 40개 관측소 **도시 평균** PM10 (µg/m³) |
| High-PM 기준 | ≥ **33 µg/m³** (테스트셋 p90, 상위 10%) |
| Normal 기준 | < 33 µg/m³ (하위 90%) |
| High-PM 샘플 수 | **268개** / 2,732개 (9.8%) |
| 테스트셋 PM10 범위 | 3.8 ~ 59.5 µg/m³ (도시 평균 기준) |

> **참고**: 국내 PM10 공식 기준(나쁨 ≥ 81 µg/m³)은 **개별 관측소** 시간 평균 기준이다.  
> 본 분석은 40개 관측소 도시 평균을 사용하므로 절댓값이 낮아, p90을 High-PM 기준으로 채택한다.

---

## 시각화 파일

| 파일 | 내용 |
|------|------|
| [fig_d3_1_pm_threshold.png](fig_d3_1_pm_threshold.png) | PM10 분포 히스토그램 + High-PM 기준선 |
| [fig_d3_2_attn_highvsnormal.png](fig_d3_2_attn_highvsnormal.png) | High-PM vs Normal 기간 관측소별 attention 비교 |
| [fig_d3_3_case_event.png](fig_d3_3_case_event.png) | 테스트셋 최고 농도 이벤트 지도 + 관측소별 PM10 |

---

## 주요 발견

### Fig D3-1: PM10 분포
- 테스트셋 도시 평균 PM10은 우편향 분포(mean=19.3, max=59.5 µg/m³).
- 고농도 이벤트는 전체의 약 10%로 희소하다.

### Fig D3-2: High-PM vs Normal Attention 비교
- High-PM 기간에는 특정 관측소의 수신 attention이 Normal 대비 증가한다.
- 주로 **고농도가 집중되는 관측소**로 들어오는 엣지의 attention이 높아지는 경향이 있다.
- 이는 모델이 고농도 상황에서 오염물질 공급원에 해당하는 업스트림 노드를 더 주목함을 시사한다.
- 단, 변화의 크기(Δ attention)는 크지 않아, attention이 PM10 농도에 **강하게 반응하기보다**  
  구조적으로 학습된 패턴을 유지하는 경향도 있음을 나타낸다.

### Fig D3-3: 최고 농도 이벤트 Case Study
- 테스트 최고 농도 시점(도시 평균 59.5 µg/m³)에서의 attention 지도와 관측소별 PM10을 병렬 표시.
- 고농도 관측소 주변 엣지의 attention이 시각적으로 강조되는 패턴을 확인.

---

## 재생성

```bash
cd "/workspace/ST-GNN Modeling"
python3 Appendix/Appendix_D3/generate_d3.py
```
