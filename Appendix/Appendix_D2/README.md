# D.2 Relationship Between Wind Direction and Attention

GAT attention weight가 **바람 방향**과 실제로 상관되는지 분석한다.  
Wind-aware graph 설계(엣지 특성에 wind alignment 포함)가 모델 내부에서  
물리적으로 의미 있는 attention 패턴을 유도하는지 검증한다.

---

## 분석 방법

각 엣지 e(src→dst)에 대해:
- **Wind Alignment**: `uu_src × sin(bearing) + vv_src × cos(bearing)`  
  → 양수면 바람이 src→dst 방향(순풍), 음수면 역풍
- **Attention Weight**: 테스트셋 전체·타임스텝·head 평균 α

두 변수의 상관관계를 엣지 수준(706개)에서 분석한다.

---

## 시각화 파일

| 파일 | 내용 |
|------|------|
| [fig_d2_1_wind_attn_scatter.png](fig_d2_1_wind_attn_scatter.png) | Wind alignment vs attention 산점도 + 풍향 사분위별 박스플롯 |
| [fig_d2_2_polar_attn.png](fig_d2_2_polar_attn.png) | 엣지 방위각별 평균 attention 극좌표 플롯 |
| [fig_d2_3_corr_by_head.png](fig_d2_3_corr_by_head.png) | Head별 wind alignment vs attention 상관 |
| [fig_d2_4_wind_dir_attn_map.png](fig_d2_4_wind_dir_attn_map.png) | 풍향별(동/남/서/북) attention 지도 (4개 서브플롯) |
| [fig_d2_5_attn_rank_stability.png](fig_d2_5_attn_rank_stability.png) | Top-15 엣지 attention 풍향별 안정성 히트맵 |

---

## 주요 발견

### Fig D2-1: Wind Alignment vs Attention 산점도
- Wind alignment와 attention 사이에 **약한 양의 상관**이 관찰된다.
- 박스플롯에서 "강한 순풍(>1 m/s)" 그룹의 attention 중앙값이 "역풍" 그룹보다 높다.
- 단, 상관계수(Pearson r)는 크지 않다 — attention이 바람 방향**만**으로 결정되지 않음을 의미한다.  
  (거리, 노드 특성, 공간 패턴 등 복합 요인이 함께 작용)

### Fig D2-2: 방위각별 Attention 극좌표 플롯
- 특정 방위각 구간에서 평균 attention이 높게 나타난다.
- 서울의 지배 풍향(북서~서풍)과 일치하는 방향의 엣지에서  
  attention이 상대적으로 높은 패턴을 확인할 수 있다.

### Fig D2-3: Head별 상관
- Head마다 wind alignment와의 상관 강도가 다르다.
- 일부 head는 바람 방향에 민감하게 반응하고, 다른 head는 공간 구조(거리·위치)를  
  더 중시하는 역할 분담 패턴을 시사한다.

### Fig D2-4: 풍향별 Attention 지도 (신규)

#### 분석 설계
- 테스트셋 2,732개 샘플 각각에 대해, 12스텝 윈도우 내 전체 관측소의  
  평균 uu(동서 방향 풍속)/vv(남북 방향 풍속)로 우세 풍향 분류  
  → D2의 wind alignment 수식과 동일한 성분 변수 사용, 일관성 확보
- 분류 기준: |uu| ≥ |vv| 이면 동/서풍, 아니면 남/북풍
- 결과: **동풍 1010개, 서풍 745개, 남풍 577개, 북풍 400개** (4방향 모두 n ≥ 5 충족)

#### 주요 결과

| 지표 | 값 |
|------|----|
| 동풍 ↔ 남풍 패턴 상관 (Pearson r) | **0.9973** |
| 동풍 ↔ 서풍 패턴 상관 | **0.9971** |
| 남풍 ↔ 북풍 패턴 상관 | **0.9977** |
| 서풍 ↔ 북풍 패턴 상관 | **0.9992** |

- 풍향에 관계없이 **상위 attention 엣지 순위가 완전히 동일**:  
  1위 공항대로→강서구, 2위 도봉구→노원구, 3위 강북구→노원구
- 4방향 모두에서 attention 패턴의 Pearson r > 0.997 — 사실상 동일한 패턴

#### 해석
이 결과는 **모델의 attention이 풍향에 따라 동적으로 재조정되지 않음**을 명확히 보여준다.  
모델은 wind-aware edge feature(wind_alignment, effective_wind)를 입력으로 받음에도,  
attention 형성은 **고정된 공간 구조(거리, 방위, 노드 연결 차수)**에 의해 지배되고 있다.  
이는 D2-1~D2-3의 낮은 Pearson r 결과와 일치하며, D.4의 한계 2항을 직접 시각화로 뒷받침한다.

---

## 해석 시 주의

Wind alignment와 attention의 상관은 **모델이 바람을 학습했다는 간접 증거**이지,  
"attention = wind alignment"를 의미하지 않는다.  
GAT의 attention은 노드 특성, 엣지 특성, 학습된 파라미터의 복합적 함수다.

---

## 재생성

```bash
cd "/workspace/ST-GNN Modeling"
python3 Appendix/Appendix_D2/generate_d2.py
```
