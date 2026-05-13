# D.1 Spatial Attention Visualization

모델: **S3 + Static + Window=12h** (best configuration, Test MAE=2.614 µg/m³)  
대상: EdgeAwareGATConv의 attention weight α ∈ [E, H] (엣지별, head별)  
데이터: 테스트셋 전체 2,732개 샘플 × 12 타임스텝

---

## 분석 개요

GAT은 각 directed edge j→i에 대해 다음과 같이 attention weight를 계산한다:

```
score_ji = LeakyReLU( a_h^T [ h_i || h_j || e_ji ] )
α_ji     = softmax over in-neighbours j of node i
```

이 α 값은 "노드 i가 이웃 j의 정보를 얼마나 참조하는가"를 나타낸다.  
본 섹션에서는 테스트셋 전체에 대해 **시간 · head 평균**한 α를 공간적으로 시각화한다.

---

## 시각화 파일

| 파일 | 내용 |
|------|------|
| [fig_d1_1_attn_map_osm.png](fig_d1_1_attn_map_osm.png) | 서울 지도 위 엣지별 attention (색상·굵기) |
| [fig_d1_2_node_attention.png](fig_d1_2_node_attention.png) | 관측소별 수신 attention 순위 |
| [fig_d1_3_head_comparison.png](fig_d1_3_head_comparison.png) | GAT head별 attention 분포 비교 |

---

## 주요 발견

### Fig D1-1: Attention 지도
- Attention은 공간적으로 **불균등하게 분포**한다 — 일부 엣지가 지속적으로 높은 가중치를 받는다.
- 고attention 엣지는 **도심 밀집 구역**(중구, 성동, 성북 인근)에 집중되는 경향이 있다.

### Fig D1-2: 관측소별 수신 Attention
- **수신 attention이 높은 관측소**는 많은 이웃 노드로부터 정보를 강하게 집약받는 허브 역할을 한다.
- 상위 관측소들은 대체로 서울 중심부에 위치하며, 인근 관측소 수(연결 차수)와 양의 상관을 보인다.

### Fig D1-3: Head별 분포
- 4개 head 모두 비슷한 분포 형태를 가지나, head별로 mean attention과 분산에 차이가 있다.
- 특정 head가 다른 head보다 더 집중적(low entropy) 패턴을 보이면, 해당 head가 더 선택적으로 정보를 필터링하는 역할을 함을 시사한다.

---

## 재생성

```bash
cd "/workspace/ST-GNN Modeling"
# (최초 1회) attention 캐시 생성
python3 Appendix/extract_attention.py
# D1 시각화
python3 Appendix/Appendix_D1/generate_d1.py
```
