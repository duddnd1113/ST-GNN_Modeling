# D.4 Interpretation and Limitations

D.1~D.3의 attention 분석 결과를 종합하고, 해석의 한계를 명시한다.

---

## 종합 해석

### Attention이 포착한 것

| 관찰 | 의미 |
|------|------|
| 공간적으로 불균등한 attention | 모델이 모든 이웃을 동등하게 취급하지 않으며, 특정 경로를 선택적으로 활성화 |
| Wind alignment와의 양의 상관 | Wind-aware edge feature(풍향, 풍속)가 attention 형성에 실제로 기여 |
| Head별 역할 분담 | 일부 head는 바람 방향에 민감, 다른 head는 거리·위치 구조를 중시하는 분업 패턴 |
| High-PM 시 attention 변화 | 고농도 이벤트에서 오염 공급원 방향 엣지의 attention이 소폭 증가 |

---

## 한계

### 1. Attention ≠ 인과관계
GAT attention은 **"어느 이웃을 얼마나 참조하는가"**를 나타내지만,  
이것이 곧 "이웃 노드가 해당 노드의 PM10을 실제로 유발한다"는 인과 관계를 증명하지 않는다.  
Attention은 예측 성능을 높이기 위해 학습된 가중치일 뿐, 물리적 인과를 보장하지 않는다.

### 2. 정적 edge feature의 한계 (Static Graph)
본 최종 모델(S3+Static)에서 edge feature는 `[dist_norm, sin_bearing, cos_bearing, wind_alignment_t, effective_wind_t]`로,  
Static graph는 구조가 고정되어 있고 dynamic feature만 타임스텝마다 바뀐다.  
따라서 attention은 **동적 바람 조건보다 고정된 공간 구조(거리, 방위)에 더 강하게 영향받을 수 있다**.

### 3. 40개 관측소 밀도 편향
서울 40개 관측소는 지리적으로 균등하게 분포되어 있지 않다.  
도심(중구, 종로, 강남 등)은 관측소 밀도가 높아 연결 차수(degree)가 크고,  
자연히 수신 attention도 높게 집계될 수 있다. 이는 공간 커버리지 편향이지 실제 물리적 중요성이 아닐 수 있다.

### 4. Head averaging의 정보 손실
D.1~D.3에서는 4개 head의 attention을 평균하여 분석했다.  
개별 head가 담당하는 역할(거리 기반 / 바람 기반 등)이 있을 경우,  
평균화 과정에서 세부 패턴이 희석될 수 있다.

### 5. 테스트셋 고농도 이벤트의 희소성
테스트셋의 도시 평균 PM10 최대값이 59.5 µg/m³로, 공식 '나쁨' 기준(81 µg/m³)에 미치지 못한다.  
극단적 고농도 이벤트에서의 attention 패턴은 테스트셋만으로 충분히 분석되지 않았을 수 있다.

---

## 향후 개선 방향

- **Integrated Gradients / GNNExplainer** 등 post-hoc explainability 기법으로 인과성 보완
- 황사·고농도 에피소드가 포함된 기간으로 데이터 확장 후 attention 재분석
- Head별 특성화 분석: head마다 attention과 특정 feature(거리, 풍향, 오염물질 농도)의 상관을 별도 산출
- Soft-Dynamic 모드와의 attention 패턴 비교 (동일 조건에서 masking 전후)
