# Appendix A: ST-GNN Graph Mode 설명

본 연구에서는 기상 관측소 간 PM10 오염물질 이동을 모델링하기 위해  
세 가지 Graph Mode를 설계·비교하였다.  
각 모드는 **동일한 40개 관측소(노드)**를 공유하며, 엣지 구성 방식과  
엣지 특성(edge feature) 처리 방법에서 차이를 보인다.

---

## 비교표

| 항목 | Static | Climatological | Soft-Dynamic |
|------|--------|----------------|--------------|
| **Graph 유형** | 무방향 (양방향) | 유방향 | 유방향 (시간별 마스킹) |
| **엣지 구성 기준** | 거리 임계값 (≤ 10 km) | 거리 임계값 + 훈련 기간 평균 풍향 | 거리 임계값 (Static과 동일) |
| **엣지 수** | 706개 (i→j, j→i 모두 포함) | 396개 (절반 이하로 pruning) | 706개 (Static 구조 공유) |
| **Graph 토폴로지 변화** | 고정 (학습 내내 동일) | 고정 (사전 계산 후 고정) | 시간 스텝마다 변화 |
| **엣지 특성 차원** | 3D static | 3D static | 5D (static 3D + dynamic 2D) |
| **Static edge features** | `[dist_norm, sin(b), cos(b)]` | `[dist_norm, sin(b), cos(b)]` | `[dist_norm, sin(b), cos(b)]` |
| **Dynamic edge features** | 없음 | 없음 | `[wind_alignment, effective_wind]` |
| **풍향 정보 활용 방식** | 미사용 | 훈련 기간 평균 풍향으로 엣지 사전 필터링 | 각 타임스텝 실시간 풍향으로 엣지 마스킹 |
| **오염 전파 방향성** | 없음 (양방향 전파) | 지배 풍향 방향만 전파 | 현재 풍향 방향만 활성화 |
| **계산 비용** | 낮음 | 낮음 (준비 후 고정) | 중간 (타임스텝별 feature 계산) |
| **물리적 해석** | 거리 기반 공간 인접성 | 계절·기후적 지배 풍향 반영 | 순간 기상 조건을 실시간 반영 |

---

## 엣지 특성 상세

### Static edge features (3D, 모든 모드 공통)

| 특성 | 수식 | 범위 | 설명 |
|------|------|------|------|
| `dist_norm` | dist / threshold_km | [0, 1] | 정규화된 두 관측소 간 거리 |
| `sin_bearing` | sin(bearing_src→dst) | [−1, 1] | 엣지 방위각의 sin 값 |
| `cos_bearing` | cos(bearing_src→dst) | [−1, 1] | 엣지 방위각의 cos 값 |

### Dynamic edge features (2D, Soft-Dynamic 전용)

| 특성 | 수식 | 범위 | 설명 |
|------|------|------|------|
| `wind_alignment` | uu·sin(b) + vv·cos(b) | (−∞, +∞) | 풍속 벡터와 엣지 방향의 내적 (순풍이면 양수) |
| `effective_wind` | max(0, wind_alignment) / 10.0 | [0, 1] | 양의 바람 기여만 추출하여 정규화 |

> **Soft-Dynamic의 마스킹 원리**: `wind_alignment ≤ 0`인 엣지는 해당 타임스텝에서  
> dynamic feature가 0으로 설정되어 메시지 패싱에 기여하지 않는다.

---

## Graph Mode별 엣지 필터링 로직

```
Static:
    connect(i→j)  if  haversine(i, j) < 10 km
    connect(j→i)  if  haversine(i, j) < 10 km       # 양방향

Climatological:
    connect(i→j)  if  haversine(i, j) < 10 km
                  AND mean_wind_alignment(i→j) > 0   # 훈련 기간 평균 풍향 기준

Soft-Dynamic:
    edges = Static graph (동일)
    at time t:
        active(i→j)   if  wind_alignment_t(i→j) > 0
        inactive(i→j) if  wind_alignment_t(i→j) ≤ 0  → dynamic features = 0
```

---

## 시각화 파일 목록

### 실제 서울 지도 기반 (OpenStreetMap, 40개 관측소)

| 파일 | 설명 |
|------|------|
| [osm_graph_modes_overview.png](osm_graph_modes_overview.png) | 세 가지 Graph Mode 3-패널 비교 (OSM 실제 지도) |
| [osm_static_graph.png](osm_static_graph.png) | Static Graph — 서울 지도 위 (706 edges, 거리별 색상) |
| [osm_climatological_graph.png](osm_climatological_graph.png) | Climatological Graph — 서울 지도 위 (396 edges, 단방향) |
| [osm_soft_dynamic_graph.png](osm_soft_dynamic_graph.png) | Soft-Dynamic Graph — 서울 지도 위 (활성 353 / 비활성 353) |

### 개념 설명용 Schematic (14개 대표 노드)

| 파일 | 설명 |
|------|------|
| [fig_graph_modes_overview.png](fig_graph_modes_overview.png) | 세 가지 Graph Mode 3-패널 개념도 |
| [fig_static_graph.png](fig_static_graph.png) | Static Graph 개념도 |
| [fig_climatological_graph.png](fig_climatological_graph.png) | Climatological Graph 개념도 |
| [fig_soft_dynamic_graph.png](fig_soft_dynamic_graph.png) | Soft-Dynamic Graph 개념도 |

### 생성 스크립트

| 파일 | 설명 |
|------|------|
| [generate_osm_graph_viz.py](generate_osm_graph_viz.py) | OSM 지도 기반 시각화 생성 (requests + PIL + matplotlib) |
| [generate_graph_mode_viz.py](generate_graph_mode_viz.py) | Schematic 다이어그램 생성 (matplotlib only) |

> **OSM 시각화**: 실제 서울 40개 PM10 관측소 좌표를 사용하며, OpenStreetMap 타일을 배경으로 활용.  
> OSM 재생성 시 인터넷 연결 필요 (`python3 generate_osm_graph_viz.py`).  
> © OpenStreetMap contributors
