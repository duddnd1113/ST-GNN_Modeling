# HiddenExtension V1

## 개요

ST-GNN hidden vector와 LUR(Land Use Regression) 공간 변수를 결합한 **Joint 학습 기반 공간 확장 모델** 첫 번째 버전.  
Station-level hidden representation → PM10 예측, 이후 grid-level 추론으로 확장.

---

## 핵심 아이디어

```
ST-GNN (frozen)
    ↓ hidden vector h ∈ R^d
Cross-Attention (station → target position)
    ↓
LUR Head: PM = θ(r) + β(X)
    ↓
PM10 prediction
```

- **Direct path**: 해당 station의 hidden vector로 직접 PM 예측 (학습용)
- **Cross path**: 나머지 station들의 hidden vector로 LOO(Leave-One-Out) PM 예측 (grid 추론용)
- Joint loss: `L = λ · L_direct + (1-λ) · L_cross`

---

## 구조 특징

| 항목 | 설정 |
|------|------|
| LUR 피처 차원 | 6 (NDVI, IBI, buildings%, greenspace%, road_struc%, river_zone%) |
| Hidden 압축 차원 (r_dim) | 16 (base) |
| Loss 가중치 (lam) | 0.5 (base) |
| LUR head | linear / mlp |
| Attention | full (위치+피처+풍향) / spatial_only (위치만) |
| Compressor | Direct/Cross 공유 단일 compressor |

---

## Ablation 실험 (11개)

| 실험 | direct MAE | vs ST-GNN |
|------|-----------|-----------|
| xsatellite, r16, lam0.5 | **2.6659** | +0.0515 |
| xall, r8, lam0.5 | 2.6963 | +0.0818 |
| xlandcover, r16, lam0.5 | 2.7132 | +0.0988 |
| xall, r32, lam0.5 | 2.7427 | +0.1283 |
| xall, r16, lam0.5 | 2.7437 | +0.1293 |
| ... | ... | ... |

**ST-GNN baseline MAE: 2.6144**  
→ 전체 11개 실험 모두 baseline 대비 악화

---

## 주요 발견

1. **모든 실험이 ST-GNN baseline보다 나쁨** → spatial extension이 오히려 성능 저하
2. **LUR 피처가 적을수록(satellite=2차원) 더 나음** → 고차원 LUR 피처는 noise
3. **r_dim이 작을수록(8) 더 나음** → 과도한 압축보다 적절한 압축이 중요
4. **lam=0.5가 cross MAE에 유리** → direct/cross 균형이 중요
5. **MLP head가 linear보다 나쁨** → 비선형성보다 단순 구조가 안정적

---

## 한계

- Direct/Cross path가 **compressor를 공유** → 두 경로가 서로 다른 표현을 학습해야 하는데 제약
- 학습 곡선: **overfitting 심함** (train loss 계속 감소, val loss flat)
- Cross path MAE(3.3~3.5) >> Direct path MAE(2.6~2.9) → grid 추론 품질 저하

---

## V2에서 개선한 점

- Direct/Cross path **독립적인 compressor** 적용
- LUR 피처 차원 확장: 6 → **9차원** (elevation, building area, building height 추가)
- base r_dim: 16 → **8** (V1 best가 r8이었으므로)
- base lam: 0.5 → **0.3** (cross 경로 강화)
- x_mode 옵션 확장: `no_building` 추가
