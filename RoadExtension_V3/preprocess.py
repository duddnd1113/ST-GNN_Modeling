"""
V3 전처리 공통 함수
- 1-99 percentile clip
- Box-Cox 변환 / 역변환
"""
import numpy as np
from scipy import stats


class RoadPMTransformer:
    """
    1-99 percentile clip + Box-Cox 변환기.
    학습 데이터 기준으로 fit하고, 학습/테스트 모두 transform.
    """

    def __init__(self, clip_low=1, clip_high=99, shift=0.01):
        self.clip_low  = clip_low
        self.clip_high = clip_high
        self.shift     = shift
        self.p_low  = None
        self.p_high = None
        self.lam    = None

    def fit(self, y: np.ndarray) -> "RoadPMTransformer":
        self.p_low  = float(np.percentile(y, self.clip_low))
        self.p_high = float(np.percentile(y, self.clip_high))
        y_clip = np.clip(y, self.p_low, self.p_high)
        _, lam = stats.boxcox(y_clip + self.shift)
        self.lam = float(lam)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y_clip = np.clip(y, self.p_low, self.p_high)
        return ((y_clip + self.shift) ** self.lam - 1.0) / self.lam

    def inverse_transform(self, y_bc: np.ndarray) -> np.ndarray:
        y_inv = (np.array(y_bc) * self.lam + 1.0) ** (1.0 / self.lam) - self.shift
        return np.clip(y_inv, 0.0, None)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def summary(self):
        print("  clip 범위: {:.2f} ~ {:.2f} μg/m³".format(self.p_low, self.p_high))
        print("  Box-Cox lambda: {:.4f}".format(self.lam))
        print("  shift: {}".format(self.shift))
