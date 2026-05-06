"""
JointHiddenExtensionModel

Ablation 축:
  lur_mode  : 'linear'       → PM = β@X + θ@r
              'mlp'          → PM = MLP([X, r])
  attn_mode : 'full'         → score = MLP([dist, sin, cos, X_tgt, X_src])
              'spatial_only' → score = MLP([dist, sin, cos])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from utils import spatial_features_batch


class SpatialAttentionScore(nn.Module):
    """
    attn_mode='full'        : 입력 = [dist, sin, cos, X_target, X_src]  (3 + 2*x_dim)
    attn_mode='spatial_only': 입력 = [dist, sin, cos]                   (3)
    """
    def __init__(self, x_dim: int, hidden: int = 32, dropout: float = 0.1,
                 attn_mode: str = 'full'):
        super().__init__()
        self.attn_mode = attn_mode
        in_dim = 3 if attn_mode == 'spatial_only' else 3 + 2 * x_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sp_feat, X_target, X_sources):
        # sp_feat: (B, N_src, 3)
        if self.attn_mode == 'spatial_only':
            inp = sp_feat
        else:
            B, N_src, _ = X_sources.shape
            x_tgt_rep = X_target.unsqueeze(1).expand(-1, N_src, -1)
            inp = torch.cat([sp_feat, x_tgt_rep, X_sources], dim=-1)
        return self.net(inp).squeeze(-1)   # (B, N_src)


class HiddenCompressor(nn.Module):
    """h_dim → r_dim. 두 경로가 공유."""
    def __init__(self, h_dim: int = 64, r_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h_dim // 2, r_dim),
        )

    def forward(self, h):
        return self.net(h)


class JointHiddenExtensionModel(nn.Module):
    """
    Args:
        h_dim, x_dim, r_dim, att_hidden, dropout : 기본 구조
        lur_mode  : 'linear' | 'mlp'
        attn_mode : 'full'   | 'spatial_only'
    """
    def __init__(
        self,
        h_dim:      int   = 64,
        x_dim:      int   = 6,
        r_dim:      int   = 16,
        att_hidden: int   = 32,
        dropout:    float = 0.1,
        lur_mode:   str   = 'linear',
        attn_mode:  str   = 'full',
    ):
        super().__init__()
        self.x_dim    = x_dim
        self.lur_mode = lur_mode

        self.attention  = SpatialAttentionScore(x_dim, att_hidden, dropout, attn_mode)
        self.compressor = HiddenCompressor(h_dim, r_dim, dropout)

        # ── LUR head (두 경로 공유) ──────────────────────────────────────────
        if lur_mode == 'linear':
            if x_dim > 0:
                self.beta = nn.Linear(x_dim, 1, bias=False)
            self.theta = nn.Linear(r_dim, 1, bias=True)
        else:  # mlp
            in_dim = (x_dim + r_dim) if x_dim > 0 else r_dim
            self.lur_mlp = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

    def _lur_head(self, X: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """(B,) PM 예측."""
        if self.lur_mode == 'linear':
            base = self.theta(r)
            if self.x_dim > 0:
                base = base + self.beta(X)
            return base.squeeze(-1)
        else:  # mlp
            inp = torch.cat([X, r], dim=-1) if self.x_dim > 0 else r
            return self.lur_mlp(inp).squeeze(-1)

    def _cross_attention(self, h_sources, coords_target, coords_sources, X_target, X_sources):
        sp_feat = spatial_features_batch(coords_target, coords_sources)  # (B, N_src, 3)
        scores  = self.attention(sp_feat, X_target, X_sources)           # (B, N_src)
        alpha   = F.softmax(scores, dim=-1)                              # (B, N_src)
        return (alpha.unsqueeze(-1) * h_sources).sum(dim=1)             # (B, H)

    def forward(
        self,
        h_sources:      torch.Tensor,
        coords_target:  torch.Tensor,
        coords_sources: torch.Tensor,
        X_target:       torch.Tensor,
        X_sources:      torch.Tensor,
        h_target: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Cross-attention path (C)
        h_cross  = self._cross_attention(h_sources, coords_target,
                                         coords_sources, X_target, X_sources)
        r_cross  = self.compressor(h_cross)
        pm_cross = self._lur_head(X_target, r_cross)

        if h_target is None:
            return pm_cross   # 격자 추론

        # Direct path (B)
        r_direct  = self.compressor(h_target)
        pm_direct = self._lur_head(X_target, r_direct)
        return pm_direct, pm_cross
