"""
JointSpatialSTGNN — V4 핵심 모델

구조:
  ST-GNN Encoder (fine-tune)
      ↓ h[B, N, d]
      ├── Head 1: Forecast head   h → PM[t+1]  (기존 ST-GNN head)
      ├── Head 2: Direct head     h → PM[t+1]  (auxiliary, 공간 표현력 강화)
      └── Head 3: Spatial LOO     cross_attn(h_ctx) → PM_tgt

Loss:
  L = L_forecast + λ₁·L_direct + λ₂·L_spatial
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from geo_loo import build_loo_batch


# ── Spatial Attention Head ────────────────────────────────────────────────────

class SpatialCrossAttention(nn.Module):
    """
    Context station hidden → Target position hidden via multi-head cross-attention.

    attn_mode='full' : query = [Fourier(coord_tgt), X_tgt, wind_avg]
    attn_mode='coord': query = [Fourier(coord_tgt)] only
    """

    def __init__(
        self,
        h_dim:      int = 64,
        r_dim:      int = 32,
        n_heads:    int = 4,
        att_hidden: int = 32,
        dropout:    float = 0.1,
        fourier_freqs: int = 8,
        attn_mode:  str = "coord",
    ):
        super().__init__()
        self.r_dim     = r_dim
        self.n_heads   = n_heads
        self.attn_mode = attn_mode

        # Fourier position encoding: lat, lon → sin/cos @ freqs
        self.fourier_freqs = fourier_freqs
        pos_enc_dim = 4 * fourier_freqs  # sin/cos × lat/lon × freqs

        query_dim = pos_enc_dim  # coord only (확장 가능)

        # context encoder: h → key/value
        self.ctx_proj = nn.Sequential(
            nn.Linear(h_dim, r_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        # query encoder: position → query
        self.qry_proj = nn.Sequential(
            nn.Linear(query_dim, r_dim), nn.ReLU(), nn.Dropout(dropout),
        )

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=r_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(r_dim)

        # PM head
        self.pm_head = nn.Sequential(
            nn.Linear(r_dim, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, N, 2) [lat, lon] → (B, N, 4*freqs)
        """
        freqs = torch.linspace(1.0, self.fourier_freqs, self.fourier_freqs,
                               device=coords.device)
        # coords: (B, N, 2, 1) × freqs: (freqs,) → (B, N, 2, freqs)
        angles = coords.unsqueeze(-1) * freqs * 3.14159
        sin_enc = torch.sin(angles)   # (B, N, 2, freqs)
        cos_enc = torch.cos(angles)   # (B, N, 2, freqs)
        # flatten: (B, N, 4*freqs)
        return torch.cat([sin_enc, cos_enc], dim=-1).flatten(-2)

    def forward(
        self,
        h_ctx:       torch.Tensor,  # (B, N_ctx, h_dim)
        coords_ctx:  torch.Tensor,  # (B, N_ctx, 2)
        coords_tgt:  torch.Tensor,  # (B, N_tgt, 2)
    ) -> torch.Tensor:
        """
        Returns: pm_pred (B, N_tgt, 1)
        """
        # Key/Value from context
        K = V = self.ctx_proj(h_ctx)                    # (B, N_ctx, r_dim)

        # Query from target position (Fourier encoding)
        q_enc = self._fourier_encode(coords_tgt)        # (B, N_tgt, 4*freqs)
        Q = self.qry_proj(q_enc)                        # (B, N_tgt, r_dim)

        # Cross-attention
        h_tgt_hat, _ = self.cross_attn(Q, K, V)        # (B, N_tgt, r_dim)
        h_tgt_hat    = self.norm(h_tgt_hat + Q)        # residual

        return self.pm_head(h_tgt_hat)                  # (B, N_tgt, 1)


# ── Joint Model ───────────────────────────────────────────────────────────────

class JointSpatialSTGNN(nn.Module):
    """
    ST-GNN + Direct Head + Spatial LOO Head

    학습 단계:
      Phase 1: ST-GNN frozen, Head 2/3만 학습
      Phase 2: ST-GNN GRU+GAT fine-tune + Head 2/3 joint
    """

    def __init__(
        self,
        stgnn:      nn.Module,
        h_dim:      int = 64,
        r_dim:      int = 32,
        n_heads:    int = 4,
        att_hidden: int = 32,
        dropout:    float = 0.1,
        fourier_freqs: int = 8,
    ):
        super().__init__()
        self.stgnn   = stgnn
        self.h_dim   = h_dim

        # Head 2: Direct PM prediction (auxiliary)
        self.direct_head = nn.Sequential(
            nn.Linear(h_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Head 3: Spatial LOO cross-attention
        self.spatial_head = SpatialCrossAttention(
            h_dim=h_dim, r_dim=r_dim, n_heads=n_heads,
            att_hidden=att_hidden, dropout=dropout,
            fourier_freqs=fourier_freqs,
        )

    def freeze_stgnn(self):
        """Phase 1: ST-GNN 전체 동결."""
        for p in self.stgnn.parameters():
            p.requires_grad_(False)

    def unfreeze_stgnn(self, gru_lr: float, gat_lr: float):
        """
        Phase 2: GRU/GAT를 각기 다른 lr로 unfreeze.
        Returns: optimizer param groups 리스트
        """
        for p in self.stgnn.parameters():
            p.requires_grad_(True)
        return [
            {"params": self.stgnn.gat.parameters(),    "lr": gat_lr,  "name": "gat"},
            {"params": self.stgnn.gru.parameters(),    "lr": gru_lr,  "name": "gru"},
            {"params": self.stgnn.output_head.parameters(), "lr": gru_lr, "name": "stgnn_head"},
        ]

    def forward(
        self,
        node_feat:   torch.Tensor,    # (B, T, N, F)
        edge_index:  torch.Tensor,    # (2, E)
        edge_feat:   torch.Tensor,    # (B, T, E, 5)
        coords:      torch.Tensor,    # (N, 2) station 좌표
        target_idx:  Optional[List[int]] = None,  # LOO target 인덱스
        pm_target:   Optional[torch.Tensor] = None,  # (B, N, 1) 정답 (loss 계산용)
    ) -> dict:
        """
        Returns dict:
            pred_forecast : (B, N, 1)  — Head 1 예측
            pred_direct   : (B, N, 1)  — Head 2 예측
            pred_spatial  : (B, N_tgt, 1)  — Head 3 예측 (target_idx 있을 때만)
            h             : (B, N, d)  — hidden vector
            pm_tgt_spatial: (B, N_tgt)  — spatial LOO 정답
        """
        # ── ST-GNN forward ────────────────────────────────────────────────
        pred_forecast, h, _ = self.stgnn(node_feat, edge_index, edge_feat)
        # pred_forecast: (B, N, 1), h: (B, N, d)

        # ── Head 2: Direct ────────────────────────────────────────────────
        pred_direct = self.direct_head(h)   # (B, N, 1)

        out = {
            "pred_forecast": pred_forecast,
            "pred_direct":   pred_direct,
            "h":             h,
            "pred_spatial":  None,
            "pm_tgt_spatial": None,
        }

        # ── Head 3: Spatial LOO (target_idx 지정 시) ─────────────────────
        if target_idx is not None:
            B, N, d = h.shape
            pm_flat = pm_target.squeeze(-1) if pm_target is not None else None

            h_ctx, h_tgt, pm_tgt, ctx_mask = build_loo_batch(
                h, pm_flat, target_idx
            )
            # coords broadcast: (1, N, 2) → (B, N, 2)
            coords_b  = coords.unsqueeze(0).expand(B, -1, -1)
            coords_ctx = coords_b[:, ctx_mask, :]
            coords_tgt = coords_b[:, ~ctx_mask, :]

            pred_spatial = self.spatial_head(h_ctx, coords_ctx, coords_tgt)
            out["pred_spatial"]   = pred_spatial   # (B, N_tgt, 1)
            out["pm_tgt_spatial"] = pm_tgt         # (B, N_tgt)

        return out


# ── Loss 계산 ─────────────────────────────────────────────────────────────────

def compute_joint_loss(
    out:            dict,
    pm_target:      torch.Tensor,   # (B, N, 1)
    mask:           torch.Tensor,   # (B, N)   0=real, 1=imputed
    lambda_direct:  float = 0.3,
    lambda_spatial: float = 0.5,
    criterion:      nn.Module = None,
) -> Tuple[torch.Tensor, dict]:
    """
    L = L_forecast + λ₁·L_direct + λ₂·L_spatial

    imputed 위치(mask=1)는 loss weight 0.1 적용.
    """
    if criterion is None:
        criterion = nn.MSELoss(reduction="none")

    weight = (1.0 - 0.9 * mask)   # real=1.0, imputed=0.1

    def masked_loss(pred, target):
        loss = criterion(pred.squeeze(-1), target.squeeze(-1))
        return (loss * weight).mean()

    L_forecast = masked_loss(out["pred_forecast"], pm_target)
    L_direct   = masked_loss(out["pred_direct"],   pm_target)

    loss_dict = {
        "L_forecast": L_forecast.item(),
        "L_direct":   L_direct.item(),
        "L_spatial":  0.0,
    }

    total = L_forecast + lambda_direct * L_direct

    if out["pred_spatial"] is not None:
        L_spatial = F.mse_loss(
            out["pred_spatial"].squeeze(-1),
            out["pm_tgt_spatial"],
        )
        total = total + lambda_spatial * L_spatial
        loss_dict["L_spatial"] = L_spatial.item()

    loss_dict["total"] = total.item()
    return total, loss_dict
