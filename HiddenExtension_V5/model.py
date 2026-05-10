"""
V5 모델 — Fixed Effect + Hierarchical LUR Prior

구조:
  PM_{i,t} = dynamic(h_{i,t}, temporal_t)
            + bias_i                        [use_bias]
            + seasonal_bias_{i,s}           [use_seasonal_bias]

  bias_i = LUR_i @ γ + u_i                 [use_hier_lur]
           (γ는 LUR → 공간 기저 매핑, u_i는 잔차)

그리드 추론:
  PM_{g,t} = dynamic(h_{g,t}, temporal_t)
            + LUR_g @ γ                     (u_g = 0, γ만 사용)
            + seasonal_bias_avg_{s}         (계절별 global 평균 사용)
"""
import torch
import torch.nn as nn
from typing import List, Optional


def build_mlp(in_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class FixedEffectPMModel(nn.Module):
    """
    V5 핵심 모델.

    Args:
        n_stations    : station 수 (40)
        h_dim         : hidden vector 차원 (64)
        lur_dim       : LUR 피처 차원 (9)
        temporal_dim  : temporal 피처 차원 (9)
        mlp_hidden    : dynamic MLP hidden layer 크기 목록
        dropout       : dropout 비율
        use_bias      : station-specific bias 사용 여부
        use_seasonal_bias : station × season 교호 bias 사용 여부
        use_hier_lur  : LUR을 bias의 계층적 prior로 사용 여부
    """

    def __init__(
        self,
        n_stations:        int   = 40,
        h_dim:             int   = 64,
        lur_dim:           int   = 9,
        temporal_dim:      int   = 9,
        mlp_hidden:        List[int] = None,
        dropout:           float = 0.1,
        use_bias:          bool  = True,
        use_seasonal_bias: bool  = True,
        use_hier_lur:      bool  = True,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [64, 32]

        self.use_bias          = use_bias
        self.use_seasonal_bias = use_seasonal_bias  # 'season' | 'monthly' | 'monthly+hour' | False
        self.use_hier_lur      = use_hier_lur
        self.n_stations        = n_stations
        self.fe_mode           = use_seasonal_bias  # alias for clarity

        # ── Dynamic head: h + temporal → PM deviation ──────────────────
        self.dynamic = build_mlp(h_dim + temporal_dim, mlp_hidden, dropout)

        # ── Station bias correction ─────────────────────────────────────
        if use_bias:
            if use_hier_lur:
                # α_i = LUR_i @ γ + u_i
                self.gamma   = nn.Linear(lur_dim, 1, bias=True)   # LUR → baseline
                self.residual = nn.Embedding(n_stations, 1)        # station residual
                nn.init.zeros_(self.residual.weight)
            else:
                # 순수 station embedding
                self.alpha = nn.Embedding(n_stations, 1)
                nn.init.zeros_(self.alpha.weight)

        # ── Temporal fixed effects (season / monthly / monthly+hour) ───
        if use_seasonal_bias == "season":
            self.seasonal = nn.Embedding(n_stations * 4, 1)
            nn.init.zeros_(self.seasonal.weight)
        elif use_seasonal_bias == "monthly":
            self.seasonal = nn.Embedding(n_stations * 12, 1)
            nn.init.zeros_(self.seasonal.weight)
        elif use_seasonal_bias == "monthly+hour":
            self.monthly  = nn.Embedding(n_stations * 12, 1)
            self.hour_fe  = nn.Embedding(n_stations * 4,  1)
            nn.init.zeros_(self.monthly.weight)
            nn.init.zeros_(self.hour_fe.weight)
        elif use_seasonal_bias:  # True (하위 호환)
            self.seasonal = nn.Embedding(n_stations * 4, 1)
            nn.init.zeros_(self.seasonal.weight)

    def get_station_bias(
        self,
        sta_idx: torch.Tensor,   # (B,) int
        lur:     torch.Tensor,   # (B, lur_dim)
    ) -> torch.Tensor:           # (B, 1)
        if not self.use_bias:
            return torch.zeros(sta_idx.size(0), 1, device=sta_idx.device)

        if self.use_hier_lur:
            alpha_lur = self.gamma(lur)                        # (B, 1)
            alpha_res = self.residual(sta_idx)                 # (B, 1)
            return alpha_lur + alpha_res
        else:
            return self.alpha(sta_idx)                         # (B, 1)

    def get_temporal_bias(
        self,
        sta_idx:    torch.Tensor,  # (B,) int
        season_idx: torch.Tensor,  # (B,) int 0~3
        month_idx:  torch.Tensor = None,  # (B,) int 0~11
        hour_bin:   torch.Tensor = None,  # (B,) int 0~3
    ) -> torch.Tensor:             # (B, 1)
        if not self.use_seasonal_bias:
            return torch.zeros(sta_idx.size(0), 1, device=sta_idx.device)

        mode = self.use_seasonal_bias
        if mode == "monthly+hour":
            m_compound = sta_idx * 12 + month_idx
            h_compound = sta_idx * 4  + hour_bin
            return self.monthly(m_compound) + self.hour_fe(h_compound)
        elif mode == "monthly":
            compound = sta_idx * 12 + month_idx
            return self.seasonal(compound)
        else:  # 'season' or True
            compound = sta_idx * 4 + season_idx
            return self.seasonal(compound)

    def forward(
        self,
        sta_idx:    torch.Tensor,           # (B,) int
        h:          torch.Tensor,           # (B, h_dim)
        lur:        torch.Tensor,           # (B, lur_dim)
        temporal:   torch.Tensor,           # (B, temporal_dim)
        season_idx: torch.Tensor,           # (B,) int
        month_idx:  torch.Tensor = None,    # (B,) int
        hour_bin:   torch.Tensor = None,    # (B,) int
    ) -> torch.Tensor:                      # (B,)
        dynamic  = self.dynamic(torch.cat([h, temporal], dim=-1))
        bias     = self.get_station_bias(sta_idx, lur)
        temporal_bias = self.get_temporal_bias(sta_idx, season_idx, month_idx, hour_bin)
        return (dynamic + bias + temporal_bias).squeeze(-1)

    def predict_grid(
        self,
        h_grid:    torch.Tensor,  # (G, h_dim)
        lur_grid:  torch.Tensor,  # (G, lur_dim)
        temporal:  torch.Tensor,  # (G, temporal_dim) or (1,)
        season_idx: int,
        month_idx:  int = 0,
        hour_bin:   int = 0,
    ) -> torch.Tensor:            # (G,)
        """
        Grid 추론: station residual(u_i) 없이 γ(LUR)만 사용.
        seasonal bias: 전체 평균 사용 (station-specific 없음).
        """
        dynamic = self.dynamic(
            torch.cat([h_grid, temporal.expand(h_grid.size(0), -1)
                       if temporal.dim() == 1 else temporal], dim=-1)
        )  # (G, 1)

        # 공간 bias: γ(LUR) only (u_i = 0)
        if self.use_bias and self.use_hier_lur:
            bias_g = self.gamma(lur_grid)                      # (G, 1)
        else:
            bias_g = torch.zeros(h_grid.size(0), 1, device=h_grid.device)

        # temporal FE: 전체 station 평균으로 근사
        G = h_grid.size(0)
        if self.use_seasonal_bias:
            all_sta = torch.arange(self.n_stations, device=h_grid.device)
            mode = self.use_seasonal_bias
            if mode == "monthly+hour":
                m_avg = self.monthly(all_sta * 12 + month_idx).mean()
                h_avg = self.hour_fe(all_sta * 4  + hour_bin).mean()
                t_bias_g = (m_avg + h_avg).expand(G, 1)
            elif mode == "monthly":
                t_avg = self.seasonal(all_sta * 12 + month_idx).mean()
                t_bias_g = t_avg.expand(G, 1)
            else:
                t_avg = self.seasonal(all_sta * 4 + season_idx).mean()
                t_bias_g = t_avg.expand(G, 1)
        else:
            t_bias_g = torch.zeros(G, 1, device=h_grid.device)

        return (dynamic + bias_g + t_bias_g).squeeze(-1)      # (G,)

    def bias_regularization_loss(self) -> torch.Tensor:
        """station residual의 L2 정규화."""
        if self.use_bias and self.use_hier_lur:
            return self.residual.weight.pow(2).mean()
        elif self.use_bias:
            return self.alpha.weight.pow(2).mean()
        return torch.tensor(0.0)
