"""
Edge-aware GAT layer and full ST-GNN model for Seoul PM2.5 forecasting.

Architecture:
    EdgeAwareGATConv  — custom multi-head GAT where edge features enter
                         both the attention score and the message.
    STGNNModel        — per-timestep spatial GAT + per-node GRU temporal encoder
                         with a linear prediction head.

Requires torch_scatter (pip install torch-scatter) for sparse softmax.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_add


# ──────────────────────────────────────────────────────────────────────────────
# Edge-aware Graph Attention Convolution
# ──────────────────────────────────────────────────────────────────────────────

class EdgeAwareGATConv(nn.Module):
    """Multi-head graph attention layer with edge-feature-aware messages.

    For each directed edge j → i (src=j, dst=i) the layer computes:

      Attention score:
          score_ji = LeakyReLU( a_h^T [ h_i_h ‖ h_j_h ‖ e_ji_h ] )  per head h
          α_ji     = softmax over in-neighbours j of node i

      Message:
          m_ji = W_node · h_j  +  W_edge · e_ji          (projected, per head)

      Aggregation:
          h_i_new = ELU( Σ_j  α_ji · m_ji )              (concatenated across heads)

    Args:
        node_dim: Input node feature dimension.
        edge_dim: Input edge feature dimension.
        hidden_dim: Output dimension (= num_heads × head_dim).
        num_heads: Number of attention heads (hidden_dim must be divisible).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Shared linear projections (applied once, then split per head)
        self.W_node = nn.Linear(node_dim, hidden_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, hidden_dim, bias=False)

        # Per-head attention vectors: [num_heads, 3 * head_dim]
        self.att = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.att)

        self.norm = nn.LayerNorm(hidden_dim)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _sparse_softmax(
        self,
        score: torch.Tensor,   # [E, num_heads]
        dst_idx: torch.Tensor, # [E]
        num_nodes: int,
    ) -> torch.Tensor:
        """Softmax of attention scores grouped by destination node.

        Uses scatter_max for numerical stability (subtract max before exp).

        Returns:
            alpha: [E, num_heads] — normalised attention weights.
        """
        # Max per destination node for stability
        score_max, _ = scatter_max(score, dst_idx, dim=0, dim_size=num_nodes)  # [N, H]
        score_shifted = score - score_max[dst_idx]                              # [E, H]

        exp_s = torch.exp(score_shifted)                                        # [E, H]
        exp_sum = scatter_add(exp_s, dst_idx, dim=0, dim_size=num_nodes)        # [N, H]
        alpha = exp_s / (exp_sum[dst_idx] + 1e-16)                             # [E, H]
        return alpha

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,          # [N, node_dim]
        edge_index: torch.Tensor, # [2, E]   — row 0: src, row 1: dst
        edge_attr: torch.Tensor,  # [E, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute updated node embeddings.

        Args:
            x: Node feature matrix [N, node_dim].
            edge_index: Directed edge indices [2, E].
            edge_attr: Edge feature matrix [E, edge_dim].

        Returns:
            out:   Updated node embeddings [N, hidden_dim].
            alpha: Attention weights [E, num_heads].
        """
        N = x.size(0)
        E = edge_attr.size(0)
        H = self.num_heads
        D = self.head_dim
        src = edge_index[0]   # [E]
        dst = edge_index[1]   # [E]

        # ── Project and reshape to multi-head ──────────────────────────────
        h = self.W_node(x).view(N, H, D)          # [N, H, D]
        e = self.W_edge(edge_attr).view(E, H, D)  # [E, H, D]

        h_src = h[src]   # [E, H, D]
        h_dst = h[dst]   # [E, H, D]

        # ── Attention score: a^T [h_dst ‖ h_src ‖ e] ──────────────────────
        cat = torch.cat([h_dst, h_src, e], dim=-1)           # [E, H, 3D]
        # self.att: [H, 3D] → broadcast over edges
        score = (cat * self.att.unsqueeze(0)).sum(dim=-1)    # [E, H]
        score = F.leaky_relu(score, negative_slope=0.2)

        alpha = self._sparse_softmax(score, dst, N)          # [E, H]

        # ── Message = projected source node + projected edge ──────────────
        msg = h_src + e                                      # [E, H, D]

        # ── Weighted aggregation → destination nodes ──────────────────────
        msg_weighted = alpha.unsqueeze(-1) * msg             # [E, H, D]

        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        idx_expand = dst.view(-1, 1, 1).expand(-1, H, D)
        out.scatter_add_(0, idx_expand, msg_weighted)        # [N, H, D]

        # ── Activation + flatten + layer norm ────────────────────────────
        out = F.elu(out).view(N, self.hidden_dim)            # [N, H*D]
        out = self.norm(out)
        return out, alpha


# ──────────────────────────────────────────────────────────────────────────────
# Full ST-GNN Model
# ──────────────────────────────────────────────────────────────────────────────

class STGNNModel(nn.Module):
    """Spatio-Temporal GNN for multi-node PM2.5 forecasting.

    Processing pipeline per sample:
        1. For each of the T lookback time steps, apply EdgeAwareGATConv
           with the corresponding (dynamic) edge features.
        2. Stack GAT outputs → [B, T, N, gat_hidden].
        3. Reshape to [B*N, T, gat_hidden] and pass through a shared GRU.
        4. Take the last GRU hidden state as node embeddings h_i [B, N, gru_hidden].
        5. Apply a per-node linear head → predicted PM2.5 [B, N, 1].

    The model returns both the prediction and the pre-head node embeddings h_i
    so that downstream modules (e.g. LUR) can consume them directly.

    Args:
        node_dim: Node feature channels (default 6).
        edge_dim: Edge feature channels (default 5).
        gat_hidden: GAT output dimension = num_heads × head_dim (default 64).
        gru_hidden: GRU hidden size (default 64).
        num_heads: Number of GAT attention heads (default 4).
        num_nodes: Number of monitoring stations (default 40).
    """

    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 5,
        gat_hidden: int = 64,
        gru_hidden: int = 64,
        num_heads: int = 4,
        num_nodes: int = 40,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_hidden = gat_hidden
        self.gru_hidden = gru_hidden

        self.gat = EdgeAwareGATConv(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gat_hidden,
            num_heads=num_heads,
        )
        self.gru = nn.GRU(
            input_size=gat_hidden,
            hidden_size=gru_hidden,
            batch_first=True,
        )
        self.output_head = nn.Linear(gru_hidden, 1)

    def _make_batched_edge_index(
        self,
        edge_index: torch.Tensor,   # [2, E]
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """Replicate edge_index for B batch items with node-index offsets.

        Returns:
            batched_edge_index: [2, B*E] where batch b's edges have indices
            offset by b * num_nodes.
        """
        B = batch_size
        E = edge_index.size(1)
        device = edge_index.device

        # offsets: [B, 1, 1]  →  broadcast over [B, 2, E]
        offsets = (torch.arange(B, device=device) * num_nodes).view(B, 1, 1)
        # edge_index: [2, E] → [1, 2, E] → [B, 2, E]
        batched = edge_index.unsqueeze(0).expand(B, -1, -1) + offsets
        return batched.permute(1, 0, 2).reshape(2, B * E)   # [2, B*E]

    def forward(
        self,
        node_features: torch.Tensor,   # [B, T, N, node_dim]
        edge_index: torch.Tensor,       # [2, E]
        edge_features: torch.Tensor,    # [B, T, E, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: [B, T, N, node_dim]
            edge_index: [2, E] — shared across batch and time.
            edge_features: [B, T, E, edge_dim] — dynamic per time step.

        Returns:
            pred: [B, N, 1] — predicted PM2.5 at t+1.
            h_i: [B, N, gru_hidden] — node embeddings (pre-prediction head).
        """
        B, T, N, F_node = node_features.shape
        E = edge_features.size(2)

        # Build batched edge_index once (same for all time steps)
        batched_edge_index = self._make_batched_edge_index(edge_index, B, N)  # [2, B*E]

        # ── Step 1: per-timestep spatial encoding ─────────────────────────
        spatial_outs = []
        attn_weights = []   # collect [E, H] per timestep (from batch item 0 for interpretability)
        for t in range(T):
            h_t = node_features[:, t, :, :].reshape(B * N, F_node)   # [B*N, F]
            e_t = edge_features[:, t, :, :].reshape(B * E, -1)       # [B*E, edge_dim]

            gat_out, alpha = self.gat(h_t, batched_edge_index, e_t)   # [B*N, gat_h], [B*E, H]
            spatial_outs.append(gat_out.view(B, N, self.gat_hidden))  # [B, N, gat_h]
            attn_weights.append(alpha.view(B, E, self.gat.num_heads).mean(dim=0).detach())  # [E, H]

        # [B, T, N, gat_hidden]
        spatial_out = torch.stack(spatial_outs, dim=1)

        # ── Step 2: temporal encoding via per-node GRU ────────────────────
        # Reshape: [B*N, T, gat_hidden]
        gru_input = spatial_out.permute(0, 2, 1, 3).reshape(B * N, T, self.gat_hidden)

        _, h_n = self.gru(gru_input)         # h_n: [1, B*N, gru_hidden]
        h_i = h_n.squeeze(0).view(B, N, self.gru_hidden)   # [B, N, gru_hidden]

        # ── Step 3: output head ───────────────────────────────────────────
        pred = self.output_head(h_i)         # [B, N, 1]

        # attn_weights: list of T tensors [E, H] → stack to [T, E, H]
        attn_stack = torch.stack(attn_weights, dim=0)   # [T, E, H]

        return pred, h_i, attn_stack
