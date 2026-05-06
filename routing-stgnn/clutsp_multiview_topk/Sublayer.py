import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, Q.size(1), 1, 1)
            attn_score = attn_score.masked_fill(mask, -1e9)
        attn_dist = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_dist, V)
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.multihead_combine = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size, q_len, k_len = Q.size(0), Q.size(1), K.size(1)
        Q = Q.view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, k_len, self.n_heads, self.d_v).transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        output = self.multihead_combine(output)
        return output, attn_dist
