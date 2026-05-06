import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Sublayer import MultiHeadAttention


class CluTSPSolver(nn.Module):
    """Multi-view attention CluTSP solver adapted for top-k grid routing.

    This version is self-contained PyTorch. It implements dense GATv2-style
    global/local attention instead of depending on torch_geometric.
    """
    def __init__(self, **model_params):
        super().__init__()
        self.encoder = CluTSPEncoder(**model_params)
        self.decoder = CluTSPDecoder(**model_params)

    def forward(self, pos_input, sampling_size=1, return_pi=False):
        batch_size, num_node, _ = pos_input.shape
        node_embeddings = self.encoder(pos_input)
        depot_embeddings = node_embeddings[:, 0:1, :]
        customer_embeddings = node_embeddings[:, 1:, :]

        xy = pos_input[:, :, 0:2]
        cluster = pos_input[:, :, -1:]
        decoder_input = torch.cat([xy, cluster], dim=2)
        customer_input = decoder_input[:, 1:, :]
        depot_xy = xy[:, 0:1, :]
        return self.decoder(customer_input, depot_xy, depot_embeddings, customer_embeddings, sampling_size, return_pi)


class CluTSPEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params["embedding_dim"]
        self.input_dim = model_params["input_dimension"]
        self.initial = nn.Linear(self.input_dim, self.embedding_dim)
        self.layers = nn.ModuleList([CluTSPEncoderLayer(**model_params) for _ in range(model_params["encoder_layer_num"])])

    def forward(self, pos_input):
        h = self.initial(pos_input[:, :, :16])
        cluster = pos_input[:, :, -1].long()
        same_cluster = cluster.unsqueeze(2) == cluster.unsqueeze(1)  # [B,N,N]
        for layer in self.layers:
            h = layer(h, same_cluster)
        return h


class DenseGATv2(nn.Module):
    """Dense GATv2-style attention for small top-k CluTSP graphs.

    global mode uses all node pairs and includes same-cluster edge feature.
    local mode masks attention to same-cluster neighbors only.
    """
    def __init__(self, embedding_dim, qkv_dim, head_num, use_edge_attr):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.qkv_dim = qkv_dim
        self.head_num = head_num
        self.use_edge_attr = use_edge_attr
        pair_dim = 2 * embedding_dim + (1 if use_edge_attr else 0)
        self.attn = nn.Linear(pair_dim, head_num, bias=False)
        self.value = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.out = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)

    def forward(self, h, same_cluster=None, local_only=False):
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        if self.use_edge_attr:
            edge = same_cluster.float().unsqueeze(-1)
            pair = torch.cat([hi, hj, edge], dim=-1)
        else:
            pair = torch.cat([hi, hj], dim=-1)

        # scores: [B, head, target_i, source_j]
        scores = F.leaky_relu(self.attn(pair), negative_slope=0.15).permute(0, 3, 1, 2)
        if local_only:
            local_mask = ~same_cluster
            scores = scores.masked_fill(local_mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(scores, dim=-1)

        v = self.value(h).view(B, N, self.head_num, self.qkv_dim).permute(0, 2, 1, 3)
        out = torch.matmul(alpha, v)  # [B,H,N,Q]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.head_num * self.qkv_dim)
        return self.out(out)


class CluTSPEncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        d = model_params["embedding_dim"]
        h = model_params["head_num"]
        q = model_params["qkv_dim"]
        ff = model_params["ff_hidden_dim"]

        self.gat_global = DenseGATv2(d, q, h, use_edge_attr=True)
        self.bn_global_1 = nn.BatchNorm1d(d)
        self.ff_global_1 = nn.Linear(d, ff)
        self.ff_global_2 = nn.Linear(ff, d)
        self.bn_global_2 = nn.BatchNorm1d(d)

        self.gat_local = DenseGATv2(d, q, h, use_edge_attr=False)
        self.bn_local_1 = nn.BatchNorm1d(d)
        self.ff_local_1 = nn.Linear(d, ff)
        self.ff_local_2 = nn.Linear(ff, d)
        self.bn_local_2 = nn.BatchNorm1d(d)

        self.fuse = nn.Linear(2 * d, d)

    def _bn(self, bn, x):
        B, N, D = x.shape
        return bn(x.reshape(B * N, D)).view(B, N, D)

    def forward(self, h, same_cluster):
        g = self.gat_global(h, same_cluster=same_cluster, local_only=False)
        g = self._bn(self.bn_global_1, g + h)
        g2 = self.ff_global_2(F.elu(self.ff_global_1(g)))
        g = self._bn(self.bn_global_2, g + g2)

        l = self.gat_local(h, same_cluster=same_cluster, local_only=True)
        l = self._bn(self.bn_local_1, l + h)
        l2 = self.ff_local_2(F.elu(self.ff_local_1(l)))
        l = self._bn(self.bn_local_2, l + l2)

        return self.fuse(torch.cat([g, l], dim=-1))


class CluTSPDecoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.logit_clipping = model_params["logit_clipping"]
        d = model_params["embedding_dim"]
        h = model_params["head_num"]
        q = model_params["qkv_dim"]

        self.ggm = GlobalGuidingModule(**model_params)
        self.w_cluster = nn.Linear(2 * d, d)
        self.w_context = nn.Linear(4 * d + 2, d)
        self.Wk_1 = nn.Linear(d, d)
        self.Wk_2 = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wg = nn.Linear(d, d)
        self.MHA = MultiHeadAttention(h, d, q, q)

        self.node_select = "sampling"
        self.cluster_select = "sampling"

    def set_node_select(self, mode):
        self.node_select = mode

    def set_cluster_select(self, mode):
        self.cluster_select = mode

    def _select_node(self, log_p, mask):
        if self.node_select == "greedy":
            selected = torch.argmax(log_p, dim=1)
        else:
            selected = torch.multinomial(log_p.exp(), 1).squeeze(1)
        mask[torch.arange(mask.size(0), device=mask.device), 0, selected] = True
        return selected, mask

    def _cost(self, pi, pos_input, depot_xy):
        xy = pos_input[:, :, :2]
        visited = xy.gather(1, pi.unsqueeze(-1).expand(-1, -1, 2))
        first = (visited[:, 0] - depot_xy[:, 0]).norm(p=2, dim=1)
        middle = (visited[:, 1:] - visited[:, :-1]).norm(p=2, dim=2).sum(1)
        last = (depot_xy[:, 0] - visited[:, -1]).norm(p=2, dim=1)
        return first + middle + last

    @staticmethod
    def _gather_log_likelihood(log_p, pi):
        return log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)

    def _cluster_masking(self, pos_input, cluster_idx, selected_nodes, cluster_mask, mask, visited_cluster_mask):
        b = pos_input.size(0)
        batch_idx = torch.arange(b, device=pos_input.device)
        selected_cluster = cluster_idx[batch_idx, selected_nodes]

        cluster_mask = (cluster_idx != selected_cluster.unsqueeze(1)).unsqueeze(1)
        current_cluster_nodes = (cluster_idx == selected_cluster.unsqueeze(1))
        all_visited = torch.all(mask.squeeze(1) | ~current_cluster_nodes, dim=1)
        cluster_mask[all_visited] = False
        cluster_mask[all_visited] |= mask[all_visited]
        visited_cluster_mask.scatter_(2, selected_cluster.long().view(-1, 1, 1), True)
        return cluster_mask, visited_cluster_mask

    def _cluster_complete(self, pos_input, mask, selected_nodes, cluster_idx):
        batch_idx = torch.arange(pos_input.size(0), device=pos_input.device)
        selected_cluster = cluster_idx[batch_idx, selected_nodes]
        cluster_nodes = (cluster_idx == selected_cluster.unsqueeze(1))
        unvisited_in_cluster = (~mask.squeeze(1)) & cluster_nodes
        return ~unvisited_in_cluster.any(dim=1, keepdim=True)

    def _cluster_embeddings(self, node_embeddings, input_data):
        cluster_indices = input_data[:, :, 2].long()
        unique_clusters = torch.unique(cluster_indices)
        cluster_eq = cluster_indices.unsqueeze(-1) == unique_clusters.view(1, 1, -1)
        mask = cluster_eq.unsqueeze(-1).float()
        emb_exp = node_embeddings.unsqueeze(2)

        cluster_sum = (emb_exp * mask).sum(dim=1)
        cluster_size = mask.sum(dim=1).clamp(min=1e-8)
        cluster_mean = cluster_sum / cluster_size

        neg_inf = torch.finfo(node_embeddings.dtype).min
        cluster_max, _ = emb_exp.masked_fill(~cluster_eq.unsqueeze(-1), neg_inf).max(dim=1)
        return torch.cat([cluster_mean, cluster_max], dim=-1), unique_clusters

    @staticmethod
    def _unvisited_mean(node_embeddings, mask):
        expanded = mask.squeeze(1).unsqueeze(-1)
        keep = (~expanded).float()
        denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (node_embeddings * keep).sum(dim=1, keepdim=True) / denom

    def forward(self, pos_input, depot_xy, depot_embeddings, node_embeddings, sampling_size=1, return_pi=False):
        pos_input = pos_input.repeat_interleave(sampling_size, dim=0)
        node_embeddings = node_embeddings.repeat_interleave(sampling_size, dim=0)
        depot_embeddings = depot_embeddings.repeat_interleave(sampling_size, dim=0)
        depot_xy = depot_xy.repeat_interleave(sampling_size, dim=0)

        batch_size, problem_size, _ = pos_input.shape
        device = pos_input.device
        cluster_idx = pos_input[:, :, 2].long()
        cluster_embed, unique_clusters = self._cluster_embeddings(node_embeddings, pos_input)
        cluster_embed = self.w_cluster(cluster_embed)
        cluster_embedding = torch.cat([depot_embeddings, cluster_embed], dim=1)
        num_cluster = len(unique_clusters)

        current_embedding = depot_embeddings
        mask = torch.zeros((batch_size, 1, problem_size), dtype=torch.bool, device=device)
        cluster_mask = torch.zeros((batch_size, 1, problem_size), dtype=torch.bool, device=device)
        visited_cluster_mask = torch.zeros((batch_size, 1, num_cluster + 1), dtype=torch.bool, device=device)
        is_new_cluster = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        aug_context_embedding = None
        cluster_guidance_embedding = None
        cluster_guidance = None

        paths, cluster_paths, log_ps, cluster_log_ps = [], [], [], []
        node_keys = self.Wk_1(node_embeddings)
        node_values = self.Wv(node_embeddings)
        node_keys_candidates = self.Wk_2(node_embeddings)

        for step in range(problem_size):
            aug_context_embedding, cluster_guidance_embedding, cluster_guidance, cluster_log_p = self.ggm(
                depot_embeddings, cluster_embedding, current_embedding, node_embeddings,
                aug_context_embedding, is_new_cluster, cluster_mask, visited_cluster_mask.clone(),
                mask, cluster_guidance_embedding, self.cluster_select, cluster_guidance, step
            )

            ninf_mask = mask | cluster_mask
            visited_mask = mask.squeeze(1)
            prev_nodes = paths[-1] if step > 0 else torch.zeros(batch_size, dtype=torch.long, device=device)

            unique_non_depot = torch.unique(cluster_idx)
            cluster_masks = cluster_idx.unsqueeze(-1) == unique_non_depot.view(1, 1, -1)
            visited_clusters = (visited_mask.unsqueeze(-1) & cluster_masks).any(dim=1).float().sum(dim=1)
            cluster_ratio = visited_clusters / max(float(unique_non_depot.numel()), 1.0)

            prev_cluster = cluster_idx[torch.arange(batch_size, device=device), prev_nodes]
            current_cluster_mask = cluster_idx == prev_cluster.unsqueeze(1)
            total_nodes = current_cluster_mask.sum(dim=1).float()
            visited_nodes = (visited_mask & current_cluster_mask).sum(dim=1).float()
            node_ratio = visited_nodes / (total_nodes + 1e-8)
            ratio_feat = torch.stack([cluster_ratio, node_ratio], dim=1).unsqueeze(1)

            context = torch.cat([aug_context_embedding, ratio_feat], dim=2)
            context = self.w_context(context)
            glimpse_query, _ = self.MHA(context, node_keys, node_values, ninf_mask)
            query = self.Wg(glimpse_query)
            logits = torch.matmul(query, node_keys_candidates.permute(0, 2, 1)).squeeze(1) / math.sqrt(query.size(-1))
            logits = torch.tanh(logits) * self.logit_clipping
            logits = logits.masked_fill(ninf_mask[:, 0, :], -1e9)
            log_p = torch.log_softmax(logits, dim=1)

            selected_nodes, mask = self._select_node(log_p, mask)
            cluster_mask, visited_cluster_mask = self._cluster_masking(
                pos_input, cluster_idx, selected_nodes, cluster_mask, mask, visited_cluster_mask
            )

            current_embedding = node_embeddings[torch.arange(batch_size, device=device), selected_nodes].unsqueeze(1)
            unvisited_cluster_mean = self._unvisited_mean(node_embeddings, mask | cluster_mask)
            aug_context_embedding = torch.cat([
                unvisited_cluster_mean, current_embedding, cluster_guidance_embedding, depot_embeddings
            ], dim=2)

            paths.append(selected_nodes)
            log_ps.append(log_p)
            if cluster_log_p is not None:
                cluster_paths.append(cluster_guidance)
                cluster_log_ps.append(cluster_log_p)

            is_new_cluster = self._cluster_complete(pos_input, mask, selected_nodes, cluster_idx)

        pi = torch.stack(paths, dim=1)
        log_p = torch.stack(log_ps, dim=1)
        cost = self._cost(pi, pos_input, depot_xy)
        likelihood = self._gather_log_likelihood(log_p, pi)

        if cluster_log_ps:
            clu_log_p = torch.stack(cluster_log_ps, dim=1)
            clu_pi = torch.stack(cluster_paths, dim=1)
            cluster_likelihood = self._gather_log_likelihood(clu_log_p, clu_pi)
        else:
            cluster_likelihood = None

        if return_pi:
            return cost, likelihood, cluster_likelihood, pi
        return cost, likelihood, cluster_likelihood


class GlobalGuidingModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        d = model_params["embedding_dim"]
        h = model_params["head_num"]
        q = model_params["qkv_dim"]
        self.logit_clipping = model_params["logit_clipping"]
        self.W_cluster_q = nn.Linear(3 * d, d, bias=False)
        self.W_cluster_k = nn.Linear(d, d, bias=False)
        self.W_cluster_v = nn.Linear(d, d, bias=False)
        self.W_cluster_k_single = nn.Linear(d, d, bias=False)
        self.MHA = MultiHeadAttention(h, d, q, q)

    @staticmethod
    def _unvisited_mean(node_embeddings, mask):
        expanded = mask.squeeze(1).unsqueeze(-1)
        keep = (~expanded).float()
        denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (node_embeddings * keep).sum(dim=1, keepdim=True) / denom

    @staticmethod
    def _select_cluster(log_p, mode):
        if mode == "greedy":
            return torch.argmax(log_p, dim=1)
        return torch.multinomial(log_p.exp(), 1).squeeze(1)

    def forward(self, depot_embedding, cluster_embedding, current_embedding, node_embeddings,
                aug_context_embedding, is_new_cluster, cluster_mask, visited_cluster_mask, mask,
                cluster_guidance_embedding, select_mode, cluster_guidance, step):
        batch_size, _, d = node_embeddings.shape
        device = node_embeddings.device

        if step == 0:
            init_aug = torch.zeros(batch_size, 1, 4 * d, device=device)
            init_guidance_emb = torch.zeros(batch_size, 1, d, device=device)
            init_guidance = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            init_aug = aug_context_embedding.clone()
            init_guidance_emb = cluster_guidance_embedding.clone()
            init_guidance = cluster_guidance.clone()

        need = torch.where(is_new_cluster.squeeze(1))[0]
        clu_prob = None
        if len(need) > 0:
            n_mask = mask[need]
            n_cluster_mask = cluster_mask[need]
            n_embed = node_embeddings[need]
            unvisit_mean = self._unvisited_mean(n_embed, n_mask)
            context = torch.cat([unvisit_mean, current_embedding[need], depot_embedding[need]], dim=2)

            q = self.W_cluster_q(context)
            k = self.W_cluster_k(cluster_embedding[need])
            v = self.W_cluster_v(cluster_embedding[need])
            prob_k = self.W_cluster_k_single(cluster_embedding[need])

            vmask = visited_cluster_mask[need].clone()
            all_real_clusters_visited = torch.all(vmask[:, 0, 1:], dim=1, keepdim=True)
            vmask[:, 0, 0] = ~all_real_clusters_visited.squeeze(1)  # depot only selectable after all clusters

            glimpse, _ = self.MHA(q, k, v, vmask)
            logits = torch.matmul(glimpse, prob_k.permute(0, 2, 1)).squeeze(1) / math.sqrt(glimpse.size(-1))
            logits = torch.tanh(logits) * self.logit_clipping
            logits = logits.masked_fill(vmask[:, 0, :], -1e9)
            local_log_p = torch.log_softmax(logits, dim=1)
            selected_cluster = self._select_cluster(local_log_p, select_mode)
            selected_cluster_emb = cluster_embedding[need][torch.arange(len(need), device=device), selected_cluster].unsqueeze(1)

            unvisit_cluster_mean = self._unvisited_mean(n_embed, n_mask | n_cluster_mask)
            new_aug = torch.cat([unvisit_cluster_mean, current_embedding[need], selected_cluster_emb, depot_embedding[need]], dim=2)

            init_aug[need] = new_aug
            init_guidance_emb[need] = selected_cluster_emb
            init_guidance[need] = selected_cluster

            clu_prob = torch.zeros(batch_size, vmask.size(2), device=device)
            clu_prob[need] = local_log_p

        return init_aug, init_guidance_emb, init_guidance, clu_prob
