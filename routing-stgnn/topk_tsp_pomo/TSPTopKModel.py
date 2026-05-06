# Same architecture pattern as the POMO TSP model: encoder + decoder.
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPTopKModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            # POMO multiple starting nodes: trajectory j starts at node j.
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']
        self.embedding = nn.Linear(2, embedding_dim) # (2,128) outputs an embedding vector for each node based on its (x,y) coordinates.
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)]) # 6 layers of transformer encoder blocks, each with multi-head attention and feed-forward sublayers.

    def forward(self, data):
        out = self.embedding(data)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_norm1 = Add_And_Normalization_Module(**model_params)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.add_norm2 = Add_And_Normalization_Module(**model_params)

    def forward(self, x):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(x), head_num)
        k = reshape_by_heads(self.Wk(x), head_num)
        v = reshape_by_heads(self.Wv(x), head_num)
        out = multi_head_attention(q, k, v)
        out = self.multi_head_combine(out)
        out = self.add_norm1(x, out)
        out2 = self.feed_forward(out)
        return self.add_norm2(out, out2)


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num)

    def forward(self, encoded_last_node, ninf_mask):
        head_num = self.model_params['head_num']
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num)
        q = self.q_first + q_last

        out = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        out = self.multi_head_combine(out)

        score = torch.matmul(out, self.single_head_key)
        score_scaled = score / self.model_params['sqrt_embedding_dim']
        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        return F.softmax(score_masked, dim=2)


def reshape_by_heads(qkv, head_num):
    batch_s, n = qkv.size(0), qkv.size(1)
    return qkv.reshape(batch_s, n, head_num, -1).transpose(1, 2)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s, head_num, n, key_dim = q.size()
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=q.device))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(batch_s, n, head_num * key_dim)


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        return self.norm((input1 + input2).transpose(1, 2)).transpose(1, 2)


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))
