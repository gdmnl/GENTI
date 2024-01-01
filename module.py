import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from position import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
PRECISION = 5
POS_DIM_ALTER = 100


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        # special linear layer for motif explainability
        self.non_linear = non_linear
        if not non_linear:
            assert(dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            #x = self.layer_norm(x)
            h = self.act(self.fc1(x))
            z = self.fc2(h)
        else: # for explainability
            # x1, x2 shape: [B, M, F]
            x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z, z_walk


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q: [B*N_src*n_head, 1, d_k]; k: [B*N_src*n_head, num_walks, d_k]
        # v: [B*N_src*n_head, num_walks, d_v], mask: [B*N_src*n_head, 1, num_walks]
        attn = torch.bmm(q, k.transpose(-1, -2))  # [B*N_src*n_head, 1, num_walks]
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]

        output = torch.bmm(attn, v)  # [B*N_src*n_head, 1, d_v]

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        B, N_src, _ = q.size() # [B, N_src, model_dim]
        B, N_ngh, _ = k.size() # [B, N_ngh, model_dim]
        B, N_ngh, _ = v.size() # [B, N_ngh, model_dim]
        assert(N_ngh % N_src == 0)
        num_walks = int(N_ngh / N_src)
        residual = q

        q = self.w_qs(q).view(B, N_src, 1, n_head, d_k)  # [B, N_src, 1, n_head, d_k]
        k = self.w_ks(k).view(B, N_src, num_walks, n_head, d_k)  # [B, N_src, num_walks, n_head, d_k]
        v = self.w_vs(v).view(B, N_src, num_walks, n_head, d_v)  # [B, N_src, num_walks, n_head, d_k]

        q = q.transpose(2, 3).contiguous().view(B*N_src*n_head, 1, d_k)  # [B*N_src*n_head, 1, d_k]
        k = k.transpose(2, 3).contiguous().view(B*N_src*n_head, num_walks, d_k)  # [B*N_src*n_head, num_walks, d_k]
        v = v.transpose(2, 3).contiguous().view(B*N_src*n_head, num_walks, d_v)  # [B*N_src*n_head, num_walks, d_v]
        mask = mask.view(B*N_src, 1, num_walks).repeat(n_head, 1, 1) # [B*N_src*n_head, 1, num_walks]
        output, attn_map = self.attention(q, k, v, mask=mask) # output: [B*N_src*n_head, 1, d_v], attn_map: [B*N_src*n_head, 1, num_walks]

        output = output.view(B, N_src, n_head*d_v)  # [B, N_src, n_head*d_v]
        output = self.dropout(self.fc(output))  # [B, N_src, model_dim]
        output = self.layer_norm(output + residual)  # [B, N_src, model_dim]
        attn_map = attn_map.view(B, N_src, n_head, num_walks)
        return output, attn_map


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk

        # Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_steps=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, pos_dim, model_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, src_p, seq, seq_t, seq_e, seq_p, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, N_src, D]
          src_t: float Tensor of shape [B, N_src, Dt], Dt == D
          seq: float Tensor of shape [B, N_ngh, D]
          seq_t: float Tensor of shape [B, N_ngh, Dt]
          seq_e: float Tensor of shape [B, N_ngh, De], De == D
          mask: boolean Tensor of shape [B, N_ngh], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        batch, N_src, _ = src.shape
        N_ngh = seq.shape[1]
        device = src.device
        src_e = torch.zeros((batch, N_src, self.edge_dim)).float().to(device)
        src_p_pad, seq_p_pad = src_p, seq_p
        if src_p is None:
            src_p_pad = torch.zeros((batch, N_src, self.pos_dim)).float().to(device)
            seq_p_pad = torch.zeros((batch, N_ngh, self.pos_dim)).float().to(device)
        q = torch.cat([src, src_e, src_t, src_p_pad], dim=2) # [B, N_src, D + De + Dt] -> [B, N_src, D]
        k = torch.cat([seq, seq_e, seq_t, seq_p_pad], dim=2) # [B, N_ngh, D + De + Dt] -> [B, N_ngh, D]
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, N_src, D + De + Dt], attn: [B, N_src, n_head, num_walks]
        output = self.merger(output, src)
        return output, attn


class GENTI(torch.nn.Module):
    def __init__(self, n_feat, e_feat, agg='walk',
                 attn_mode='prod', use_time='time', attn_agg_method='attn',
                 pos_dim=0, pos_enc='spd', walk_pool='attn', walk_n_head=8, walk_mutual=False,
                 num_steps=3, n_head=4, drop_out=0.1, num_walks=20,
                 verbosity=1, get_checkpoint_path=None, walk_linear_out=False, device='cpu'):
        super(GENTI, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity
        N = n_feat.shape[0]

        # subgraph extraction hyper-parameters
        self.num_walks, self.num_steps = num_walks, num_steps
        self.ngh_finder = None

        # features
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)

        # dimensions of 4 elements: node, edge, time, position
        self.out_dim = 100
        self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
        self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
        self.time_dim = self.feat_dim  # default to be time feature dimension
        self.pos_dim = pos_dim  # position feature dimension
        self.pos_enc = pos_enc
        self.model_dim = self.feat_dim + self.e_feat_dim + self.time_dim + self.pos_dim
        self.logger.info('neighbors: {}, node dim: {}, edge dim: {}, pos dim: {}, time dim: {}'.format(self.num_walks, self.feat_dim, self.e_feat_dim, self.pos_dim, self.time_dim))

        # aggregation method
        self.agg = agg

        # walk-based attention/summation model hyperparameters
        self.walk_pool = walk_pool
        self.walk_n_head = walk_n_head
        self.walk_mutual = walk_mutual
        self.walk_linear_out = walk_linear_out

        # dropout for both tree and walk based model
        self.dropout_p = drop_out

        # embedding layers and encoders
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        # self.source_edge_embed = nn.parameter(torch.tensor()self.e_feat_dim)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.time_encoder = self.init_time_encoder(use_time, seq_len=self.num_walks)
        self.position_encoder = nn.Sequential(nn.Linear(in_features=num_steps + 1, out_features=pos_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=pos_dim, out_features=pos_dim))  # landing prob at [0, 1, ... num_steps]
        self.lps = torch.zeros((N + 5, num_steps + 1)).float().to(device)

        # attention model
        self.random_walk_attn_model = self.init_random_walk_attn_model()

        # final projection layer
        self.affinity_score = MergeLayer(self.out_dim, self.out_dim, self.out_dim, 1, non_linear=not self.walk_linear_out)

        self.get_checkpoint_path = get_checkpoint_path

        self.flag_for_cur_edge = True  # flagging whether the current edge under computation is real edges, for data analysis
        self.common_node_percentages = {'pos': [], 'neg': []}
        self.walk_encodings_scores = {'encodings': [], 'scores': []}

    def init_attn_model_list(self, attn_agg_method, attn_mode, n_head, drop_out):
        if attn_agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, self.e_feat_dim, self.time_dim,
                                                             self.pos_dim, self.model_dim,
                                                             attn_mode=attn_mode, n_head=n_head, drop_out=drop_out)
                                                   for _ in range(self.num_steps)])
        elif attn_agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_steps)])
        elif attn_agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_steps)])
        else:
            raise NotImplementedError('invalid agg_method value, use attn or lstm')
        return attn_model_list

    def init_random_walk_attn_model(self):
        random_walk_attn_model = RandomWalkAttention(feat_dim=self.model_dim, pos_dim=self.pos_dim,
                                                     model_dim=self.model_dim, out_dim=self.out_dim,
                                                     walk_pool=self.walk_pool,
                                                     n_head=self.walk_n_head, mutual=self.walk_mutual,
                                                     dropout_p=self.dropout_p, logger=self.logger, walk_linear_out=self.walk_linear_out)
        return random_walk_attn_model

    def init_time_encoder(self, use_time, seq_len):
        if use_time == 'time':
            self.logger.info('Using time encoding')
            time_encoder = TimeEncode(expand_dim=self.time_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            time_encoder = PosEncode(expand_dim=self.time_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            time_encoder = EmptyEncode(expand_dim=self.time_dim)
        else:
            raise ValueError('invalid time option!')
        return time_encoder

    def contrast(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l=None, test=False):
        '''
        1. grab subgraph for src, tgt, bgd
        2. add positional encoding for src & tgt nodes
        3. forward propagate to get src embeddings and tgt embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and bgd embeddings (and finally neg_score (shape: [batch, ]))
        '''
        device = self.n_feat_th.device
        cut_time_l = torch.from_numpy(cut_time_l).float().to(device)
        src_idx_l = torch.from_numpy(src_idx_l).long().to(device)
        tgt_idx_l = torch.from_numpy(tgt_idx_l).long().to(device)
        bgd_idx_l = torch.from_numpy(bgd_idx_l).long().to(device)
        if e_idx_l is not None:
            e_idx_l = torch.from_numpy(e_idx_l).long().to(device)

        self.ngh_finder.event.wait()
        subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_bgd = self.grab_subgraph(bgd_idx_l, cut_time_l, e_idx_l=None)

        # update graph
        ts_max = max(cut_time_l)
        self.ngh_finder.update_async(ts_max)

        self.flag_for_cur_edge = True
        pos_score = self.forward(src_idx_l, tgt_idx_l, cut_time_l, (subgraph_src, subgraph_tgt), test=test)
        self.flag_for_cur_edge = False
        neg_score1 = self.forward(src_idx_l, bgd_idx_l, cut_time_l, (subgraph_src, subgraph_bgd), test=test)

        return pos_score.sigmoid(), neg_score1.sigmoid()

    def forward(self, src_idx_l, tgt_idx_l, cut_time_l, subgraphs=None, test=False):
        subgraph_src, subgraph_tgt = subgraphs
        position_features = self.retrieve_position_feature(subgraph_src[0], subgraph_tgt[0])
        src_embed = self.forward_msg(src_idx_l, cut_time_l, subgraph_src, position_features[0], test=test)
        tgt_embed = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt, position_features[1], test=test)
        if self.walk_mutual:
            src_embed, tgt_embed = self.tune_msg(src_embed, tgt_embed)
        score, score_walk = self.affinity_score(src_embed, tgt_embed) # score_walk shape: [B, M]
        score.squeeze_(dim=-1)
        return score

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        subgraph = self.ngh_finder.gather_l_hop_walks(self.num_steps, src_idx_l, cut_time_l ,self.num_walks, e_idx=e_idx_l)
        return subgraph

    def forward_msg(self, src_idx_l, cut_time_l, subgraph_src, position_features, test=False):
        node_records, eidx_records, t_records = subgraph_src

        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx_l, node_records)  # length self.num_steps+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_steps+1
        edge_features = self.retrieve_edge_features(eidx_records)  # length self.num_steps
        position_features = self.position_encoder(position_features)

        final_node_embeddings = self.forward_msg_walk(hidden_embeddings, time_features, edge_features, position_features, masks)

        return final_node_embeddings

    def tune_msg(self, src_embed, tgt_embed):
        return self.random_walk_attn_model.mutual_query(src_embed, tgt_embed)

    def init_hidden_embeddings(self, src_idx_l, node_records):
        hidden_embeddings = self.node_raw_embed(node_records)  # shape [batch, n_walk, len_walk+1, node_dim]
        masks = (node_records != 0).sum(dim=-1).long()  # shape [batch, n_walk], here the masks means differently: it records the valid length of each walk

        return hidden_embeddings, masks

    def retrieve_time_features(self, cut_time_l, t_records):
        batch = len(cut_time_l)

        t_records = t_records.select(dim=-1, index=0).unsqueeze(dim=2) - t_records
        n_walk, len_walk = t_records.size(1), t_records.size(2)
        time_features = self.time_encoder(t_records.view(batch, -1)).view(batch, n_walk, len_walk,
                                                                                self.time_encoder.time_dim)
        return time_features

    def retrieve_edge_features(self, eidx_records):
        # Notice that if subgraph is tree, then len(eidx_records) is just the number of hops, excluding the src node
        # but if subgraph is walk, then eidx_records contains the random walks of length len_walk+1, including the src node
        eidx_records[:, :, 0] = 0   # NOTE: this will NOT be mixed with padded 0's since those paddings are denoted by masks and will be ignored later in lstm
        edge_features = self.edge_raw_embed(eidx_records)  # shape [batch, n_walk, len_walk+1, edge_dim]

        return edge_features    
    
    def retrieve_position_feature_2(self, nodes_src, nodes_tgt):
        B, _, L = nodes_src.shape
        feature_src, feature_tgt = [], []
        nodes = torch.concat([nodes_src, nodes_tgt], dim=1)
        for b in range(B):
            for i in range(L):
                cur_nodes = nodes[b, :, i]
                unique_values, indices = torch.unique(cur_nodes, return_inverse=True)
                counts = torch.bincount(indices) / self.num_walks
                self.lps[unique_values, i] += counts

            feature_src.append(self.lps[nodes_src[b], :])
            feature_tgt.append(self.lps[nodes_tgt[b], :])
            self.lps[nodes[b], :] = 0

        feature_src = torch.stack(feature_src, dim=0)
        feature_tgt = torch.stack(feature_tgt, dim=0)
        return feature_src, feature_tgt

    def retrieve_position_feature(self, nodes_src, nodes_tgt):
        L = self.num_steps + 1
        nodes = torch.concat([nodes_src, nodes_tgt], dim=0).reshape(-1, L)
        for i in range(L):
            cur_nodes = nodes[:, i]
            unique_values, indices = torch.unique(cur_nodes, return_inverse=True)
            counts = torch.bincount(indices) / self.num_walks
            self.lps[unique_values, i] += counts

        feature_src = self.lps[nodes_src, :]
        feature_tgt = self.lps[nodes_tgt, :]
        self.lps[nodes, :] = 0
        return feature_src, feature_tgt

    def forward_msg_layer(self, hidden_embeddings, time_features, edge_features, position_features, masks, attn_m):
        assert(len(hidden_embeddings) == len(time_features)) 
        assert(len(hidden_embeddings) == (len(edge_features) + 1)) 
        assert(len(masks) == len(edge_features))
        assert(len(hidden_embeddings) == len(position_features))
        new_src_embeddings = []
        for i in range(len(edge_features)):
            src_embedding = hidden_embeddings[i]
            src_time_feature = time_features[i]
            src_pos_feature = position_features[i]
            ngh_embedding = hidden_embeddings[i+1]
            ngh_time_feature = time_features[i+1]
            ngh_edge_feature = edge_features[i]
            ngh_pos_feature = position_features[i+1]
            ngh_mask = masks[i]
            # NOTE: n_neighbor_support = n_source_support * num_neighbor this layer
            # new_src_embedding shape: [batch, n_source_support, feat_dim]
            # attn_map shape: [batch, n_source_support, n_head, num_walks]
            new_src_embedding, attn_map = attn_m(src_embedding,  # shape [batch, n_source_support, feat_dim]
                                                 src_time_feature,  # shape [batch, n_source_support, time_feat_dim]
                                                 src_pos_feature, # shape [batch, n_source_support, pos_dim]
                                                 ngh_embedding,  # shape [batch, n_neighbor_support, feat_dim]
                                                 ngh_time_feature,  # shape [batch, n_neighbor_support, time_feat_dim]
                                                 ngh_edge_feature,  # shape [batch, n_neighbor_support, edge_feat_dim]
                                                 ngh_pos_feature, # shape [batch, n_neighbor_support, pos_dim]
                                                 ngh_mask)  # shape [batch, n_neighbor_support]

            new_src_embeddings.append(new_src_embedding)
        return new_src_embeddings

    def forward_msg_walk(self, hidden_embeddings, time_features, edge_features, position_features, masks):
        return self.random_walk_attn_model.forward_one_node(hidden_embeddings, time_features, edge_features,
                                                            position_features, masks)

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder


class RandomWalkAttention(nn.Module):
    '''
    RandomWalkAttention have two modules: lstm + tranformer-self-attention
    '''
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, walk_pool='attn', mutual=False, n_head=8, dropout_p=0.1, walk_linear_out=False):
        '''
        masked flags whether or not use only valid temporal walks instead of full walks including null nodes
        '''
        super(RandomWalkAttention, self).__init__()
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim//2  # half the model dim to save computation cost for attention
        self.out_dim = out_dim
        self.walk_pool = walk_pool
        self.mutual = mutual
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.logger = logger

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.dropout_p)  # encode all types of features along each temporal walk
        self.position_encoder = FeatureEncoder(self.pos_dim, self.pos_dim, self.dropout_p)  # encode specifially spatio-temporal features along each temporal walk
        self.projector = nn.Sequential(nn.Linear(self.feature_encoder.model_dim+self.position_encoder.model_dim, self.attn_dim),  # notice that self.feature_encoder.model_dim may not be exactly self.model_dim is its not even number because of the usage of bi-lstm
                                       nn.ReLU(), nn.Dropout(self.dropout_p))  # TODO: whether to add #[, nn.Dropout())]?
        self.self_attention = TransformerEncoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                      dim_feedforward=4*self.attn_dim, dropout=self.dropout_p,
                                                      activation='relu')
        if self.mutual:
            self.mutual_attention_src2tgt = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
            self.mutual_attention_tgt2src = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
        self.pooler = SetPooler(n_features=self.attn_dim, out_features=self.out_dim, dropout_p=self.dropout_p, walk_linear_out=walk_linear_out)
        self.logger.info('bi-lstm actual encoding dim: {} + {}, attention dim: {}, attention heads: {}'.format(self.feature_encoder.model_dim, self.position_encoder.model_dim, self.attn_dim, self.n_head))

    def forward_one_node(self, hidden_embeddings, time_features, edge_features, position_features, masks=None):
        '''
        Input shape [batch, n_walk, len_walk, *_dim]
        Return shape [batch, n_walk, feat_dim]
        '''
        combined_features = self.aggregate(hidden_embeddings, time_features, edge_features, position_features)
        combined_features = self.feature_encoder(combined_features, masks)
        if self.pos_dim > 0:
            position_features = self.position_encoder(position_features, masks)
            combined_features = torch.cat([combined_features, position_features], dim=-1)
        X = self.projector(combined_features)
        if self.walk_pool == 'sum':
            X = self.pooler(X, agg='mean')  # we are actually doing mean pooling since sum has numerical issues
            return X
        else:
            X = self.self_attention(X)
            if not self.mutual:
                X = self.pooler(X, agg='mean')  # we are actually doing mean pooling since sum has numerical issues
            return X

    def mutual_query(self, src_embed, tgt_embed):
        '''
        Input shape: [batch, n_walk, feat_dim]
        '''
        src_emb = self.mutual_attention_src2tgt(src_embed, tgt_embed)
        tgt_emb = self.mutual_attention_tgt2src(tgt_embed, src_embed)
        src_emb = self.pooler(src_emb)
        tgt_emb = self.pooler(tgt_emb)
        return src_emb, tgt_emb

    def aggregate(self, hidden_embeddings, time_features, edge_features, position_features):
        batch, n_walk, len_walk, _ = hidden_embeddings.shape
        device = hidden_embeddings.device
        if position_features is None:
            assert(self.pos_dim == 0)
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features], dim=-1)
        else:
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features, position_features], dim=-1)
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim)
        return combined_features


class FeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_p=0.1):
        super(FeatureEncoder, self).__init__()
        self.hidden_features_one_direction = hidden_features//2
        self.model_dim = self.hidden_features_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be use less
            return
        self.lstm_encoder = nn.LSTM(input_size=in_features, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch*n_walk, len_walk, feat_dim)
        if mask is not None:
            lengths = mask.view(batch*n_walk).cpu()
            X = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        encoded_features = self.lstm_encoder(X)[0]
        if mask is not None:
            encoded_features, lengths = pad_packed_sequence(encoded_features, batch_first=True)
        encoded_features = encoded_features.select(dim=1, index=-1).view(batch, n_walk, self.model_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features


class SetPooler(nn.Module):
    """
    Implement similar ideas to the Deep Set
    """
    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, agg='sum'):
        if self.walk_linear_out:  # for explainability, postpone summation to merger function
            return self.out_proj(X)
        if agg == 'sum':
            return self.out_proj(X.sum(dim=-2))
        else:
            assert(agg == 'mean')
            return self.out_proj(X.mean(dim=-2))


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_t = src.transpose(0, 1)
        src2 = self.self_attn(src_t, src_t, src_t, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt_t = tgt.transpose(0, 1)
        tgt2 = self.self_attn(tgt_t, tgt_t, tgt_t, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

