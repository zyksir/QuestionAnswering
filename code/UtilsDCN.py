from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()

#out-of-vocabulary words to zero
def get_pretrained_embedding(np_embd):
    embedding = nn.Embedding(*np_embd.shape)
    embedding.weight = nn.Parameter(torch.from_numpy(np_embd).float())
    embedding.weight.requires_grad = False
    return embedding

def init_lstm_forget_bias(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

class DCNEncoder(nn.Module):
    def __init__(self, hidden_dim, emb_layer, dropout_ratio, if_bidirectional):
        super(DCNEncoder, self).__init__()
        '''
        params:
            hidden_dim: the hidden size of LSTM networks
            embedding_dim: the size of word embedding
            if_bidirectional: whether use bi-lstm
            dropout_ratio: the dropout_rate of LSTM networks
        '''

        self.hidden_dim = hidden_dim
        self.if_bidirectional = if_bidirectional
        self.embedding = emb_layer
        self.emb_dim = self.embedding.embedding_dim

        self.encoder = nn.LSTM(self.emb_dim, hidden_dim, 1, bias=True, batch_first=True,dropout=dropout_ratio, bidirectional=False)

        init_lstm_forget_bias(self.encoder)
        self.dropout_emb = nn.Dropout(p=dropout_ratio)
        self.sentinel = nn.Parameter(torch.rand(hidden_dim,))

    def forward(self, seq, seq_len):
        mask = torch.zeros(seq.shape[0],seq.shape[1]).to(seq.device)
        for i in range(seq_len.shape[0]):
            mask[i][:seq_len[i]] = 1
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True) # descending
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0) #ascending
        seq_ = torch.index_select(seq, 0, lens_argsort)

        seq_embd = self.embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)

        h0 = torch.zeros(1, seq.shape[0], self.hidden_dim).to(seq.device)
        c0 = torch.zeros(1, seq.shape[0], self.hidden_dim).to(seq.device)
        output, _ = self.encoder(packed,(h0,c0))

        # e sorted by seq length (descending)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()

        # e recovered by id
        # size: (batch_size, seq_len, embedding_size*2)
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x l
        e = self.dropout_emb(e)
        b, _ = list(mask.size()) 

        # copy sentinel vector at the end
        # size: (batch_size, 1, embedding_size)
        sentinel_exp = self.sentinel.unsqueeze(0).expand(b, self.hidden_dim).unsqueeze(1).contiguous()  
        lens = lens.unsqueeze(1).expand(b, self.hidden_dim).unsqueeze(1)

        sentinel_zero = torch.zeros(b, 1, self.hidden_dim).to(seq.device)
        
        # final embedding
        # size: (batch_size, seq_len+1, embedding_size)
        lens = lens.long()
        e = torch.cat([e, sentinel_zero], 1)  
        e = e.scatter_(1, lens, sentinel_exp)

        return e

class DCNFusionBiLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(DCNFusionBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.fusion_bilstm = nn.LSTM(3 * hidden_dim, hidden_dim, 1, bias=True, batch_first=True,
                                     dropout=dropout_ratio, bidirectional=True)
        init_lstm_forget_bias(self.fusion_bilstm)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, seq_len):
        mask = torch.zeros(seq.shape[0],seq.shape[1]).to(seq.device)
        for i in range(seq.shape[0]):
            mask[i][:seq_len[i]] = 1
        lens = torch.sum(mask, 1)

        # sorted by seq_len (descending)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)

        h0 = torch.zeros(2, seq.shape[0], self.hidden_dim).to(seq.device)
        c0 = torch.zeros(2, seq.shape[0], self.hidden_dim).to(seq.device)
        output, _ = self.fusion_bilstm(packed, (h0, c0))

        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        
        # final embedding
        # size: (batch_size, seq_len, 2*embedding_size)
        e = torch.index_select(e, 0, lens_argsort_argsort)  
        e = self.dropout(e)
        return e

class DynamicDecoder(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio):
        super(DynamicDecoder, self).__init__()
        self.max_dec_steps = max_dec_steps
        self.decoder = nn.LSTM(4 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        init_lstm_forget_bias(self.decoder)

        self.maxout_start = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)
        self.maxout_end = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)

    def forward(self, U, d_mask, span):
        b, m, _ = list(U.size())

        curr_mask_s,  curr_mask_e = None, None
        results_mask_s, results_s = [], []
        results_mask_e, results_e = [], []
        step_losses = []

        mask_mult = (1.0 - d_mask.float()) * (-1e30)
        indices = torch.arange(0, b, out=torch.LongTensor(b))

        # ??how to initialize s_i_1, e_i_1
        s_i_1 = torch.zeros(b, ).long()
        e_i_1 = torch.sum(d_mask, 1)
        e_i_1 = e_i_1 - 1

        if use_cuda:
            s_i_1 = s_i_1.cuda()
            e_i_1 = e_i_1.cuda()
            indices = indices.cuda()

        dec_state_i = None
        s_target = None
        e_target = None
        if span is not None:
            s_target = span[:, 0]
            e_target = span[:, 1]
        u_s_i_1 = U[indices, s_i_1, :]  # b x 2l
        for _ in range(self.max_dec_steps):
            u_e_i_1 = U[indices, e_i_1, :]  # b x 2l
            u_cat = torch.cat((u_s_i_1, u_e_i_1), 1)  # b x 4l

            lstm_out, dec_state_i = self.decoder(u_cat.unsqueeze(1), dec_state_i)
            h_i, c_i = dec_state_i

            s_i_1, curr_mask_s, step_loss_s = self.maxout_start(h_i, U, curr_mask_s, s_i_1,
                                                                u_cat, mask_mult, s_target)
            u_s_i_1 = U[indices, s_i_1, :]  # b x 2l
            u_cat = torch.cat((u_s_i_1, u_e_i_1), 1)  # b x 4l

            e_i_1, curr_mask_e, step_loss_e = self.maxout_end(h_i, U, curr_mask_e, e_i_1,
                                                              u_cat, mask_mult, e_target)

            if span is not None:
                step_loss = step_loss_s + step_loss_e
                step_losses.append(step_loss)

            results_mask_s.append(curr_mask_s)
            results_s.append(s_i_1)
            results_mask_e.append(curr_mask_e)
            results_e.append(e_i_1)

        result_pos_s = torch.sum(torch.stack(results_mask_s, 1), 1).long()
        result_pos_s = result_pos_s - 1
        idx_s = torch.gather(torch.stack(results_s, 1), 1, result_pos_s.unsqueeze(1)).squeeze()

        result_pos_e = torch.sum(torch.stack(results_mask_e, 1), 1).long()
        result_pos_e = result_pos_e - 1
        idx_e = torch.gather(torch.stack(results_e, 1), 1, result_pos_e.unsqueeze(1)).squeeze()

        loss = None

        if span is not None:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / self.max_dec_steps
            loss = torch.mean(batch_avg_loss)

        return loss, idx_s, idx_e


class MaxOutHighway(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, dropout_ratio):
        super(MaxOutHighway, self).__init__()
        self.hidden_dim = hidden_dim
        self.maxout_pool_size = maxout_pool_size

        self.r = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)
        #self.dropout_r = nn.Dropout(p=dropout_ratio)

        self.m_t_1_mxp = nn.Linear(3 * hidden_dim, hidden_dim*maxout_pool_size)
        #self.dropout_m_t_1 = nn.Dropout(p=dropout_ratio)

        self.m_t_2_mxp = nn.Linear(hidden_dim, hidden_dim*maxout_pool_size)
        #self.dropout_m_t_2 = nn.Dropout(p=dropout_ratio)

        self.m_t_12_mxp = nn.Linear(2 * hidden_dim, maxout_pool_size)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, h_i, U, curr_mask, idx_i_1, u_cat, mask_mult, target=None):
        b, m, _ = list(U.size())

        r = F.tanh(self.r(torch.cat((h_i.view(-1, self.hidden_dim), u_cat), 1)))  # b x 5l => b x l
        #r = self.dropout_r(r)

        r_expanded = r.unsqueeze(1).expand(b, m, self.hidden_dim).contiguous()  # b x m x l

        m_t_1_in = torch.cat((U, r_expanded), 2).view(-1, 3*self.hidden_dim)  # b*m x 3l

        m_t_1 = self.m_t_1_mxp(m_t_1_in)  # b*m x p*l
        #m_t_1 = self.dropout_m_t_1(m_t_1)
        m_t_1, _ = m_t_1.view(-1, self.hidden_dim, self.maxout_pool_size).max(2) # b*m x l

        m_t_2 = self.m_t_2_mxp(m_t_1)  # b*m x l*p
        #m_t_2 = self.dropout_m_t_2(m_t_2)
        m_t_2, _ = m_t_2.view(-1, self.hidden_dim, self.maxout_pool_size).max(2)  # b*m x l

        alpha_in = torch.cat((m_t_1, m_t_2), 1)  # b*m x 2l
        alpha = self.m_t_12_mxp(alpha_in)  # b * m x p
        alpha, _ = alpha.max(1)  # b*m
        alpha = alpha.view(-1, m) # b x m

        alpha = alpha + mask_mult  # b x m
        alpha = F.log_softmax(alpha, 1)  # b x m
        _, idx_i = torch.max(alpha, dim=1)

        if curr_mask is None:
            curr_mask = (idx_i == idx_i)
        else:
            idx_i = idx_i*curr_mask.long()
            idx_i_1 = idx_i_1*curr_mask.long()
            curr_mask = (idx_i != idx_i_1)

        step_loss = None

        if target is not None:
            step_loss = self.loss(alpha, target)
            step_loss = step_loss * curr_mask.float()

        return idx_i, curr_mask, step_loss


