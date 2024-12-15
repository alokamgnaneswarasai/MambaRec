import numpy as np
import torch
from mamba_ssm import Mamba
from jamba import Jamba

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


import torch
import numpy as np
from torch import nn

class Jamba4Rec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(Jamba4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Embedding layers
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Jamba block
        self.jamba = Jamba(
            num_tokens=self.item_num + 1,  # Number of tokens
            heads=args.num_heads,       # Number of attention heads
            dim=args.hidden_units,  # Embedding dimension
            depth=2,                   # Number of Jamba blocks
            d_state=32,                # SSM state expansion factor
            d_conv=4,                  # Local convolution width
            expand=2,                  # Block expansion factor
            pre_emb_norm=True          # Normalize embeddings before processing
        ).to(self.dev)

        # Feed-Forward Network (Optional)
        self.feedforward_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-6)
        self.feedforward_dropout = nn.Dropout(p=args.dropout_rate)
        self.feedforward = nn.Sequential(
            nn.Linear(args.hidden_units, 4 * args.hidden_units),
            nn.ReLU(),
            nn.Linear(4 * args.hidden_units, args.hidden_units)
        )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Process sequences through the Jamba block
        seqs = self.jamba(seqs)

        # Process through Feed-Forward Network (Optional)
        feedforward_output = self.feedforward(seqs)
        seqs = self.feedforward_dropout(feedforward_output) + seqs  # Residual connection
        seqs = self.feedforward_layernorm(seqs)  # Layer normalization

        return seqs

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


class GatingSASmambaRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GatingSASmambaRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Embedding layers
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Mamba block
        self.mamba1 = Mamba(
            d_model=args.hidden_units,  # Embedding dimension
            d_state=32,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to(self.dev)

        # Self-Attention layer
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim=args.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate)
        self.attention_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-6)
        self.attention_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Gating Network
        self.gating_network = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, 2),
            torch.nn.Softmax(dim=-1)  # Output probabilities for Mamba and Attention
        )

        # Feed-Forward Network (Optional for added flexibility)
        self.feedforward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-6)
        self.feedforward_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, 4 * args.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * args.hidden_units, args.hidden_units)
        )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Compute gating probabilities
        gating_probs = self.gating_network(seqs.mean(dim=1))  # Mean pooling along sequence length
        gate_decisions = torch.argmax(gating_probs, dim=-1)  # Discrete choice: 0 for Mamba, 1 for Attention

        # Initialize output container
        output = torch.zeros_like(seqs).to(self.dev)

        # Process through the chosen block
        for i, gate in enumerate(gate_decisions):
            seq = seqs[i:i+1]  # Select individual sequence for processing

            if gate == 0:  # Mamba block
                output[i:i+1] = self.mamba1(seq)
            else:  # Self-Attention block
                seq = seq.transpose(0, 1)  # Convert to (seq_len, batch_size, hidden_units) for attention
                attn_output, _ = self.attention_layer(seq, seq, seq)
                attn_output = self.attention_dropout(attn_output) + seq  # Residual connection
                attn_output = self.attention_layernorm(attn_output)  # Layer normalization
                output[i:i+1] = attn_output.transpose(0, 1)  # Convert back to (batch_size, seq_len, hidden_units)

        # Process through Feed-Forward Network (Optional)
        feedforward_output = self.feedforward(output)
        seqs = self.feedforward_dropout(feedforward_output) + output  # Residual connection
        seqs = self.feedforward_layernorm(seqs)  # Layer normalization

        return seqs

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

class SASmambaRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASmambaRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Embedding layers
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Mamba block
        self.mamba1 = Mamba(
            d_model=args.hidden_units,  # Embedding dimension
            d_state=32,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to(self.dev)

        # Self-Attention layer
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim=args.hidden_units, num_heads=args.num_heads, dropout=args.dropout_rate)
        self.attention_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-6)
        self.attention_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Learnable weights for Mixture of Experts
        self.mamba_weight = torch.nn.Parameter(torch.tensor(0.5))
        # self.attention_weight = torch.nn.Parameter(torch.tensor(0.5))

        # Feed-Forward Network (Optional for added flexibility)
        self.feedforward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-6)
        self.feedforward_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, 4 * args.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * args.hidden_units, args.hidden_units)
        )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Process through Mamba block
        mamba_output = self.mamba1(seqs)

        # Process through Self-Attention
        seqs = seqs.transpose(0, 1)  # Convert to (seq_len, batch_size, hidden_units) for attention
        attn_output, _ = self.attention_layer(seqs, seqs, seqs)
        attn_output = self.attention_dropout(attn_output) + seqs  # Residual connection
        attn_output = self.attention_layernorm(attn_output)  # Layer normalization
        attn_output = attn_output.transpose(0, 1)  # Convert back to (batch_size, seq_len, hidden_units)

        # Combine Mamba and Attention outputs using learnable weights
        combined_output = self.mamba_weight * mamba_output + (1 - self.mamba_weight) * attn_output

        # Process through Feed-Forward Network (Optional)
        feedforward_output = self.feedforward(combined_output)
        seqs = self.feedforward_dropout(feedforward_output) + combined_output  # Residual connection
        seqs = self.feedforward_layernorm(seqs)  # Layer normalization

        return seqs

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

class MambaRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MambaRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        # print('number of users:', user_num)
        # print('number of items:', item_num)
        # print('hidden units:', args.hidden_units)
        # print('maxlen:', args.maxlen)
        
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.mamba1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model= args.hidden_units,  # Embedding dimension #64
            d_state=32,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to(self.dev)
        
        

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        # print('seqs',seqs,seqs.shape)

        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        log_feats = self.mamba1(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        
        
        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
        # print('log_feats',log_feats.shape)
        # print('pos_embs',pos_embs.shape)
        # print('neg_embs',neg_embs.shape)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)






# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    

import copy
def clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



import math
def linrec(query, key, value, mask=None, dropout=None):
    elu = torch.nn.ELU()
    query = elu(query)
    key = elu(key)


    N_K = query.size(-2)
    key_norms = torch.norm(key, dim=2, keepdim=True) * math.sqrt(N_K)
    tmpk = key / key_norms
    key = tmpk

    d_k = query.size(-1)
    query_norms = torch.norm(query, dim=3, keepdim=True) * math.sqrt(d_k)
    tmpquery = query / query_norms
    query = tmpquery    
    scores = torch.matmul(key.transpose(-2, -1),value)
    logits = torch.matmul(query,scores)
    return logits

class MultiHeadedLinrec(torch.nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedLinrec, self).__init__()
        assert embedding_dim%head== 0
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(torch.nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value,  mask=None):

        if mask is not None:
            mask = mask.unsqueeze(0)
        # print('multmaskshape===', mask.shape) #multmaskshape=== torch.Size([1, 8, 4, 4])
        batch_size = query.size(0)
        # view中的四个参数的意义
        # batch_size: 批次的样本数量
        # -1这个位置应该是： 每个句子的长度
        # self.head*self.d_k应该是embedding的维度， 这里把词嵌入的维度分到了每个头中， 即每个头中分到了词的部分维度的特征
        # query, key, value形状torch.Size([2, 8, 4, 64])
        query, key, value = [model(x).view(batch_size, -1,  self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]
        # query, key, value = [model(x) for model, x in zip(self.linears, (query, key, value))]
        # print('-=-=', query.shape)
        # print('-=-=', key.shape)
        # print('-=-=', value.shape)
        '''
        -=-= torch.Size([2, 4, 512])
        -=-= torch.Size([2, 4, 512])
        -=-= torch.Size([2, 4, 512])
        '''
        # 所以mask的形状 torch.Size([1, 8, 4, 4])  这里的所有参数都是4维度的   进过dropout的也是4维度的
        x = linrec(query, key, value, mask=mask, dropout=self.dropout)
        # contiguous解释:https://zhuanlan.zhihu.com/p/64551412
        # 这里相当于图中concat过程
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)

        return self.linears[-1](x)









class LinRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(LinRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.linrec_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.linrec_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_linrec_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.linrec_layernorms.append(new_linrec_layernorm)
            new_linrec_layer =  MultiHeadedLinrec(args.num_heads,args.hidden_units,args.dropout_rate)
            self.linrec_layers.append(new_linrec_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        # tl = seqs.shape[1] # time dim len for enforce causality
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.linrec_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.linrec_layernorms[i](seqs)
            mha_outputs = self.linrec_layers[i](Q, seqs, seqs)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            # seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)




