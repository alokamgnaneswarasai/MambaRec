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
        # self.mamba_weight = torch.nn.Parameter(torch.tensor(0.5))
        self.mamba_weight = torch.nn.Linear(args.hidden_units, 1)
        
        # self.attention_weight = torch.nn.Parameter(torch.tensor(0.5))

        # Feed-Forward Network (Optional for added flexibility)
        self.feedforward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-6)
        self.feedforward_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_units, 4 * args.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * args.hidden_units, args.hidden_units)
        )
        
        # 2nd try
        # self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        # self.attention_layers = torch.nn.ModuleList()
        # self.forward_layernorms = torch.nn.ModuleList()
        # self.forward_layers = torch.nn.ModuleList()
        # self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        # for _ in range(args.num_blocks):
        #     new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        #     self.attention_layernorms.append(new_attn_layernorm)

        #     new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
        #                                                     args.num_heads,
        #                                                     args.dropout_rate)
        #     self.attention_layers.append(new_attn_layer)

        #     new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        #     self.forward_layernorms.append(new_fwd_layernorm)

        #     new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        #     self.forward_layers.append(new_fwd_layer)

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
        
        mamba_weight = torch.sigmoid(self.mamba_weight(seqs.mean(dim=0))).unsqueeze(-1)  # Mean pooling along sequence length
        # print('mamba_output',mamba_output.shape)        
        # mamba_weight = torch.sigmoid(self.mamba_weight(seqs))  # shape (seq_len, batch_size, 1)
        # print('mamba_weight',mamba_weight.shape)
        
        # Combine Mamba and Attention outputs using learnable weights
        combined_output = mamba_weight * mamba_output + (1 - mamba_weight) * attn_output

        # Process through Feed-Forward Network (Optional)
        feedforward_output = self.feedforward(combined_output)
        seqs = self.feedforward_dropout(feedforward_output) + combined_output  # Residual connection
        seqs = self.feedforward_layernorm(seqs)  # Layer normalization
        
        # # 2nd try
        # seqs = self.mamba1(seqs)
        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        # seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        # tl = seqs.shape[1] # time dim len for enforce causality
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # for i in range(len(self.attention_layers)):
        #     seqs = torch.transpose(seqs, 0, 1)
        #     Q = self.attention_layernorms[i](seqs)
        #     mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
        #                                     attn_mask=attention_mask)
        #                                     # key_padding_mask=timeline_mask
        #                                     # need_weights=False) this arg do not work?
        #     seqs = Q + mha_outputs
        #     seqs = torch.transpose(seqs, 0, 1)

        #     seqs = self.forward_layernorms[i](seqs)
        #     seqs = self.forward_layers[i](seqs)
        #     seqs *=  ~timeline_mask.unsqueeze(-1)

        # log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        # return log_feats

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




class HierarchicalSASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(HierarchicalSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.segment_len = 10  # Length of each segment
        self.downscale_factor = 32  # Downsampling factor
        
        # Embedding layers
        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)  
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Downsampling transformer
        self.downsampling_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_units,
                nhead=args.num_heads,
                dim_feedforward=3,
                dropout=args.dropout_rate,
            ),
            num_layers=args.num_blocks,
        )
        
        # Hierarchical transformer
        self.hierarchical_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_units,
                nhead=args.num_heads,
                dim_feedforward=3,
                dropout=args.dropout_rate,
            ),
            num_layers=args.num_blocks,
        )

        # Upsampling transformer
        self.upsampling_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_units,
                nhead=args.num_heads,
                dim_feedforward=3,
                dropout=args.dropout_rate,
            ),
            num_layers=args.num_blocks,
        )
        
        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

    def downsample(self, seqs):
        """
        Downsample the sequence by reducing the sequence length using fixed stride.
        """
        B, L, D = seqs.size()
        stride = self.downscale_factor
        downsampled_len = (L + stride - 1) // stride  # Calculate the reduced length
        downsampled_seqs = seqs[:, ::stride, :]  # Downsample with fixed stride
        return downsampled_seqs

    def upsample(self, downsampled_seqs, target_len):
        """
        Upsample the sequence back to the original length using interpolation.
        """
        B, downsampled_len, D = downsampled_seqs.size()
        interpolated = torch.nn.functional.interpolate(
            downsampled_seqs.permute(0, 2, 1),  # (B, D, downsampled_len)
            size=target_len,  # Upsample to target sequence length
            mode="linear",
            align_corners=True,
        )
        return interpolated.permute(0, 2, 1)  # Back to (B, target_len, D)

    def log2feats(self, log_seqs):
        """
        Process the input sequences hierarchically and return final representations.
        """
        # Embedding
        log_seqs= torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb(log_seqs.to(self.dev)) * (self.hidden_units ** 0.5)
        positions = np.tile(np.arange(log_seqs.shape[1]), (log_seqs.shape[0], 1))
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Mask padding tokens
        padding_mask = log_seqs == 0
        seqs = seqs * ~padding_mask.unsqueeze(-1)

        # Downsample
        downsampled_seqs = self.downsample(seqs)

        # Process with downsampling transformer
        downsampled_outputs = self.downsampling_transformer(downsampled_seqs.permute(1, 0, 2))
        downsampled_outputs = downsampled_outputs.permute(1, 0, 2)  # (B, reduced_len, D)
        

        # Hierarchical processing
        hierarchical_outputs = self.hierarchical_transformer(downsampled_outputs.permute(1, 0, 2))
        hierarchical_outputs = hierarchical_outputs.permute(1, 0, 2)

        # Upsample
        upsampled_seqs = self.upsample(hierarchical_outputs, target_len=seqs.size(1))

        # Process with upsampling transformer
        upsampled_outputs = self.upsampling_transformer(upsampled_seqs.permute(1, 0, 2))
        upsampled_outputs = upsampled_outputs.permute(1, 0, 2)  # (B, original_len, D)

        # Apply final layer normalization
        return self.last_layernorm(upsampled_outputs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
       
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Last timestep for prediction
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class MoEMambaRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MoEMambaRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.mamba_layers =Mamba(
                d_model=args.hidden_units,  # Embedding dimension
                d_state=32,                # SSM state expansion factor
                d_conv=4,                  # Local convolution width
                expand=2                   # Block expansion factor
            ).to(self.dev)
            

        self.moe_layers = MoE(
                d_model=args.hidden_units,
                num_experts=4,
                top_k=1
            ).to(self.dev)
            
        
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        seqs = self.mamba_layers(seqs)
        
        seqs = self.moe_layers(seqs)
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


class MoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
       
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # Gate to distribute input
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Combine outputs from top-k experts
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=-1)
        
        
        top_k_indices = top_k_indices.unsqueeze(-2).expand(-1, -1, expert_outputs.size(-2), -1)  # [batch_size, seq_len, d_model, top_k]
        
        top_k_outputs = torch.gather(expert_outputs, dim=-1, index=top_k_indices)  # [batch_size, seq_len, d_model, top_k]
        
        
        # Weighted sum of top-k expert outputs
        weighted_outputs = (top_k_outputs * top_k_values.unsqueeze(-2)).sum(dim=-1)
       
        return weighted_outputs
    
    
class SWA(nn.Module):
    """
    Sliding Window Attention (SWA) Layer
    """
    def __init__(self, d_model, window_size):
        super(SWA, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)

    def forward(self, x):
        # Implement sliding window mechanism
        seq_len = x.size(1)
        outputs = []
        for start in range(0, seq_len, self.window_size):
            end = min(start + self.window_size, seq_len)
            window_input = x[:, start:end, :]  # Extract window
            attn_output, _ = self.attention(window_input, window_input, window_input)
            outputs.append(attn_output)
        
        return torch.cat(outputs, dim=1)


class SAMBA4Rec(nn.Module):
    """
    SAMBA4Rec: Recommendation model with SAMBA architecture
    """
    def __init__(self, user_num, item_num, args):
        super(SAMBA4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        # Mamba layer
        self.mamba = Mamba(
            d_model=args.hidden_units,
            d_state=32,
            d_conv=4,
            expand=2,
        ).to(self.dev)

        # Sliding Window Attention (SWA) layer
        self.swa = SWA(d_model=args.hidden_units, window_size=4)  # Define window size

        # Fully connected MLP layers
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(args.hidden_units, args.hidden_units),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.dropout_rate)
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(args.hidden_units, args.hidden_units),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.dropout_rate)
        # )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # Apply Mamba layer
        mamba_output = self.mamba(seqs)

        # Apply MLP1
        # mlp1_output = self.mlp1(mamba_output)

        # Apply SWA
        # swa_output = self.swa(mlp1_output)

        # Apply MLP2
        # log_feats = self.mlp2(swa_output)
        
        log_feats =  self.swa(mamba_output)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Use the last output from the sequence

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F    
# class HourglassTransformer(nn.Module):
#     def __init__(self, user_num, item_num, args):
#         super(HourglassTransformer, self).__init__()
#         self.user_num = user_num
#         self.item_num = item_num
#         self.device = args.device
        
#         self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
#         self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
#         self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
#         self.encoder = nn.TransformerEncoderLayer(
#             d_model=args.hidden_units,
#             nhead=args.num_heads,
#             dropout=args.dropout_rate,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoderLayer(
#             d_model=args.hidden_units,
#             nhead=args.num_heads,
#             dropout=args.dropout_rate,
#             batch_first=True
#         )
        
#         self.num_layers = args.num_blocks
#         self.scaling_factors =  [2, 2]
    
#     def shortening(self, x, k):
#         """ Shift right and downsample """
#         x = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
#         return x[:, ::k, :]
    
#     def upsampling(self, x, x_shortened, k):
#         """ Restore sequence length """
#         x_upsampled = F.interpolate(x_shortened.permute(0, 2, 1), size=x.shape[1], mode='nearest')
#         x_upsampled = x_upsampled.permute(0, 2, 1)  # Restore original shape (batch, seq_len, features)
        
#         # print(f'x.shape: {x.shape}, x_shortened.shape: {x_shortened.shape}, x_upsampled.shape: {x_upsampled.shape}')
        
#         return x + x_upsampled  # Now x_upsampled has the same shape as x
    
#     def hourglass(self, x, scaling_factors):
        
#         # print(x.shape)
#         x = self.encoder(x)
        
#         if not scaling_factors:
#             return self.decoder(x, x)
        
#         k = scaling_factors[0]
#         x_shortened = self.shortening(x, k)
#         x_processed = self.hourglass(x_shortened, scaling_factors[1:])
#         x = self.upsampling(x, x_processed, k)
        
#         return self.decoder(x, x)
    
#     def log2feats(self, log_seqs):
#         seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
#         seqs += self.pos_emb(torch.arange(seqs.shape[1]).to(self.device))
#         seqs = self.emb_dropout(seqs)
        
#         return self.hourglass(seqs, self.scaling_factors)

#     def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
#         log_feats = self.log2feats(log_seqs)
#         pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
#         neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))
        
#         pos_logits = (log_feats * pos_embs).sum(dim=-1)
#         neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
#         return pos_logits, neg_logits
    
#     def predict(self, user_ids, log_seqs, item_indices):
#         log_feats = self.log2feats(log_seqs)
#         final_feat = log_feats[:, -1, :]
        
#         item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))
#         logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
#         return logits

# class HourglassTransformer(nn.Module):
#     def __init__(self, user_num, item_num, args):
#         super(HourglassTransformer, self).__init__()
#         self.user_num = user_num
#         self.item_num = item_num
#         self.device = args.device
        
#         self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
#         self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
#         self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
#         self.encoder = nn.TransformerEncoderLayer(
#             d_model=args.hidden_units,
#             nhead=args.num_heads,
#             dropout=args.dropout_rate,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoderLayer(
#             d_model=args.hidden_units,
#             nhead=args.num_heads,
#             dropout=args.dropout_rate,
#             batch_first=True
#         )
        
#         self.num_layers = args.num_blocks
#         self.scaling_factors =  [2, 2]

#     def attention_downsample(self, x, k):
#         """ Attention-based downsampling by aggregating attention over k steps """
#         attention_scores = torch.bmm(x, x.transpose(1, 2))  # Shape: (batch_size, seq_len, seq_len)
#         attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize across sequence
#         downsampled = torch.bmm(attention_weights, x)  # Apply attention weights to the sequence
        
#         # Now take only every k-th token (downsampling factor)
#         downsampled = downsampled[:, ::k, :]
#         return downsampled

#     def attention_upsample(self, x_shortened, target_len):
#         """ Attention-based upsampling by restoring sequence length """
#         attention_scores = torch.bmm(x_shortened, x_shortened.transpose(1, 2))  # Shape: (batch_size, reduced_len, reduced_len)
#         attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize
        
#         upsampled = torch.bmm(attention_weights, x_shortened)  # Apply attention
#         # Upsample the sequence length
#         upsampled = F.interpolate(upsampled.permute(0, 2, 1), size=target_len, mode='nearest')
#         upsampled = upsampled.permute(0, 2, 1)
#         return upsampled

#     def hourglass(self, x, scaling_factors):
#         """ Apply attention-based upsampling and downsampling recursively """
#         x = self.encoder(x)
        
#         if not scaling_factors:
#             return self.decoder(x, x)
        
#         k = scaling_factors[0]
#         x_shortened = self.attention_downsample(x, k)
#         x_processed = self.hourglass(x_shortened, scaling_factors[1:])
#         x_upsampled = self.attention_upsample(x_processed, x.shape[1])
        
#         return self.decoder(x_upsampled, x)

#     def log2feats(self, log_seqs):
#         seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
#         seqs += self.pos_emb(torch.arange(seqs.shape[1]).to(self.device))
#         seqs = self.emb_dropout(seqs)
        
#         return self.hourglass(seqs, self.scaling_factors)

#     def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
#         log_feats = self.log2feats(log_seqs)
#         pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
#         neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))
        
#         pos_logits = (log_feats * pos_embs).sum(dim=-1)
#         neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
#         return pos_logits, neg_logits
    
#     def predict(self, user_ids, log_seqs, item_indices):
#         log_feats = self.log2feats(log_seqs)
#         final_feat = log_feats[:, -1, :]
        
#         item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))
#         logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
#         return logits




class HourglassTransformer(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(HourglassTransformer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device
        
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model=args.hidden_units,
            nhead=args.num_heads,
            dropout=args.dropout_rate,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoderLayer(
            d_model=args.hidden_units,
            nhead=args.num_heads,
            dropout=args.dropout_rate,
            batch_first=True
        )
        
        self.num_layers = args.num_blocks
        self.scaling_factors = [8, 8]  # determine the downsampling and upsampling factors
        self.linear_upsample = nn.Linear(args.hidden_units, self.scaling_factors[0] * args.hidden_units)

    def attention_downsample(self, x, k):
        """ Attention-based downsampling (Dai et al., 2020) """
        
        
        # shape of x is (batch_size, seq_len, hidden_units) and k is the downsampling factor
        shortened = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=k, stride=k).permute(0, 2, 1) # shape of shortened is (batch_size, reduced_len, hidden_units),x is of shape (batch_size, seq_len, hidden_units)  , reduced_len = seq_len // k
        attention_scores = torch.bmm(shortened, x.transpose(1, 2))  # (batch_size, reduced_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1) # shape of attention_weights is (batch_size, reduced_len, seq_len)
        downsampled = torch.bmm(attention_weights, x) # (batch_size, reduced_len, hidden_units)
        return shortened + downsampled  # Residual connection

    def attention_upsample(self, x_shortened, x_original, k):
        """ Attention-based upsampling (Subramanian et al., 2020) """
        upsampled = self.linear_upsample(x_shortened)  # Linear upsampling
        upsampled = upsampled.view(x_shortened.shape[0], -1, x_original.shape[-1])  # Reshape to match original length
        
        attention_scores = torch.bmm(upsampled, x_original.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        upsampled_attention = torch.bmm(attention_weights, x_original)
        return upsampled + upsampled_attention  # Residual connection

    def hourglass(self, x, scaling_factors):
        """ Recursive attention-based downsampling and upsampling . This function is called recursively to apply the attention-based downsampling and upsampling. """
        
        x = self.encoder(x)
        
        if not scaling_factors:
            return self.decoder(x, x)
        
        k = scaling_factors[0]
        x_shortened = self.attention_downsample(x, k)
        x_processed = self.hourglass(x_shortened, scaling_factors[1:])
        x_upsampled = self.attention_upsample(x_processed, x, k)
        
        return self.decoder(x_upsampled, x)

    def log2feats(self, log_seqs):
        
        """ This function is used to convert the log sequence into features using the hourglass transformer"""
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        seqs += self.pos_emb(torch.arange(seqs.shape[1]).to(self.device))
        seqs = self.emb_dropout(seqs)
        
        return self.hourglass(seqs, self.scaling_factors)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)  # Convert log sequence to features shape of log_feats is (batch_size, seq_len, hidden_units)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits
    
    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # shape of logits is (batch_size, item_num) , item_num is the number of items 
        
        return logits




import torch
import torch.nn as nn
import numpy as np

class TransformerXLRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(TransformerXLRec, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.mem_len = 25
        
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        self.layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.layers.append(RelativeMultiheadTransformerXL(args.hidden_units, args.num_heads, args.dropout_rate))
        
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        
    def log2feats(self, log_seqs, mems=None):
        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        positions = torch.arange(log_seqs.shape[1], device=self.dev).expand(log_seqs.shape[0], -1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)  # Convert to tensor
        timeline_mask = (log_seqs == 0)

        seqs *= ~timeline_mask.unsqueeze(-1)
        
        if mems is None:
            mems = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            seqs, mems[i] = layer(seqs, mems[i])
            seqs *= ~timeline_mask.unsqueeze(-1)
        
        log_feats = self.last_layernorm(seqs)
        return log_feats, mems
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mems=None):        
        log_feats, mems = self.log2feats(log_seqs, mems)
        
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits
    
    def predict(self, user_ids, log_seqs, item_indices, mems=None):
        log_feats, mems = self.log2feats(log_seqs, mems)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

class RelativeMultiheadTransformerXL(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.ReLU(),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x, mem):
        if mem is not None:
            x = torch.cat([mem, x], dim=0)
        attn_output, _ = self.attn(x, x, x)
        x = self.layernorm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm(x + ffn_output)
        return x, x[-1].detach()



class CompressiveTransformerRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CompressiveTransformerRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.mem_len = 25  # Short-Term Memory (STM) length
        self.comp_len = 10  # Long-Term Memory (LTM) length
        self.compress_ratio = 2  # Compression rate (every 2 STM states → 1 LTM state)

        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.layers.append(CompressiveTransformerBlock(args.hidden_units, args.num_heads, args.dropout_rate))

        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def compress_memory(self, old_mems):
        seq_len, hidden_dim = old_mems.shape[-2], old_mems.shape[-1]
        
        # Ensure divisibility
        truncate_len = (seq_len // self.compress_ratio) * self.compress_ratio
        truncated_mems = old_mems[:truncate_len]

        # Reshape and apply mean pooling
        return truncated_mems.view(-1, self.compress_ratio, hidden_dim).mean(dim=1)


    def log2feats(self, log_seqs, stm=None, ltm=None):
        """ Computes item sequence features with compressive memory. """
        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = torch.arange(log_seqs.shape[1], device=self.dev).expand(log_seqs.shape[0], -1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        if stm is None:
            stm = [None] * len(self.layers)
        if ltm is None:
            ltm = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            seqs, new_stm = layer(seqs, stm[i], ltm[i])

            # Memory Update:
            # 1. Shift STM and truncate
            if new_stm is not None and new_stm.size(0) > self.mem_len:
                old_mems, new_stm = new_stm[:-self.mem_len], new_stm[-self.mem_len:]
                
                # 2. Compress and move to LTM
                new_ltm = self.compress_memory(old_mems)
            else:
                new_ltm = ltm[i]

            stm[i], ltm[i] = new_stm, new_ltm
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats, stm, ltm

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, stm=None, ltm=None):
        log_feats, stm, ltm = self.log2feats(log_seqs, stm, ltm)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices, stm=None, ltm=None):
        log_feats, stm, ltm = self.log2feats(log_seqs, stm, ltm)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class CompressiveTransformerBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.ReLU(),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, stm, ltm):
        """ Forward pass with short-term and long-term memory. """
        if stm is not None:
            x = torch.cat([stm, x], dim=0)  # Append STM to input

        if ltm is not None:
            x = torch.cat([ltm, x], dim=0)  # Append LTM to input

        attn_output, _ = self.attn(x, x, x)
        x = self.layernorm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm(x + ffn_output)

        return x, x[-1].detach()  # Return new STM state




import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


from typing import List, Optional
def shift_right(x: torch.Tensor):
    """
    This method shifts $i^{th}$ row of a matrix by $i$ columns.

    If the input is `[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]`, the shifted
    result would be `[[1, 2 ,3], [0, 4, 5], [6, 0, 7]]`.
    *Ideally we should mask out the lower triangle but it's ok for our purpose*.
    """

    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    #
    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
   

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        # print(heads, d_model, dropout_prob)
        super().__init__(heads, d_model, dropout_prob, bias=False)
        
        # Number of relative positions
        self.P = 2 ** 12

        # Relative positional embeddings for key relative to the query.
        # We need $2P$ embeddings because the keys can be before or after the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        

        # $\textcolor{orange}{R_k}$
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\textcolor{orange}{S_k}$
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        # $\textcolor{orange}{v^\top}$
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        # ${(\textcolor{lightgreen}{\mathbf{A + C}})}_{i,j} =
        # Q_i^\top K_j +
        # \textcolor{orange}{v^\top} K_j$
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        # $\textcolor{lightgreen}{\mathbf{B'}_{i,k}} = Q_i^\top \textcolor{orange}{R_k}$
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        # $\textcolor{lightgreen}{\mathbf{D'}_{i,k}} = \textcolor{orange}{S_k}$
        d = key_pos_bias[None, :, None, :]
        # Shift the rows of $\textcolor{lightgreen}{\mathbf{(B' + D')}_{i,k}}$
        # to get $$\textcolor{lightgreen}{\mathbf{(B + D)}_{i,j} = \mathbf{(B' + D')}_{i,i - j}}$$
        bd = shift_right(b + d)
        # Remove extra positions
        bd = bd[:, -key.shape[0]:]

      
        return ac + bd

class TransformerXLLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: RelativeMultiHeadAttention, dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.norm_linear = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor], mask: torch.Tensor):
        z = self.norm_self_attn(x)
        if mem is not None:
            mem = self.norm_self_attn(mem)
            
            m_z = torch.cat((mem, z), dim=0)
        else:
            m_z = z
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        x = x + self.dropout(self_attn)
        z = self.norm_linear(x)
        linear_out = self.linear(z)
        x = x + self.dropout(linear_out)
        return x

class TransformerXL(nn.Module):
    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor):
        new_mem = []
        for i, layer in enumerate(self.layers):
            new_mem.append(x.detach())
            m = mem[i] if mem else None
            x = layer(x=x, mem=m, mask=mask)
        return self.norm(x), new_mem

class TransformerXLEncoder(nn.Module):
    def __init__(self, num_items, embed_dim, num_layers, num_heads, hidden_dim, mem_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim)
        self.mem_length = mem_length
        # print(embed_dim)
        self.transformer = TransformerXL(
            TransformerXLLayer(
                d_model=embed_dim,
                self_attn=RelativeMultiHeadAttention(num_heads,embed_dim, dropout),
                dropout_prob=dropout
            ),
            n_layers=num_layers
        )
        self.linear = nn.Linear(embed_dim, num_items)
    
    def forward(self, x, memory=None):
        x = self.embedding(x)  # Shape: (B, S, D)
        x = x.permute(1, 0, 2)  # Shape: (S, B, D)
        mask = None  # Define mask if needed
        output, new_memory = self.transformer(x, memory, mask)
        logits = self.linear(output)  # Shape: (S, B, num_items)
        return logits.permute(1, 2, 0), new_memory  # Shape: (B, num_items, S)