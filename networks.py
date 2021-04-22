
""" 
Network definitions.

Probably going to use a pre-trained BERT encoder
and train a decoder model from scratch.
"""

import math
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertModel


class MultiheadSelfAttention(nn.Module):

    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.heads = num_heads
        self.model_dim = model_dim
        if self.model_dim % self.heads != 0:
            raise ValueError(f'Working dimension ({model_dim}) not a multiple of num_heads ({num_heads})')
        
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False) 
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_out=None):
        bs, n, _ = x.size()     
        q = self.query(x).view(bs, n, self.heads, self.model_dim // self.heads)                         # (bs, n, heads, d)

        if encoder_out is None:           
            k = self.key(x).view(bs, n, self.heads, self.model_dim // self.heads)                       # (bs, n, heads, d)
            v = self.value(x).view(bs, n, self.heads, self.model_dim // self.heads)                     # (bs, n, heads, d)
        else:
            bs, n_enc, _ = encoder_out.size()
            k = self.key(encoder_out).view(bs, n_enc, self.heads, self.model_dim // self.heads)         # (bs, n, heads, d)
            v = self.value(encoder_out).view(bs, n_enc, self.heads, self.model_dim // self.heads)       # (bs, n, heads, d)

        q = q.permute(0, 2, 1, 3).contiguous()                                                          # (bs, heads, n, d)
        k = k.permute(0, 2, 1, 3).contiguous()                                                          # (bs, heads, n, d)
        v = v.permute(0, 2, 1, 3).contiguous()                                                          # (bs, heads, n, d)

        attn_scores = torch.einsum('bhid,bhjd->bhij', [q, k]) / math.sqrt(self.model_dim)               # (bs, heads, n, n)                           
        attn_probs = F.softmax(attn_scores, dim=-1) 
        context = torch.einsum('bhij,bhjd->bhid', [attn_probs, v])                                      # (bs, heads, n, d)
        context = context.permute(0, 2, 1, 3).contiguous().view(bs, n, -1)                              # (bs, n, model_dim)
        return self.layer_norm(context + x)


class MaskedMultiheadSelfAttention(nn.Module):

    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.heads = num_heads
        self.model_dim = model_dim
        if self.model_dim % self.heads != 0:
            raise ValueError(f'Working dimension ({model_dim}) not a multiple of num_heads ({num_heads})')
        
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False) 
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        bs, n, _ = x.size()                     
        q = self.query(x).view(bs, n, self.heads, self.model_dim // self.heads)                     # (bs, n, heads, d)
        k = self.key(x).view(bs, n, self.heads, self.model_dim // self.heads)                       # (bs, n, heads, d)
        v = self.value(x).view(bs, n, self.heads, self.model_dim // self.heads)                     # (bs, n, heads, d)

        q = q.permute(0, 2, 1, 3).contiguous()                                                      # (bs, heads, n, d)
        k = k.permute(0, 2, 1, 3).contiguous()                                                      # (bs, heads, n, d)
        v = v.permute(0, 2, 1, 3).contiguous()                                                      # (bs, heads, n, d)

        attn_scores = torch.einsum('bhid,bhjd->bhij', [q, k]) / math.sqrt(self.model_dim)           # (bs, heads, n, n) 
        mask = torch.full((attn_scores.size(2), attn_scores.size(3)), -np.inf).to(x.device)
        mask = torch.triu(mask).fill_diagonal_(0).repeat(bs, self.heads, 1, 1)
        attn_probs = F.softmax(attn_scores + mask, dim=-1)

        context = torch.einsum('bhij,bhjd->bhid', [attn_probs, v])                                  # (bs, heads, n, d)
        context = context.permute(0, 2, 1, 3).contiguous().view(bs, n, -1)                          # (bs, n, model_dim)
        return self.layer_norm(context + x)


class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.fc_1 = nn.Linear(model_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, model_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        out = self.fc_2(self.gelu(self.fc_1(x)))
        return self.layer_norm(out + x)


class DecoderBlock(nn.Module):

    def __init__(self, heads, model_dim, ff_dim):
        super().__init__()
        self.masked_attention = MaskedMultiheadSelfAttention(num_heads=heads, model_dim=model_dim)
        self.attention = MultiheadSelfAttention(num_heads=heads, model_dim=model_dim)
        self.feedfwd = Feedforward(model_dim=model_dim, hidden_dim=ff_dim)

    def forward(self, x, encoder_out):
        x = self.masked_attention(x)
        x = self.attention(x, encoder_out=encoder_out)
        x = self.feedfwd(x)
        return x


class Decoder(nn.Module):

    def __init__(self, config, embedding_layer, tokenizer_vocab_size):
        super().__init__()
        heads = config["num_heads"]
        model_dim = config["model_dim"]
        ff_dim = config["ff_dim"]
        self.maxlen = config["max_length"]
        self.num_layers = config["num_blocks"]
        self.blocks = nn.ModuleList([DecoderBlock(heads, model_dim, ff_dim) for _ in range(self.num_layers)])
        self.linear_out = nn.Linear(in_features=model_dim, out_features=tokenizer_vocab_size, bias=True)
        self.decoder_embeddings = embedding_layer 

    def forward(self, x, encoder_out=None):
        x = self.decoder_embeddings(x)
        for i in range(self.num_layers):
            x = self.blocks[i](x, encoder_out)
        x = self.linear_out(x)
        return x