"""
    This is an implementation of transformer from scratch
    Source: https://www.youtube.com/watch?v=U0s0f995w14
    
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, heads):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads
        
        assert self.head_dim * heads == embedding_size, "Embedding size need to divisible by number of heads."
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * heads, self.embedding_size)
    
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]  # how many example we send in at the same time (batch size?)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # split embedding into self.heads pieces:
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        # QK^T
        energy = torch.einsum("nqhd, nkhd -> nhqk", queries, keys)
        
        if mask:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.head_dim ** 1 / 2), dim=3)
        
        # attention shape: (N, heads, q_len, k_len)
        # value shape: (N, v_len, heads, head_dim)
        # out shape after einsum: (N, q_len, heads, head_dim)
        # final out shape: (N, q_len, embedding_size) after flatten the last two dimensions
        out = torch.einsum("nhql, nlhd -> nqhd", attention, values).reshape(
            N, query_len, self.embedding_size
        )
        
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size)
        )
        
        self.dropout = dropout
    
    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        
        x = self.dropout(self.norm1(attention + queries))
        forward = self.ff(x)
        
        out = self.dropout(self.norm2(forward + x))
        
        return out


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embedding_size,
                 num_layers,
                 head,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.pos_embedding = nn.Embedding(max_length, embedding_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size,
                    head,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.pos_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_blocks = TransformerBlock(
            embedding_size, heads, dropout, forward_expansion
        )
        self.dropout = dropout
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_blocks(value, key, query, src_mask)
        
        return out


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embedding_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = dropout
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_voc_size,
                 trg_voc_size,
                 scr_pad_idx,
                 trg_pad_idx,
                 embedding_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device="cuda",
                 max_length=100
                 ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_voc_size,
            embedding_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        
        self.decoder = Decoder(
            trg_voc_size,
            embedding_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        
        self.src_pad_idx = scr_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)


    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, src_mask, trg_mask)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    exit()
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.tensor([[1,5,6,4,3,9,5,2,0],
                      [1,8,7,3,4,5,6,7,2]]).to(device)
    trg = torch.tensor([[1,7,4,3,5,9,2,0],
                        [1,5,6,2,4,7,6,2]]).to(device)
    
    src_pad_idx = 0
    trg_pad_idx = 0
    src_voc_size = 10
    trg_voc_size = 10
    
    model = Transformer(src_voc_size, trg_voc_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    
    print(out.shape)
    
    
    
