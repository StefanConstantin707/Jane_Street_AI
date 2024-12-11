import torch
from torch import nn
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding


class RotationalPositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim=channels//2)

    def forward(self, q, k):
        return self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)


class FeedForwardMultiLayer(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, depth=2, dropout=0.):
        super().__init__()

        layers = []

        # Initial normalization and first linear layer
        layers.append(nn.LayerNorm(dim_in))
        layers.append(nn.Linear(dim_in, dim_hidden))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Add additional hidden layers based on depth
        for _ in range(depth - 2):  # We already added the first one above
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Final output layer
        layers.append(nn.Linear(dim_hidden, dim_out))
        layers.append(nn.Dropout(dropout))

        # Create the network
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, dim_out, rotary_emb, heads = 8, dropout = 0.0):
        super().__init__()

        dim_head = dim_v // heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim_in)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim_in, dim_qk, bias = False)
        self.to_k = nn.Linear(dim_in, dim_qk, bias = False)
        self.to_v = nn.Linear(dim_in, dim_v, bias = False)

        self.rotary_emb = RotationalPositionalEncoding1D(dim_qk) if rotary_emb else nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(dim_v, dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # N, Seq_L, dim_in -> N, Seq_L, dim_in
        x = self.norm(x)

        # N, Seq_L, dim_in -> N, Seq_L, dim_qk
        q = self.to_q(x)
        k = self.to_k(x)
        # N, Seq_L, dim_in -> N, Seq_L, dim_v
        v = self.to_v(x)

        # N, Seq_L, dim_qk/v -> N, H, Seq_L, dim_qk/v // H
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        q, k = self.rotary_emb(q, k)

        # N, H, Seq_L, dim_qk // H @ N, H, Seq_L, dim_qk // H -> N, H, Seq_L, Seq_L
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # N, H, Seq_L, Seq_L @ N, H, Seq_L, dim_v // H -> N, H, Seq_L, dim_v // H
        out = torch.matmul(attn, v)

        # N, H, Seq_L, dim_v // H -> N, Seq_L, dim_v
        out = rearrange(out, 'b h n d -> b n (h d)')

        # N, Seq_L, dim_v -> N, Seq_L, dim_out
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, attention_depth, mlp_depth, heads, mlp_dim, rotary_emb, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.layers = nn.ModuleList([])
        for _ in range(attention_depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, dim_out=dim_in, heads=heads, dropout=dropout, rotary_emb=rotary_emb),
                FeedForwardMultiLayer(dim_in=dim_in, dim_hidden=mlp_dim, dim_out=dim_in, depth=mlp_depth, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, *, dim_in, dim_attn, attention_depth, mlp_depth, dim_out, heads, rotary_emb=True, dropout=0.):
        super().__init__()

        self.to_input = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_attn),
            nn.LayerNorm(dim_attn),
        )

        # 1, 1, dim_attn
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_attn))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim_in=dim_in, dim_qk=dim_attn, dim_v=dim_attn, attention_depth=attention_depth, mlp_depth=mlp_depth, heads=heads, mlp_dim=dim_attn, rotary_emb=rotary_emb, dropout=dropout)

        self.to_responders = nn.Linear(dim_attn, dim_out)

    def forward(self, X):
        # N, Seq_L, dim_in -> N, Seq_L, dim_attn
        x = self.to_input(X)
        b, s, _ = x.shape

        # 1, 1, dim_attn -> N, 1, dim_attn
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        # N, Seq_L, dim_attn + N, 1, dim_attn -> N, Seq_L + 1, dim_attn
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        # N, Seq_L + 1, dim_attn -> N, Seq_L + 1, dim_attn
        x = self.transformer(x)

        # N, Seq_L + 1, dim_attn -> N, 1, dim_attn
        x = x[:, 0]

        # N, 1, dim_attn -> N, 1, dim_out -> N, dim_out
        x = self.to_responders(x).squeeze(1)

        return x