import torch
from torch import nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class RotationalPositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim=channels//2)

    def forward(self, q, k):
        return self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)


class FeedForwardSingleLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.):
        super().__init__()

        layers = []

        # Initial normalization and first linear layer
        # layers.append(nn.LayerNorm(dim_in))
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Create the network
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeedForwardMultiLayerGelu(nn.Module):
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


class FeedForwardMultiLayer(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, depth=2, dropout=0.):
        super().__init__()

        layers = []

        # Initial normalization and first linear layer
        layers.append(nn.Linear(dim_in, dim_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Add additional hidden layers based on depth
        for _ in range(depth - 2):  # We already added the first one above
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Final output layer
        layers.append(nn.Linear(dim_hidden, dim_out))
        layers.append(nn.Dropout(dropout))

        # Create the network
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeedForwardGeneral(nn.Module):
    def __init__(self, layer_widths: list, activation_fct, dropout=0.):
        super().__init__()
        layers = []
        depth = len(layer_widths) - 1

        for i in range(depth - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < depth - 2:
                if activation_fct == 'relu':
                    layers.append(nn.ReLU())
                elif activation_fct == 'gelu':
                    layers.append(nn.GELU())
                elif activation_fct == 'tanh':
                    layers.append(nn.Tanh())
                elif activation_fct == 'silu':
                    layers.append(nn.SiLU())
                else:
                    raise NotImplementedError(activation_fct)
            layers.append(nn.Dropout(dropout))

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

        self.rotary_emb = RotationalPositionalEncoding1D(dim_qk//heads) if rotary_emb else nn.Identity()

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


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, dim_out, rotary_emb, dropout = 0.0):
        super().__init__()

        self.scale = dim_qk ** -0.5

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

        q, k = self.rotary_emb(q, k)

        # N, Seq_L, dim_qk @ N, Seq_L, dim_qk -> N, Seq_L, Seq_L
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # N, Seq_L, Seq_L @ N, Seq_L, dim_v -> N, Seq_L, dim_v
        out = torch.matmul(attn, v)

        # N, Seq_L, dim_v -> N, Seq_L, dim_out
        return self.to_out(out)