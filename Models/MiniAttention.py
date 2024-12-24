import torch
from torch import nn
from einops import repeat

from Models.AttentionBlocks import SingleHeadSelfAttention, FeedForwardSingleLayer, SelfAttention, \
    FeedForwardMultiLayerGelu, FeedForwardMultiLayer, FeedForwardGeneral, SymbolAndTimeEmbedding
from Models.Noise import GaussianNoise


class TransformerLayerGeneral(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, attention_depth, rotary_emb, mlp_layer_widths, activation_fct, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.layers = nn.ModuleList([])
        for _ in range(attention_depth):
            self.layers.append(nn.ModuleList([
                SingleHeadSelfAttention(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, dim_out=dim_in, rotary_emb=rotary_emb, dropout=dropout),
                FeedForwardGeneral(layer_widths=mlp_layer_widths, activation_fct=activation_fct, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class TransformerGeneral(nn.Module):
    def __init__(self, *, dim_in, dim_attn, dim_qk, dim_v, attention_depth, dim_out, rotary_emb=True, mlp_layer_widths, activation_fct, dropout=0., noise=0.0):
        super().__init__()

        self.embedding = SymbolAndTimeEmbedding(1, 1)

        self.batch_norm = nn.BatchNorm1d(dim_in)

        self.noise = GaussianNoise(std=noise)

        self.to_input = nn.Linear(dim_in, dim_attn)

        # 1, 1, dim_attn
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_attn))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerLayerGeneral(dim_in=dim_attn, dim_qk=dim_qk, dim_v=dim_v, attention_depth=attention_depth, rotary_emb=rotary_emb, mlp_layer_widths=mlp_layer_widths, activation_fct=activation_fct, dropout=dropout)
        self.to_responders = nn.Linear(dim_attn, dim_out)

    def forward(self, x):

        x = self.embedding(x)

        x = self.batch_norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.noise(x)

        # N, Seq_L, dim_in -> N, Seq_L, dim_attn
        x = self.to_input(x)
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

    def save(self, path=".\\savedModels\\attention_nn.pt"):
        """
        Save the model state dictionary to a file.

        Parameters:
        - path: The file path where the model will be saved. Default is '..\\savedModels\\simple_nn.pt'.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, *args, **kwargs):
        """
        Load the model state dictionary from a file and create a new model instance.

        Parameters:
        - path: The file path from which to load the model.
        - *args, **kwargs: Additional arguments for the model's constructor.

        Returns:
        - An instance of SimpleNN with the loaded weights.
        """
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return model

