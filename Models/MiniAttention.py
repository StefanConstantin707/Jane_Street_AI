import torch
from torch import nn
from einops import repeat

from Models.AttentionBlocks import SingleHeadSelfAttention, FeedForwardSingleLayer, SelfAttention, \
    FeedForwardMultiLayerGelu, FeedForwardMultiLayer
from Models.Noise import GaussianNoise


class TransformerLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, attention_depth, mlp_depth, heads, mlp_dim, rotary_emb, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.layers = nn.ModuleList([])
        for _ in range(attention_depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, dim_out=dim_in, heads=heads, dropout=dropout, rotary_emb=rotary_emb),
                FeedForwardMultiLayerGelu(dim_in=dim_in, dim_hidden=mlp_dim, dim_out=dim_in, depth=mlp_depth, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class SimpleTransformerLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, attention_depth, rotary_emb, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(attention_depth):
            self.layers.append(nn.ModuleList([
                SingleHeadSelfAttention(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, dim_out=dim_in, dropout=dropout, rotary_emb=rotary_emb),
                FeedForwardSingleLayer(dim_in=dim_in, dim_out=dim_in, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class MultiLayerMLPTransformerLayer(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, attention_depth, mlp_depth, mlp_dim, rotary_emb, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(attention_depth):
            self.layers.append(nn.ModuleList([
                SingleHeadSelfAttention(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, dim_out=dim_in, dropout=dropout, rotary_emb=rotary_emb),
                FeedForwardMultiLayer(dim_in=dim_in, dim_hidden=mlp_dim, dim_out=dim_in, depth=mlp_depth, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class FullSimpleTransformer(nn.Module):
    def __init__(self, *, dim_in, dim_attn, attention_depth, mlp_dim, mlp_depth, dim_out, rotary_emb=True, dropout=0., noise=0.0):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(dim_in)

        if noise > 0.0:
            self.noise = GaussianNoise(std=noise)
        else:
            self.noise = nn.Identity()
        self.to_input = nn.Linear(dim_in, dim_attn)

        # 1, 1, dim_attn
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_attn))
        self.dropout = nn.Dropout(dropout)

        self.transformer = MultiLayerMLPTransformerLayer(dim_in=dim_attn, dim_qk=dim_attn, dim_v=dim_attn, attention_depth=attention_depth, mlp_dim=mlp_dim, mlp_depth=mlp_depth, rotary_emb=rotary_emb, dropout=dropout)

        self.to_responders = nn.Linear(dim_attn, dim_out)

    def forward(self, x):
        x = self.batch_norm(x.transpose(-1, -2)).transpose(-1, -2)
        if self.training:
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

class FullSimpleDeepTransformer(nn.Module):
    def __init__(self, *, dim_in, dim_attn, attention_depth, dim_out, rotary_emb=True, dropout=0.):
        super().__init__()

        self.to_input = nn.Linear(dim_in, dim_attn)

        # 1, 1, dim_attn
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_attn))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerSingleHead(dim_in=dim_attn, dim_qk=dim_attn, dim_v=dim_attn, attention_depth=attention_depth, rotary_emb=rotary_emb, dropout=dropout)

        self.to_responders = nn.Linear(dim_attn, dim_out)

    def forward(self, x):
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

class FullTransformer(nn.Module):
    def __init__(self, *, dim_in, dim_attn, attention_depth, mlp_depth, dim_out, heads, rotary_emb=True, dropout=0.):
        super().__init__()

        self.to_input = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_attn),
            nn.LayerNorm(dim_attn),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, 30 + 1, dim_attn))

        # 1, 1, dim_attn
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_attn))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim_in=dim_attn, dim_qk=dim_attn, dim_v=dim_attn, attention_depth=attention_depth, mlp_depth=mlp_depth, heads=heads, mlp_dim=dim_attn, rotary_emb=rotary_emb, dropout=dropout)

        self.to_responders = nn.Linear(dim_attn, dim_out)

    def forward(self, x):
        # N, Seq_L, dim_in -> N, Seq_L, dim_attn
        x = self.to_input(x)
        b, s, _ = x.shape

        # 1, 1, dim_attn -> N, 1, dim_attn
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        # N, Seq_L, dim_attn + N, 1, dim_attn -> N, Seq_L + 1, dim_attn
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding
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

class FullDualTransformer(nn.Module):
    def __init__(self, *, dim_in, dim_attn, attention_depth, mlp_depth, dim_out, rotary_emb=True, dropout=0.):
        super().__init__()

        self.to_input = nn.Sequential(
            # nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_attn),
            # nn.LayerNorm(dim_attn),
        )

        # 1, 1, dim_attn
        self.cls_tokens_1 = nn.Parameter(torch.randn(1, 22, 1, dim_attn))
        self.cls_tokens_2 = nn.Parameter(torch.randn(1, 1, dim_attn))

        self.dropout = nn.Dropout(dropout)

        self.transformer1 = MultiLayerMLPTransformerLayer(dim_in=dim_attn, dim_qk=dim_attn, dim_v=dim_attn, attention_depth=attention_depth, mlp_depth=mlp_depth, mlp_dim=dim_attn, rotary_emb=rotary_emb, dropout=dropout)
        self.transformer2 = MultiLayerMLPTransformerLayer(dim_in=dim_attn, dim_qk=dim_attn, dim_v=dim_attn, attention_depth=attention_depth, mlp_depth=mlp_depth, mlp_dim=dim_attn, rotary_emb=rotary_emb, dropout=dropout)

        self.to_responders = nn.Linear(dim_attn, dim_out)
        self.batch = nn.BatchNorm1d(dim_out)

    def forward(self, x):
        # Seq_L = 968*2 = 88 * 22
        # N, Seq_L, dim_in -> N, Seq_L, dim_attn
        x = self.to_input(x)
        b, s, _ = x.shape

        # N, Seq_L, dim_in -> N, 22, 88, dim_attn
        x = x.view(b, 22, 88, -1)

        # 1, 22, 1, dim_attn -> N, 22, 1, dim_attn
        cls_tokens_1 = self.cls_tokens_1.repeat(b, 1, 1, 1)

        # N, 22, 88, dim_attn ++ N, 22, 1, dim_attn -> N, 22, 89, dim_attn
        x = torch.cat((cls_tokens_1, x), dim=2)
        x = self.dropout(x)

        # N, 22, 89, dim_attn -> N, 22, 89, dim_attn
        x = self.transformer1(x)

        # N, 22, 89, dim_attn -> N, 22, dim_attn
        x = x[:, :, 0]

        # Second Round #

        # 1, 1, dim_attn -> N, 1, dim_attn
        cls_tokens_2 = repeat(self.cls_tokens_2, '1 1 d -> b 1 d', b=b)

        # N, 22, dim_attn ++ N, 1, dim_attn -> N, 23, dim_attn
        x = torch.cat((cls_tokens_2, x), dim=1)
        x = self.dropout(x)

        # N, 23, dim_attn -> N, 23, dim_attn
        x = self.transformer2(x)

        # N, 23, dim_attn -> N, dim_attn
        x = x[:, 0]

        # N, dim_attn -> N, dim_out
        x = self.to_responders(x)
        return x

    def save(self, path=".\\savedModels\\dual_day_transformer.pt"):
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