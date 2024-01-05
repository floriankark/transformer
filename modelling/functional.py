import torch
import torch.nn as nn
from collections import OrderedDict

from modelling.attention import MultiHeadAttention


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        feature_dim,
        dropout=0.1,
        mask_future=False,
        bias=False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.self_attention = MultiHeadAttention(
            input_dim, num_heads, mask_future, bias
        )
        self.layer_norm_1 = nn.LayerNorm(input_dim)

        self.feature_transformation = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_dim, feature_dim)),
                    ("relu", nn.ReLU()),
                    ("linear2", nn.Linear(feature_dim, input_dim)),
                ]
            )
        )
        self.layer_norm_2 = nn.LayerNorm(input_dim)

    def forward(self, x, attention_mask=None):
        x = self.layer_norm_1(
            x + self.dropout(self.self_attention(x, x, x, attention_mask))
        )
        x = self.layer_norm_2(x + self.dropout(self.feature_transformation(x)))
        return x
