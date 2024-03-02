import torch
import torch.nn as nn
from collections import OrderedDict

from modelling.attention import MultiHeadAttention
from modelling.positional_encoding import PositionalEncoding
from modelling.word_embedding import WordEmbedding
from math import sqrt


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

        self.feature_transformation = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_dim, feature_dim)),
                    ("relu", nn.ReLU()),
                    ("linear2", nn.Linear(feature_dim, input_dim)),
                ]
            )
        )

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

    def forward(self, x, attention_mask=None):
        x_attn = self.layer_norm_1(
            x + self.dropout(self.self_attention(x, x, x, attention_mask))
        )
        x_ffn = self.layer_norm_2(x_attn + self.dropout(self.feature_transformation(x_attn)))
        return x_ffn


class TransformerDecoderLayer(nn.Module):
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
            input_dim, num_heads, mask_future=True, bias=bias
        )

        self.encoder_attention = MultiHeadAttention(
            input_dim, num_heads
        )

        self.feature_transformation = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_dim, feature_dim)),
                    ("relu", nn.ReLU()),
                    ("linear2", nn.Linear(feature_dim, input_dim)),
                ]
            )
        )

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

    def forward(
        self, x, encoder_output, encoder_attention_mask=None, attention_mask=None
    ):
        x_enc = self.layer_norm_1(
            x + self.dropout(self.self_attention(x, x, x, attention_mask))
        )
        x_dec = self.layer_norm_2(
            x_enc
            + self.dropout(
                self.encoder_attention(
                    x_enc, encoder_output, encoder_output, encoder_attention_mask
                )
            )
        )
        x_out = self.layer_norm_3(x_dec + self.dropout(self.feature_transformation(x_dec)))
        return x_out


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        max_len,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [
                BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, n_heads, dim_feedforward, dropout=dropout
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.linear = nn.Linear(d_model, vocab_size)  # batch x seq_len x vocab_size

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        """for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        self.linear.weight = self.embedding.embedding.weight"""

    def forward(self, x, y, encoder_attention_mask=None, decoder_attention_mask=None):
        x = self.positional_encoding(self.embedding(x) * sqrt(self.d_model))
        y = self.positional_encoding(self.embedding(y) * sqrt(self.d_model))

        for layer in self.encoder_layers:
            x = layer(x, encoder_attention_mask)

        for layer in self.decoder_layers:
            y = layer(y, x, encoder_attention_mask, decoder_attention_mask)

        return self.linear(y)
