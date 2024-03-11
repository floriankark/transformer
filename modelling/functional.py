import copy
import torch
import torch.nn as nn
from math import sqrt
from typing import Optional

from modelling.attention import MultiHeadAttention
from modelling.positional_encoding import PositionalEncoding
from modelling.word_embedding import WordEmbedding

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        norm_first: bool = False,
        ):
        super().__init__()
        self.dropout = dropout
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        # see https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_hparams.py#L144
        # pytorch uses eps=1e-5
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6) # eps=1e-6, bias=True 
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6) # eps=1e-6, bias=True
    
    def _sa_block(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask)
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.ffn(x))

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):

        # Pre Norm as in Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        encoder_layer, 
        num_layers, 
        norm: Optional[nn.Module] = None,
        ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        output = src

        for mod in self.layers:
            output = mod(output, attn_mask=attn_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        norm_first: bool = False,
        ):
        super().__init__()
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, mask_future: bool = True) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, mask_future=mask_future)
        return self.dropout1(x)

    def _ca_block(self, x: torch.Tensor, enc_x: torch.Tensor, enc_attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.cross_attn(x, enc_x, enc_x, attn_mask=enc_attn_mask)
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout3(self.ffn(x))

    def forward(self, x, enc_x, enc_attn_mask: Optional[torch.Tensor] = None, dec_attn_mask: Optional[torch.Tensor] = None, 
                mask_future: bool = True) -> torch.Tensor:

        # Pre Norm as in Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask=dec_attn_mask, mask_future=mask_future)
            x = x + self._ca_block(self.norm2(x), enc_x, enc_attn_mask=enc_attn_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask=dec_attn_mask, mask_future=mask_future))
            x = self.norm2(x + self._ca_block(x, enc_x, enc_attn_mask=enc_attn_mask))
            x = self.norm3(x + self._ff_block(x))

        return x
    
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers, 
        norm: Optional[nn.Module] = None,
        ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt: torch.Tensor, enc_x: torch.Tensor, enc_attn_mask: Optional[torch.Tensor] = None,
                dec_attn_mask: Optional[torch.Tensor] = None, mask_future: bool = True) -> torch.Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(output, enc_x, enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask, mask_future=mask_future)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.src_embed = WordEmbedding(vocab_size, d_model)
        self.tgt_embed = WordEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.linear = nn.Linear(d_model, vocab_size, bias=True) # not sure if False or True 

        self._reset_parameters()
        self._tie_weights()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _tie_weights(self):
        self.src_embed.lut.weight = self.tgt_embed.lut.weight
        self.linear.weight = self.tgt_embed.lut.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, enc_attn_mask: Optional[torch.Tensor] = None, 
                dec_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    
        x_enc = self.pos_enc(self.src_embed(src))
        x_dec = self.pos_enc(self.tgt_embed(tgt))

        x_enc = self.encoder(x_enc, attn_mask=enc_attn_mask)
        x_dec = self.decoder(x_dec, x_enc, enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask)

        return self.linear(x_dec)
    
def uniform_unit_scaling_initializer(tensor: torch.Tensor, nonlinearity: str = "linear") -> torch.Tensor:
    """
    Initalizer which preserves output variance (i.e. doesn't scale variance) for 
    approximately gaussian distributed inputs.

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. If the input is `x` and the operation `x * W`,
    and we want to initialize `W` uniformly at random, we need to pick `W` from

        [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

    to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
    A similar calculation for convolutional networks gives an analogous result
    with `dim` equal to the product of the first 3 dimensions.  When
    nonlinearities are present, we need to multiply this by a constant factor 
    called `gain`. See (Sussillo et al., 2014) for deeper motivation.

    Args:
        tensor : `torch.Tensor`, required. 
            The tensor to initialise.
        nonlinearity : `str`, optional (default = `"linear"`)
            The non-linearity which is performed after the projection that this
            tensor is involved in. This must be the name of a function contained
            in the `torch.nn.functional` package.
            
    References:
        [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
        ([pdf](http://arxiv.org/pdf/1412.6558.pdf))
    
    See https://www.tensorflow.org/api_docs/python/tf/compat/v1/uniform_unit_scaling_initializer 
    for the original code.
    """
    
    size = 1.0
    # Estimating input size is not possible to do perfectly, but we try.
    # The estimate, obtained by multiplying all dimensions but the last one,
    # is the right thing for matrix multiply and convolutions (see above)
    for dimension in list(tensor.size())[:-1]:
        size *= dimension

    # Avoid errors when initializing zero-size tensors
    size = max(1.0, size)
    activation_scaling = torch.nn.init.calculate_gain(nonlinearity, tensor)
    max_value = sqrt(3 / size) * activation_scaling

    return tensor.data.uniform_(-max_value, max_value)
