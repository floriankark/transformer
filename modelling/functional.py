import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

from modelling.attention import MultiHeadAttention
from modelling.positional_encoding import PositionalEncoding
from modelling.word_embedding import Embedding

class LayerNorm(nn.Module):
    """ 
    LayerNorm with an optional bias. PyTorch nn.LayerNorm doesn't support bias=False until recently: 
    AttributeError: 'NoneType' object has no attribute 'zero_' -> https://github.com/pytorch/pytorch/issues/108048
    For benefits of bias=False, see https://arxiv.org/abs/1911.07013
    """

    def __init__(self, d_model, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        norm_first,
        ):
        super().__init__()
        self.dropout = dropout
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=True), # Paper includes bias however GPT2 uses bias=False
            nn.Dropout(dropout),
            nn.ReLU(), # no inplace=True, because I got enough gpu memory, if not, then inplace=True
            nn.Linear(dim_feedforward, d_model, bias=True),
            nn.Dropout(dropout),
        )
        # see https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_hparams.py#L144
        self.norm1 = LayerNorm(d_model, bias=False) # GPT2 uses eps=1e-5 and no bias
        self.norm2 = LayerNorm(d_model, bias=False)
    
    def _sa_block(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x, x, x, attn_mask=attn_mask)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

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
        num_layers, 
        d_model, 
        n_heads, 
        dim_feedforward, 
        dropout,
        norm_first,
        ):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, norm_first) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model, bias=False)
        self.norm_first = norm_first

    def forward(self, src: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        output = src

        for mod in self.layers:
            output = mod(output, attn_mask=attn_mask)

        if self.norm_first:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        norm_first,
        ):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=True),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model, bias=True),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model, bias=False)
        self.norm2 = LayerNorm(d_model, bias=False)
        self.norm3 = LayerNorm(d_model, bias=False)

    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, mask_future: bool = True) -> torch.Tensor:
        return self.self_attn(x, x, x, attn_mask=attn_mask, mask_future=mask_future)

    def _ca_block(self, x: torch.Tensor, enc_x: torch.Tensor, enc_attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.cross_attn(x, enc_x, enc_x, attn_mask=enc_attn_mask)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

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
        num_layers,
        d_model, 
        n_heads, 
        dim_feedforward, 
        dropout,
        norm_first,
        ):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, norm_first) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model, bias=False)
        self.norm_first = norm_first

    def forward(self, tgt: torch.Tensor, enc_x: torch.Tensor, enc_attn_mask: Optional[torch.Tensor] = None,
                dec_attn_mask: Optional[torch.Tensor] = None, mask_future: bool = True) -> torch.Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(output, enc_x, enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask, mask_future=mask_future)

        if self.norm_first:
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
        norm_first: bool = False,
        ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.src_embed = Embedding(vocab_size, d_model)
        self.tgt_embed = Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, n_heads, dim_feedforward, dropout, norm_first)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, n_heads, dim_feedforward, dropout, norm_first)

        self.linear = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._reset_parameters)
        self._tie_weights()

    """def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)"""
    
    def _reset_parameters(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    
    def _tie_weights(self):
        self.src_embed.lut.weight = self.tgt_embed.lut.weight
        self.linear.weight = self.tgt_embed.lut.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, enc_attn_mask: Optional[torch.Tensor] = None, 
                dec_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    
        x_enc = self.src_dropout(self.pos_enc(self.src_embed(src)))
        x_dec = self.tgt_dropout(self.pos_enc(self.tgt_embed(tgt)))

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

        for layer in self.encoder_layers:
            x = layer(x, encoder_attention_mask)
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
