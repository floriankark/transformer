import copy
import torch
import torch.nn as nn
from math import sqrt
from typing import Optional

from modelling.attention import MultiHeadAttention
from modelling.positional_encoding import PositionalEncoding

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        norm_first=False,
        ):
        super().__init__()
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        # see https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_hparams.py#L144
        self.norm1 = nn.LayerNorm(d_model) # eps=1e-6, bool=True 
        self.norm2 = nn.LayerNorm(d_model) # eps=1e-6, bool=True
    
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
        norm=None,
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
        norm_first = False,
        ):
        super().__init__()
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(d_model, n_heads)

        self.cross_attn = MultiHeadAttention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

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
        norm=None
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

        self.padding_idx = 0
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.linear = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()
        self._init_weights()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self) -> None:
        # init and scale weights 
        # TODO: e.g. Kaparthy uses 0.02, maybe try that
        nn.init.normal_(self.embedding.weight, std=1/sqrt(self.d_model)) # (d_model,std) -> (512, 0.0442), (256, 0.0625), (128, 0.0884)    
        # remove bias and tie weights
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        self.linear.weight = self.embedding.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, enc_attn_mask: Optional[torch.Tensor] = None, 
                dec_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    
        x_enc = self.positional_encoding(self.embedding(src))
        x_dec = self.positional_encoding(self.embedding(tgt))

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
