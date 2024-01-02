import torch
import torch.nn as nn

"""
def positional_encoding(seq_len, d_model):
    pos = torch.arange(0, seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2)
    pe = pos / torch.pow(10000, i / d_model)
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe
"""


# more efficient
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe = torch.zeros(1, seq_len, d_model)  # one uniform pe for all batches
        pos = torch.arange(0, seq_len).unsqueeze(1)
        # 1/10000^(2*i/d_model) -> 10000^(-2*i/d_model) -> exp(log(10000^(-2*i/d_model))) -> exp(2*i*(-log(10000))/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # broadcasting
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)  # we want pe constant and added to state_dict

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1), :]
