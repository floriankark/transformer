import torch
import torch.nn as nn
import math

"""
# naive implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe = torch.zeros(1, seq_len, d_model)
        x = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1) / torch.pow(
            10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        pe[:, :, 0::2] = torch.sin(x)
        pe[:, :, 1::2] = torch.cos(x)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]
"""


"""# more efficient
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
        # x: (batch_size, seq_len, d_model)

        print(x.size(), self.pe[:, : x.size(1), :].size(), x.size(1))
        return x + self.pe[:, : x.size(1), :]"""

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return x + self.pe[:x.size(0)]