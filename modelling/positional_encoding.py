import torch

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
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1)
    # 1/10000^(2*i/d_model) -> 10000^(-2*i/d_model) -> exp(log(10000^(-2*i/d_model))) -> exp(2*i*(-log(10000))/d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / d_model)
    )

    # broadcasting
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe
