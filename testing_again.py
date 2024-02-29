"""import torch
from datasets import load_dataset
from dataset import MyDataset
from torch.utils.data import DataLoader

test_data = torch.load("/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")

test_dataset = MyDataset(test_data)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# get first batch
first_batch = next(iter(test_loader))

source, target_input, target_output = first_batch['source'], first_batch['target_input'], first_batch['target_output']

# decode source
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/gpfs/project/flkar101/transformer_project/gpt2_from_bpe")

decoded_source = tokenizer.decode(torch.tensor(source), skip_special_tokens=True)
print(decoded_source)
print("original source:" , torch.tensor(source))"""

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

q = torch.tensor([
    [[1.9269, 1.4873, 0.9007, -2.1055],
     [0.6784, -1.2345, -0.0431, -1.6047],
     [0.3559, -0.6866, -0.4934, 0.2415]],
    [[-1.1109, 0.0915, -2.3169, -0.2168],
     [-0.3097, -0.3957, 0.8034, -0.6216],
     [0.0000, 0.0000, 0.0000, 0.0000]]
])
k = q
v = q

mask_future = True

attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

from timeit import default_timer as timer

start = timer()
# input shape of q, k, v: (batch_size, seq_len, d_model)
mask = torch.zeros(q.size(1), k.size(1))
if mask_future:
    mask = mask.masked_fill(
        torch.triu(torch.ones(mask.shape), diagonal=1) == 1, float("-inf")
    )

mask = torch.stack([mask] * q.size(0))

repetition = int(
    q.size(0) / attention_mask.size(0)
)  # basically number of heads

if attention_mask is not None:
    attention_mask = (
        attention_mask.unsqueeze(-1)  # (batch_size, 1, key seq_len)
        .transpose(1, 2)  # (batch_size, key seq_len, 1)
        .repeat_interleave(
            repetition, dim=0
        )  # (batch_size*n_heads, key seq_len, 1)
        .expand_as(mask)  # (batch_size*n_heads, key seq_len, query seq_len)
    )

mask = mask.to(q.device)
mask = mask.masked_fill(attention_mask == 0, float("-inf"))

attention = torch.matmul(
    F.softmax(
        torch.matmul(q, k.transpose(1, 2)) / np.sqrt(q.size(-1)) + mask,
        dim=-1,
    ),
    v,
)
end = timer()
print(end - start)
print(attention)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

q = torch.tensor([
    [[1.9269, 1.4873, 0.9007, -2.1055],
     [0.6784, -1.2345, -0.0431, -1.6047],
     [0.3559, -0.6866, -0.4934, 0.2415]],
    [[-1.1109, 0.0915, -2.3169, -0.2168],
     [-0.3097, -0.3957, 0.8034, -0.6216],
     [0.0000, 0.0000, 0.0000, 0.0000]]
])
k = q
v = q

mask_future = True

attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

start = timer()
# input shape of q, k, v: (batch_size, seq_len, d_model)
L, S = q.size(-2), k.size(-2)
scale_factor = 1/math.sqrt(q.size(-1))
mask = torch.zeros(L, S, dtype=q.dtype, device=q.device)

if mask_future:
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    mask = mask.masked_fill(temp_mask.logical_not(), float("-inf"))

if attention_mask is not None:
    attn_bias = attention_mask.unsqueeze(1).expand(attention_mask.size(0), L, S)
    mask = mask.masked_fill(attn_bias == 0, float("-inf"))

attn_weight = torch.baddbmm(mask, q, k.transpose(-2, -1), beta=scale_factor, alpha=scale_factor)
attn_weight = F.softmax(attn_weight, dim=-1)
attention = torch.bmm(attn_weight, v)
end = timer()
print(end - start)
print(attention)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

q = torch.tensor([
    [[1.9269, 1.4873, 0.9007, -2.1055],
     [0.6784, -1.2345, -0.0431, -1.6047],
     [0.3559, -0.6866, -0.4934, 0.2415]],
    [[-1.1109, 0.0915, -2.3169, -0.2168],
     [-0.3097, -0.3957, 0.8034, -0.6216],
     [0.0000, 0.0000, 0.0000, 0.0000]]
])
k = q
v = q

mask_future = True

attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

start = timer()
# input shape of q, k, v: (batch_size, seq_len, d_model)
L, S = q.size(-2), k.size(-2)
scale_factor = 1/math.sqrt(q.size(-1))
mask = torch.zeros(L, S, dtype=q.dtype, device=q.device)

if mask_future:
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    mask = mask.masked_fill(temp_mask.logical_not(), float("-inf"))

if attention_mask is not None:
    attn_bias = attention_mask.unsqueeze(1).expand(attention_mask.size(0), L, S)
    mask = mask.masked_fill(attn_bias == 0, float("-inf"))

attention = F.scaled_dot_product_attention(q, k, v, mask)
end = timer()
print(end - start)
print(attention)
# %%
print(0.003239104989916086 / 0.0007131239399313927)
print(0.0016105390386655927 / 0.0007131239399313927)
# %%
"""# get mean and std of embedding weights
from modelling.word_embedding import WordEmbedding
import torch
import numpy as np

embedding = WordEmbedding(10000, 512)

# get mean and std of embedding weights
mean = torch.mean(embedding.embedding.weight * np.sqrt(512))
std = torch.std(embedding.embedding.weight * np.sqrt(512))
print(mean, std)"""
# %%
print(0.0031436249846592546/0.002215428976342082)
# %%
import torch
ATTENTION_MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])
ATTENTION_MASK.unsqueeze(1)
# %%
