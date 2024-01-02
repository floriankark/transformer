# Implement the word embedding layer in pytorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
# naive implementation, work in progress
class WordEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.padding_idx = padding_idx

        if padding_idx is not None:
            nn.init.constant_(self.weights[padding_idx], 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embeddings = self.weights[input]
        embeddings[input == self.padding_idx] = 0
        return embeddings
"""


# wrapper for nn.Embedding
class WordEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)
