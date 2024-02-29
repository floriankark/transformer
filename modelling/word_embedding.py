# Implement the word embedding layer in pytorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WordEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)
