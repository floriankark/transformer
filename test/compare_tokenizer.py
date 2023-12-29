import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import BPETokenizer
from tokenizers import CharBPETokenizer

with open("./test/corpus.txt", "r") as f:
    corpus = f.read().splitlines()

tokenizer = BPETokenizer(corpus, 64)
tokenizer.train()
tokens = tokenizer.tokenize("Machine learning is a subset of artificial intelligence.")

tokenizer = CharBPETokenizer()
tokenizer.train(["./test/corpus.txt"])
encoded = tokenizer.encode("Machine learning is a subset of artificial intelligence.")

print("-" * 50)
print(tokens)
print(encoded.tokens)
