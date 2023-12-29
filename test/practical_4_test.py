import sys
import os

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import BPETokenizer

CORPUS = [
    "hug hug hug hug hug hug hug hug hug hug pug pug pug pug pug pun pun pun pun pun pun pun pun pun pun pun pun bun bun bun bun hugs hugs hugs hugs hugs"
]

VOCAB_SIZE = 9

SENTENCE = "pugs"


def test_tokenize():
    tokenizer = BPETokenizer(CORPUS, VOCAB_SIZE)
    tokenizer.train()
    tokens = tokenizer.tokenize(SENTENCE)
    assert isinstance(tokens, list), "The tokenize method should return a list."
    assert all(
        isinstance(token, str) for token in tokens
    ), "All tokens should be strings."
    assert tokens == ["p", "ug", "s"], "The tokens do not match the expected output."
