import re
from collections import defaultdict, Counter
import itertools


def word_frequency(corpus):
    word_freq = Counter()
    for sentence in corpus:
        word_freq.update(sentence.split())
    return {" ".join(word): freq for word, freq in word_freq.items()}


def co_occurrence_frequencies(word_freq):
    co_occurrence_freq = defaultdict(int)
    for word, freq in word_freq.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            co_occurrence_freq[symbols[i], symbols[i + 1]] += freq
    return co_occurrence_freq


def merge(pair, word_freq, vocab):
    word_freq = {
        word.replace(" ".join(pair), "".join(pair)): freq
        for word, freq in word_freq.items()
    }
    for word in word_freq:
        vocab.update(word.split())
    return word_freq


class BPETokenizer:
    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab = set()
        self.vocab_size = vocab_size
        self.merge_rules = defaultdict(str)

    def train(self):
        word_freq = word_frequency(self.corpus)
        while len(self.vocab) < self.vocab_size:
            co_occurrence_freq = co_occurrence_frequencies(word_freq)
            pair = max(co_occurrence_freq, key=co_occurrence_freq.get)
            self.merge_rules[pair] = "".join(pair)
            self.merge_rules = dict(
                sorted(self.merge_rules.items(), key=lambda item: len(item[1]))
            )
            word_freq = merge(pair, word_freq, self.vocab)

    def tokenize(self, text):
        text = text.split()
        text = [" ".join(word) for word in text]
        for i in range(len(text)):
            word = text[i]
            for pair, replacement in self.merge_rules.items():
                pair = " ".join(pair)
                word = word.replace(pair, replacement)
            text[i] = word.split()
        return list(itertools.chain(*text))
