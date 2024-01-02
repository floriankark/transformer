from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import GPT2Tokenizer
import json


class CustomBPETokenizer:
    def __init__(self, dataset, vocab_size=50000):
        self.bpe_tokenizer = self.train_bpe_tokenizer(dataset, vocab_size)
        self.gpt2_tokenizer = self.convert_to_gpt2_tokenizer()

    def train_bpe_tokenizer(self, dataset, vocab_size):
        bpe_tokenizer = Tokenizer(models.BPE())

        bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        bpe_tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"],
            vocab_size=vocab_size,
            show_progress=True,
        )

        bpe_tokenizer.train_from_iterator(dataset, trainer=trainer)

        return bpe_tokenizer

    def convert_to_gpt2_tokenizer(self):
        vocab = self.bpe_tokenizer.get_vocab()
        merges = self.bpe_tokenizer.model._tokenizer.get_merges()

        with open("vocab.json", "w") as vocab_file:
            json.dump(vocab, vocab_file)
        with open("merges.txt", "w") as merges_file:
            merges_file.write("\n".join(" ".join(map(str, merge)) for merge in merges))

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            vocab_file="vocab.json", merges_file="merges.txt"
        )

        return gpt2_tokenizer

    def tokenize(self, text):
        return self.gpt2_tokenizer.tokenize(text)

    def encode(self, text):
        return self.gpt2_tokenizer.encode(text)

    def decode(self, tokens):
        return self.gpt2_tokenizer.decode(tokens)
