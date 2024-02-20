from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import GPT2Tokenizer
import json


class CustomBPETokenizer:
    def __init__(self, dataset, vocab_size=50000, max_length=64):
        with open(dataset, 'r') as file:
            data_txt = file.read()
        data_txt = data_txt.split('\n')
        self.data = data_txt
        self.bpe_tokenizer = self.train_bpe_tokenizer(vocab_size)
        self.gpt2_tokenizer = self.convert_to_gpt2_tokenizer()
        self.max_length = max_length

    def train_bpe_tokenizer(self, vocab_size):
        bpe_tokenizer = Tokenizer(models.BPE())

        bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        bpe_tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], # no masking because already done in attention 
            vocab_size=vocab_size,
            show_progress=True,
        )

        bpe_tokenizer.train_from_iterator(self.data, trainer=trainer)

        return bpe_tokenizer

    def convert_to_gpt2_tokenizer(self):
        model = json.loads(self.bpe_tokenizer.to_str())['model']
        vocab_dict = model['vocab']
        merges_list = model['merges']

        with open("./gpt2_from_bpe/vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        with open("./gpt2_from_bpe/merges.txt", "w") as merges_file:
            merges_file.write("\n".join(" ".join(map(str, merge)) for merge in merges_list))

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_from_bpe")

        return gpt2_tokenizer

    def tokenize(self, example):
        return self.gpt2_tokenizer.tokenize(example)

    def encode(self, example):
        return self.gpt2_tokenizer.encode(example)

    def decode(self, tokens):
        return self.gpt2_tokenizer.decode(tokens)


tokenizer = CustomBPETokenizer("data/wmt17_de_en_sentences.txt")



