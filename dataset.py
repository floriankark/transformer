import re
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def __len__(self):
        return len(self.data)


"""class MyDataset(Dataset):
    def __init__(self, data, tokenizer, pad_length=64):
        data = data
        tokenizer = tokenizer
        pad_length = pad_length

    def add_padding_or_truncate(self, tokenized_text):
            if len(tokenized_text) < pad_length:
                left = pad_length - len(tokenized_text)
                padding = [tokenizer.convert_tokens_to_ids("[PAD]")] * left
                tokenized_text += padding
            else:
                tokenized_text = tokenized_text[:pad_length]

            return tokenized_text

    def __getitem__(self, idx):
        item = data["translation"][idx]

        source = item["de"]
        target = item["en"]

        bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
        eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

        encoded_source = tokenizer.encode(source)
        encoded_target = tokenizer.encode(target)

        encoded_target_input = [bos_token_id] + encoded_target
        encoded_target_output = encoded_target + [eos_token_id]

        encoded_source = add_padding_or_truncate(encoded_source)
        encoded_target_input = add_padding_or_truncate(encoded_target_input)
        encoded_target_output = add_padding_or_truncate(encoded_target_output)

        encoded_source = torch.tensor(encoded_source, dtype=torch.long)
        encoded_target_input = torch.tensor(encoded_target_input, dtype=torch.long)
        encoded_target_output = torch.tensor(encoded_target_output, dtype=torch.long)

        return {
            "source": encoded_source,
            "target_input": encoded_target_input,
            "target_output": encoded_target_output,
        }

    def __len__(self):
        return len(data["translation"])"""


#dataset = load_dataset("wmt17", "de-en")

URL_PATTERN = re.compile(r"http\S+")
TAG_PATTERN = re.compile(r"<.*?>")


def clean_text(text):
    text = text.encode("utf-8").decode("utf-8")
    text = URL_PATTERN.sub("", text)
    text = TAG_PATTERN.sub("", text)
    whitelist = set(
        "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    )
    text = "".join(char.lower() for char in text if char in whitelist)
    return text


def filter_by_length(source, target, min_length, max_length):
    return (
        min_length <= len(source.split()) <= max_length
        and min_length <= len(target.split()) <= max_length
    )


def filter_by_ratio(source, target, ratio):
    return len(source.split()) / len(target.split()) < ratio


def preprocess(example, min_length=5, max_length=64, ratio=1.5):
    source = clean_text(example["de"])
    target = clean_text(example["en"])
    if filter_by_length(source, target, min_length, max_length) and filter_by_ratio(
        source, target, ratio
    ):
        example["de"] = source
        example["en"] = target
        return example
    else:
        return None


#cleaned_dataset = dataset["train"].map(
#    lambda example: {"translation": preprocess(example["translation"], 5, 64, 1.5)},
#    num_proc=8,
#)
#cleaned_dataset = cleaned_dataset.filter(
#    lambda example: example["translation"] is not None
#)
#
#cleaned_dataset.save_to_disk("data/wmt17_de_en")
    
#from datasets import load_from_disk
#from transformers import GPT2Tokenizer
#import torch

#train_data = load_from_disk("/gpfs/project/flkar101/transformer_project/data/wmt17_de_en_train") 
#test_data = load_from_disk("/gpfs/project/flkar101/transformer_project/data/wmt17_de_en_test")
#val_data = load_from_disk("/gpfs/project/flkar101/transformer_project/data/wmt17_de_en_val")

#tokenizer = GPT2Tokenizer.from_pretrained("/gpfs/project/flkar101/transformer_project/gpt2_from_bpe")

#train_dataset = MyDataset(train_data, tokenizer=tokenizer)
#test_dataset = MyDataset(test_data, tokenizer=tokenizer)
#val_dataset = MyDataset(val_data, tokenizer=tokenizer)

# safe as torch dataset
#torch.save(train_dataset, "/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
#torch.save(test_dataset, "/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")
#torch.save(val_dataset, "/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")
    
"""from datasets import load_from_disk
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("/gpfs/project/flkar101/transformer_project/gpt2_from_bpe")

train_data = load_from_disk("/gpfs/project/flkar101/transformer_project/data/wmt17_de_en_train")
test_data = load_from_disk("/gpfs/project/flkar101/transformer_project/data/wmt17_de_en_test")
val_data = load_from_disk("/gpfs/project/flkar101/transformer_project/data/wmt17_de_en_val")

pad_length = 64
def add_padding_or_truncate(tokenized_text):
    if len(tokenized_text) < pad_length:
        left = pad_length - len(tokenized_text)
        padding = [tokenizer.convert_tokens_to_ids("[PAD]")] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:pad_length]

    return tokenized_text

def preprocess_function(examples):
    source = examples["translation"]["de"]
    target = examples["translation"]["en"]

    bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    encoded_source = tokenizer.encode(source)
    encoded_target = tokenizer.encode(target)

    encoded_target_input = [bos_token_id] + encoded_target
    encoded_target_output = encoded_target + [eos_token_id]

    encoded_source = add_padding_or_truncate(encoded_source)
    encoded_target_input = add_padding_or_truncate(encoded_target_input)
    encoded_target_output = add_padding_or_truncate(encoded_target_output)

    return {
        "source": encoded_source,
        "target_input": encoded_target_input,
        "target_output": encoded_target_output,
    }

train_data = train_data.map(preprocess_function)
test_data = test_data.map(preprocess_function)
val_data = val_data.map(preprocess_function)

train_data = train_data.remove_columns(["translation"])
test_data = test_data.remove_columns(["translation"])
val_data = val_data.remove_columns(["translation"])

# safe as torch dataset
torch.save(train_data, "/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
torch.save(test_data, "/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")
torch.save(val_data, "/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")"""
