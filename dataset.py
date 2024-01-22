import re
from datasets import load_dataset
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.tokenizer.encode(item)
        return item

    def __len__(self):
        return len(self.data)


dataset = load_dataset("wmt17", "de-en")

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


cleaned_dataset = dataset["train"].map(
    lambda example: {"translation": preprocess(example["translation"], 5, 64, 1.5)},
    num_proc=8,
)
cleaned_dataset = cleaned_dataset.filter(
    lambda example: example["translation"] is not None
)

cleaned_dataset.save_to_disk("data/wmt17_de_en")
