"""#%%
from datasets import load_from_disk
from transformers import GPT2Tokenizer

train_data = load_from_disk("data/wmt17_de_en_train")
print(train_data)
item = train_data['translation']
print(item[0])
# %%
item[3]["de"]
# %%
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_from_bpe")
encoded_source = tokenizer.encode(source)
print(encoded_source)
print(type(encoded_source))
# %%
bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
print(type(bos_token_id))
print(type(encoded_source))
encoded_source = [bos_token_id] + encoded_source
# %%
pad_length = 64
def add_padding_or_truncate(tokenized_text):
            if len(tokenized_text) < pad_length:
                left = pad_length - len(tokenized_text)
                padding = [tokenizer.convert_tokens_to_ids("[PAD]")] * left
                tokenized_text += padding
            else:
                tokenized_text = tokenized_text[:pad_length]

            return tokenized_text
encoded_source = add_padding_or_truncate(encoded_source)
print(encoded_source)
# %%
import torch
encoded_source = torch.tensor(encoded_source, dtype=torch.long)
print(encoded_source)
# %%
print(len(encoded_source))
# %%
from datasets import load_from_disk
from transformers import GPT2Tokenizer
import torch
from dataset import MyDataset
from torch.utils.data import DataLoader

train_data = load_from_disk("./data/wmt17_de_en_train")

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_from_bpe")

train_dataset = MyDataset(train_data, tokenizer=tokenizer)

# %%
print(train_dataset[0])

# %%
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# %%
# get first batch
first_batch = next(iter(train_loader))
print(first_batch)
# %%
from datasets import load_from_disk
from transformers import GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_from_bpe")

train_data = load_from_disk("data/wmt17_de_en_train")
test_data = load_from_disk("data/wmt17_de_en_test")
val_data = load_from_disk("data/wmt17_de_en_val")

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

# %%
train_data = torch.load("data/train_dataset.pt")
val_data = torch.load("data/val_dataset.pt")

train_data = train_data.remove_columns(["translation"])
val_data = val_data.remove_columns(["translation"])

# safe as torch dataset
torch.save(train_data, "data/train_dataset.pt")
torch.save(val_data, "data/val_dataset.pt")
# %%
import torch

train_data = torch.load("data/train_dataset.pt")
print(train_data[0])

#train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)"""
# %%
