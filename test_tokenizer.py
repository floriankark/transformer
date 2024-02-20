#%%

from datasets import load_dataset, load_from_disk

train_dataset = load_from_disk("data/wmt17_de_en_train")
val_dataset = load_from_disk("data/wmt17_de_en_val")

values_list_train = [list(d.values()) for d in train_dataset["translation"]]
values_list_val = [list(d.values()) for d in val_dataset["translation"]]

sentences_list_train = [item for sublist in values_list_train for item in sublist]
sentences_list = [item for sublist in values_list_val for item in sublist]

# combine both lists
sentences_list.extend(sentences_list_train)
# save as txt file
with open("data/wmt17_de_en_sentences.txt", "w") as file:
    for sentence in sentences_list:
        file.write(sentence + "\n")
# %%
# get first items data/test_dataset.pt
import torch
from dataset import MyDataset

test_dataset = torch.load("./data/test_dataset.pt")
print(test_dataset[0])
# %%
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_from_bpe")
print(tokenizer.bos_token_id)
print(tokenizer.convert_tokens_to_ids("[BOS]"))
# %%
