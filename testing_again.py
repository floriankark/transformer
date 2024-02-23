import torch
from datasets import load_dataset
from dataset import MyDataset
from torch.utils.data import DataLoader

test_data = torch.load("/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")

test_dataset = MyDataset(test_data)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# get first batch
first_batch = next(iter(test_loader))

source, target_input, target_output = first_batch['source'], first_batch['target_input'], first_batch['target_output']

# print type of source
print(type(source[0]))
print(source[0])

print(type(torch.stack(source)))
print(torch.stack(source))
