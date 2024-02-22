import torch
from datasets import load_dataset

train_data = torch.load("/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
test_data = torch.load("/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")
val_data = torch.load("/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")

torch.save(train_data, "/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
torch.save(test_data, "/gpfs/project/flkar101/transformer_project/data/test_dataset.pt")
torch.save(val_data, "/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")
