from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from modelling.lr_scheduler import TransformerLRScheduler
from modelling.functional import TransformerModel
from modelling.tokenizer import CustomBPETokenizer
from dataset import MyDataset
import pyarrow.parquet as pq


# Initialisieren Sie das Transformer-Modell
model = TransformerModel(
    vocab_size=10000,
    d_model=64,
    n_heads=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=256,
    dropout=0.1,
    max_len=100,
)

train_data = pq.read_table("data/data-00000-of-00003.arrow").to_pandas()
validation_data = pq.read_table("data/data-00001-of-00003.arrow").to_pandas()
test_data = pq.read_table("data/data-00002-of-00003.arrow").to_pandas()

tokenizer = CustomBPETokenizer("data/vocab.json", "data/merges.txt")
train_dataset = MyDataset(train_data, tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss_function = CrossEntropyLoss()
optimizer_grouped_parameters = [
    {"params": model.parameters(), "weight_decay": 0.01},
    {"params": model.encoder.bias.parameters(), "weight_decay": 0.0},
    {"params": model.decoder.bias.parameters(), "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=0.001, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = TransformerLRScheduler(optimizer, d_model=64, warmup_steps=4000)

for epoch in range(5):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_function(output, batch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
