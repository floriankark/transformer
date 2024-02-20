from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from modelling.lr_scheduler import TransformerLRScheduler
from modelling.functional import TransformerModel
from dataset import MyDataset, load_from_disk

d_model = 64
batch_size = 32

model = TransformerModel(
    vocab_size=10000,
    d_model=d_model,
    n_heads=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=256,
    dropout=0.1,
    max_len=100,
)

train_dataset = load_from_disk("data/train_dataset.pt")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_function = CrossEntropyLoss(ignore_index=0)

optimizer_grouped_parameters = [
    {'params': [param for name, param in model.named_parameters()
                if 'bias' in name or 'layer_norm' in name], 'weight_decay': 0.0},
    {'params': [param for name, param in model.named_parameters()
                if 'bias' not in name and 'layer_norm' not in name], 'weight_decay': 1e-2}
]

lr = 1 ** (-1 / 2) * d_model ** (-1 / 2)
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = TransformerLRScheduler(optimizer, d_model=64, warmup_steps=4000)

# try scaled fp16
for epoch in range(5):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_function(
            output.reshape(-1, output.size(-1)), batch[:, 1:].reshape(-1)
        )  # batch_size, seq_len, d_model
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
