print("Importing modules...")
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from modelling.lr_scheduler import TransformerLRScheduler
from modelling.functional import TransformerModel
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from dataset import MyDataset
print("All modules are imported.")

d_model = 128 # 64
batch_size = 32
src_pad_idx = 0
num_epochs = 3
vocab_size = 50000

model = TransformerModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=64, # 256
    dropout=0.1,
    max_len=64,
)

print("Loading datasets...")

train_dataset = torch.load("/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
val_dataset = torch.load("/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("All datasets are loaded.")

print("Setting up model, optimizer, and lr_scheduler...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Device: ", device)

print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer_grouped_parameters = [
    {'params': [param for name, param in model.named_parameters()
                if 'bias' in name or 'layer_norm' in name], 'weight_decay': 0.0},
    {'params': [param for name, param in model.named_parameters()
                if 'bias' not in name and 'layer_norm' not in name], 'weight_decay': 1e-2}
]

lr = 1 ** (-1 / 2) * d_model ** (-1 / 2)
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = TransformerLRScheduler(optimizer, d_model=d_model, warmup_steps=4000) # 1200

def make_mask(src_input, trg_input, pad_id):
    e_mask = (src_input != pad_id).int()
    d_mask = (trg_input != pad_id).int()
    return e_mask, d_mask

def validation(model, val_loader, src_pad_idx, vocab_size, device):
    print("Validation processing...")
    model.eval()
    valid_losses = []
    criterion = CrossEntropyLoss(ignore_index=src_pad_idx)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            src_input, trg_input, trg_output = batch['src'], batch['tgt_inp'], batch['tgt_out']
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            e_mask, d_mask = make_mask(src_input, trg_input, src_pad_idx)
            e_mask, d_mask = e_mask.to(device), d_mask.to(device)

            output = model(src_input, e_mask, trg_input, d_mask)

            loss = criterion(output.view(-1, vocab_size), trg_output.view(-1))

            valid_losses.append(loss.item())
            del src_input, trg_input, trg_output, e_mask, d_mask, output

    mean_valid_loss = np.mean(valid_losses)
    return mean_valid_loss

print("Training...")

best_loss = float('inf')
validloss_curr_epoch = 0
loss_list = []
valid_loss_list = []
for epoch in range(num_epochs):
    model.train()
    loss_step = []
    criterion = CrossEntropyLoss(ignore_index=src_pad_idx)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        print("Batch: ", batch)
        src_input, trg_input, trg_output = batch['source'], batch['target_input'], batch['target_output']
        print("src_input: ", src_input)
        e_mask, d_mask = make_mask(src_input, trg_input, src_pad_idx)

        src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)
        e_mask, d_mask = e_mask.to(device), d_mask.to(device)

        optimizer.zero_grad()


        output = model(src_input, e_mask, trg_input, d_mask)
        loss = criterion(output.view(-1, vocab_size), trg_output.view(-1))

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_curr_epoch = np.mean(loss_step)
    valid_loss_curr_epoch = validation(model, val_loader, src_pad_idx, vocab_size, device)

    # Print epoch results to screen
    msg = (
        f'Ep {epoch}/{num_epochs}:  Loss: Train {loss_curr_epoch:.3f}  Loss: Val {valid_loss_curr_epoch:.3f}')
    print(msg)
    loss_list.append(loss_curr_epoch)
    valid_loss_list.append(valid_loss_curr_epoch)

    # safe lists
    np.save("./results/loss_list.npy", loss_list)
    np.save("./results/valid_loss_list.npy", valid_loss_list)

    if validloss_curr_epoch < best_loss:
        best_loss = valid_loss_curr_epoch
        state_dict = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }
        torch.save(state_dict, f"./results/best_val_loss_model_epoch{epoch}.pth")
        print("Best checkpoint is saved with epoch = ", epoch)