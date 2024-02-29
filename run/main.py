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
from transformers import GPT2Tokenizer
from torch.cuda.amp import GradScaler, autocast
from utils import make_mask
print("All modules are imported.")

d_model = 512
batch_size = 512
src_pad_idx = 0
num_epochs = 10
vocab_size = 50000

model = TransformerModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048, # 4 * d_model
    dropout=0.1, # test 0.2 -> paper says thats better
    max_len=64
)

print("Loading datasets...")

train_data = torch.load("/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
val_data = torch.load("/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")

tokenizer = GPT2Tokenizer.from_pretrained("/gpfs/project/flkar101/transformer_project/gpt2_from_bpe")

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

optimizer = AdamW(optimizer_grouped_parameters, lr=0.001, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = TransformerLRScheduler(optimizer, d_model=d_model, warmup_steps=4000) # 1200
criterion = CrossEntropyLoss(ignore_index=src_pad_idx, label_smoothing=0.1)
scaler = GradScaler()

def validation(model, val_loader, src_pad_idx, vocab_size, device):
    print("Validation processing...")
    model.eval()
    valid_losses = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            src_input, trg_input, trg_output = torch.stack(batch['source']), torch.stack(batch['target_input']), torch.stack(batch['target_output'])
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

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        src_input, trg_input, trg_output = torch.stack(batch['source']), torch.stack(batch['target_input']), torch.stack(batch['target_output'])
        e_mask, d_mask = make_mask(src_input, trg_input, src_pad_idx)

        src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)
        e_mask, d_mask = e_mask.to(device), d_mask.to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            output = model(src_input, trg_input, e_mask, d_mask)
            loss = criterion(output.view(-1, vocab_size), trg_output.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        loss_step.append(loss.item())
    
    loss_curr_epoch = np.mean(loss_step)
    valid_loss_curr_epoch = validation(model, val_loader, src_pad_idx, vocab_size, device)

    msg = (
        f'| epoch {epoch+1}/{num_epochs} | train loss: {loss_curr_epoch:.3f}  validation loss: {valid_loss_curr_epoch:.3f} |'
        f' ppl: {np.exp(loss_curr_epoch):.2f}')
    print(msg)
    loss_list.append(loss_curr_epoch)
    valid_loss_list.append(valid_loss_curr_epoch)

    # safe lists
    np.save("/gpfs/project/flkar101/transformer_project/results/loss_list.npy", loss_list)
    np.save("/gpfs/project/flkar101/transformer_project/results/valid_loss_list.npy", valid_loss_list)

    if validloss_curr_epoch < best_loss:
        best_loss = valid_loss_curr_epoch
        state_dict = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }
        torch.save(state_dict, "/gpfs/project/flkar101/transformer_project/results/best_val_loss_model_mini.pth")
        print("Best checkpoint is saved with epoch = ", epoch)