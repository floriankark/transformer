import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from modelling.lr_scheduler import TransformerLRScheduler
from modelling.functional import TransformerModel
from tqdm import tqdm
import numpy as np
from dataset import MyDataset
from utils import make_mask, collate
from transformers import GPT2Tokenizer
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# hyperparameters from https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_hparams.py
label_smoothing=0.1
optimizer="adam"
epsilon=1e-6
beta1=0.85
beta2=0.997
momentum=0.9
learning_rate=0.1


d_model = 512
dim_feedforward = 4*d_model
batch_size = 1024 # the larger the better convergence, but the slower the training (1024 max for a100) and maybe worse generalization
src_pad_idx = 0
num_epochs = 30
grad_clip = 1.0
# Always but most efficient with multiples of 8; on A100, multiples of 64.
# Table 1. https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
vocab_size = 50048 # nearest multiple of 64 see https://twitter.com/karpathy/status/1621578354024677377
compile_model = True
seed = 1337
torch.manual_seed(seed) # seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed(seed)

model = TransformerModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=dim_feedforward,
    dropout=0.1,
    max_len=64,
    norm_first=True # enables pre-norm
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Device: ", device)

print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# model with torch.compile results in a significant speedup. Speedup mainly comes from reducing Python overhead and GPU read/writes, speedup may vary on factors such as model architecture and batch size. 
# For example, if architecture is simple and the amount of data is large, then the bottleneck would be GPU compute and the observed speedup may be less significant.
# Different speedup results depending on "mode". The "reduce-overhead" mode uses CUDA graphs to further reduce the overhead of Python. Experiment with different modes to maximize speedup. More: https://pytorch.org/get-started/pytorch-2.0/#user-experience
# First few iterations torch.compile is significantly slower than the other runs, although it is much faster than the first run. This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.
if compile_model:
    # make sure not to have copy/deepcopy -> throws /bin/ld: skipping incompatible /usr/lib/libcuda.so when searching for -lcuda
    model = torch.compile(model) # 25% faster and allows for more batch size 


train_data = torch.load("/gpfs/project/flkar101/transformer_project/data/train_dataset.pt")
val_data = torch.load("/gpfs/project/flkar101/transformer_project/data/val_dataset.pt")

tokenizer = GPT2Tokenizer.from_pretrained("/gpfs/project/flkar101/transformer_project/gpt2_from_bpe")

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

#trainset_1 = torch.utils.data.Subset(train_dataset, range(0, 100000))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
#valset_1 = torch.utils.data.Subset(val_dataset, range(0, 1000))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)

optimizer_grouped_parameters = [
    {'params': [param for name, param in model.named_parameters()
                if 'bias' in name or 'layer_norm' in name], 'weight_decay': 0.0},
    {'params': [param for name, param in model.named_parameters()
                if 'bias' not in name and 'layer_norm' not in name], 'weight_decay': 0.01}
]


# small batches use lr = 0.5 and large batches use lr = 1.0 -> finding verified by https://arxiv.org/pdf/1806.00187.pdf
# setting lr=1.0 corresponds to max lr of 7e-4 for adamw (for very many steps ~400k-500k, make sure to set min lr to 7e-5 as per Chincilla from DeepMind)
optimizer = AdamW(optimizer_grouped_parameters, lr=1.0, betas=(0.9, 0.98), eps=1e-09, weight_decay=0.01, fused=False) # fused=True leads to unscaling the gradients twice https://github.com/pytorch/pytorch/issues/90752
accumulation_steps = 8 # simulating training on this number of gpu nodes (8 for 1024 batch size, 16 for 512 batch size)
lr_scheduler = TransformerLRScheduler(optimizer, d_model=d_model, warmup_steps=4000)
criterion = CrossEntropyLoss(ignore_index=src_pad_idx, label_smoothing=label_smoothing)

def validation(model, val_loader, src_pad_idx, device):
    print("Validation processing...")
    model.eval()
    valid_losses = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            src_input, tgt_input, tgt_output = batch['src_input'], batch['tgt_input'], batch['tgt_output']
            src_input, tgt_input, tgt_output = src_input.to(device), tgt_input.to(device), tgt_output.to(device)

            enc_attn_mask, dec_attn_mask = make_mask(src_input, tgt_input, src_pad_idx)
            enc_attn_mask, dec_attn_mask = enc_attn_mask.to(device), dec_attn_mask.to(device)

            output = model(src=src_input, tgt=tgt_input, enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask)

            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

            valid_losses.append(loss.item())
            del src_input, tgt_input, tgt_output, enc_attn_mask, dec_attn_mask, output

    mean_valid_loss = np.mean(valid_losses)
    msg = f"Validation loss: {mean_valid_loss:.3f}"
    print(msg)
    return mean_valid_loss

print("Training...")

best_loss = float('inf')
validloss_curr_epoch = 0
loss_step = []
mean_loss_list = []
valid_loss_list = []
for epoch in range(num_epochs):
    loss_epoch = []
    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, batch in enumerate(pbar):
        src_input, tgt_input, tgt_output = batch['src_input'], batch['tgt_input'], batch['tgt_output']
        enc_attn_mask, dec_attn_mask = make_mask(src_input, tgt_input, src_pad_idx)

        src_input, tgt_input, tgt_output = src_input.to(device), tgt_input.to(device), tgt_output.to(device)
        enc_attn_mask, dec_attn_mask = enc_attn_mask.to(device), dec_attn_mask.to(device)

        optimizer.zero_grad(set_to_none=True) if i % accumulation_steps == 0 else None # set_to_none=True here can modestly improve performance

        # Switching to torch.bfloat16 from fp16 can resolve overflow issues during training due to its significantly larger dynamic range, despite sacrificing a bit of precision.
        with autocast(dtype=torch.float16):
            output = model(src=src_input, tgt=tgt_input, enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))

        scaler.scale(loss).backward()

        # gradient accumulation
        if ((i+1) % accumulation_steps == 0 or i+1 == len(train_loader)):

            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        loss_step.append(loss.item())
        loss_epoch.append(loss.item())

        pbar.set_postfix({'loss': round(loss.item(), 2), 'lr': round(lr_scheduler.get_last_lr()[0], 10)})
    
    loss_curr_epoch = np.mean(loss_epoch)
    valid_loss_curr_epoch = validation(model, val_loader, src_pad_idx, device)

    msg = (f'| epoch {epoch+1}/{num_epochs} | train loss: {loss_curr_epoch:.3f} ' 
           f'| validation loss: {valid_loss_curr_epoch:.3f} | ppl: {np.exp(loss_curr_epoch):.2f} |')
    print(msg)
    mean_loss_list.append(loss_curr_epoch)
    valid_loss_list.append(valid_loss_curr_epoch)

    # safe lists
    np.save("/gpfs/project/flkar101/transformer_project/results/loss_list.npy", loss_step)
    np.save("/gpfs/project/flkar101/transformer_project/results/mean_loss_list.npy", mean_loss_list)
    np.save("/gpfs/project/flkar101/transformer_project/results/valid_loss_list.npy", valid_loss_list)

    if validloss_curr_epoch < best_loss:
        best_loss = valid_loss_curr_epoch
        state_dict = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': best_loss
        }
        torch.save(state_dict, "/gpfs/project/flkar101/transformer_project/results/best_val_loss_model_base_test.pth")
        print("Best checkpoint is saved with epoch = ", epoch)