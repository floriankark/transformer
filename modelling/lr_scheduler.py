import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# inherit from LambdaLR to get get_last_lr() and step() methods for free
class TransformerLRScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(TransformerLRScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        arg1 = step ** (-0.5) if step != 0 else 1 ** (-0.5)
        arg2 = step * self.warmup_steps ** (-1.5)
        return self.d_model ** (-0.5) * min(arg1, arg2)
