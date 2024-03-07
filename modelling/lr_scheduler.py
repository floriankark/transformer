from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# inherit from LambdaLR to get get_last_lr() and step() methods for free
class TransformerLRScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int = 4000) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(TransformerLRScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        step = step if step != 0 else 1
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
