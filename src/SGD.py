import torch
from typing import List

class SGD:
    def __init__(self, params: List[torch.nn.Parameter], lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= param.grad * self.lr
