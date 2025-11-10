import torch.nn as nn
import torch
class ADD(nn.Module):
    #  Add two tensors

    def __init__(self, arg):
        super(ADD, self).__init__()
        # 128 256 512
        self.arg = arg

    def forward(self, x):
        return x[0] + x[1]