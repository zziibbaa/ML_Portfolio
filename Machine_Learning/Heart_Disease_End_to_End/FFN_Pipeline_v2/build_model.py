import torch
import torch.nn as nn
import torch.nn.functional as F


class HeartFFN(nn.Module):
    def __init__(self , input_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 32)
        self.hidden = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        x = self.drop(x)
        x = F.leaky_relu(self.hidden(x))
        x = self.drop(x)
        return self.output(x)
