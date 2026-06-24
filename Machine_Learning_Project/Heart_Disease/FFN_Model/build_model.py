import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import preprocess_data

train_loader ,_ ,_ ,_ = preprocess_data()
input_dim = train_loader.dataset.tensors[0].shape[1]
def build_model(input_dim=input_dim):

    class HeartFFN(nn.Module):
        def __init__(self):
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

    model = HeartFFN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer