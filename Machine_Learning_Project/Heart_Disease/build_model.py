import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(input_dim=20):
    class Heart_diseaseFFN(nn.Module):
        def __init__(self):
            super().__init__()

            self.input=nn.Linear(input_dim,32)
            self.hidden=nn.Linear(32,16)
            self.output=nn.Linear(16,2)

            self.drop_out=nn.Dropout(.3)


        def forward(self,x):
            x=F.leaky_relu(self.input(x))
            x=self.drop_out(x)
            
            x=F.leaky_relu(self.hidden(x))
            x=self.drop_out(x)

            return self.output(x)

    model=Heart_diseaseFFN()
    loss_fun=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters() , lr=.001)
    return model , loss_fun , optimizer