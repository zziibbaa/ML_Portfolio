import torch
import torch.nn as nn

def build_optimizer_scheduler_loss_fun(model , optimizer_name , scheduler_name=None , optimizer_kwargs=None , scheduler_kwargs=None):
    if optimizer_kwargs is None:
        optimizer_kwargs={}
    if scheduler_kwargs is None:
        scheduler_kwargs={}

        
    optimizer_func=getattr(torch.optim , optimizer_name)
    optimizer=optimizer_func(model.parameters() , **optimizer_kwargs )

    scheduler=None
    if scheduler_name is not None:
        scheduler_func=getattr(torch.optim.lr_scheduler , scheduler_name)
        scheduler=scheduler_func(optimizer , **scheduler_kwargs)
    loss_fun=nn.CrossEntropyLoss()
    
    return optimizer , scheduler , loss_fun