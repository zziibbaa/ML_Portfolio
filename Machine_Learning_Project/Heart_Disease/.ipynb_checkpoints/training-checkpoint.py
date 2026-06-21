import torch
import numpy as np
import copy
from build_model import build_model
from preprocessing import preprocess_data

train_loader , test_loader , preprocesser=preprocess_data()

def training(train_loader , test_loader , epochs=80):
    train_acc=[]
    test_acc=[]
    losses=torch.zeros(epochs)
    best_model={'accuracy' : 0,
                'epoch':0 ,
                'model': None}
    sample_x, _ = next(iter(train_loader))

    input_dim = sample_x.shape[1]

    model, loss_fun, optimizer = build_model(input_dim)

    for epoch_i in range(epochs):
        model.train()
        batch_acc=[]
        batch_loss=[]

        for x , y in train_loader:
            yHat=model(x)
            loss=loss_fun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_acc.append(100*torch.mean((torch.argmax(yHat , dim=1)==y).float()).item())
            batch_loss.append(loss.item())

        train_acc.append(np.mean(batch_acc))
        losses[epoch_i]=np.mean(batch_loss)


        model.eval()
        with torch.no_grad():
            x,y=next(iter(test_loader))
            y_pred=model(x)
            test_acc.append(100*torch.mean((torch.argmax(y_pred , dim=1)==y).float()).item())

            if test_acc[-1]>best_model['accuracy']:
                best_model['accuracy']=test_acc[-1]
                best_model['epoch']=epoch_i+1
                best_model['model']=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model['model'])
    return train_acc , test_acc , losses , model , best_model