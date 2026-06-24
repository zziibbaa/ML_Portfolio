import torch
import numpy as np
import copy
from build_model import build_model
from preprocessing import preprocess_data



train_loader, val_loader , test_loader , preprocesser=preprocess_data()


def training(train_loader , val_loader , epochs=50):
    train_acc=[]
    val_acc=[]
    losses=torch.zeros(epochs)
    best_model={'accuracy' : 0,
                'epoch':0 ,
                'model': None}
    model , loss_fun , optimizer=build_model()

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
            val_batch_acc=[]
            for x,y in val_loader:
                y_pred=model(x)
                val_batch_acc.append(100*torch.mean((torch.argmax(y_pred , dim=1)==y).float()).item())

            val_acc.append(np.mean(val_batch_acc))

            if val_acc[-1]>best_model['accuracy']:
                best_model['accuracy']=val_acc[-1]
                best_model['epoch']=epoch_i+1
                best_model['model']=copy.deepcopy(model.state_dict())
    if best_model['model'] is not None:
        model.load_state_dict(best_model['model'])
    return train_acc , val_acc , losses , model , best_model