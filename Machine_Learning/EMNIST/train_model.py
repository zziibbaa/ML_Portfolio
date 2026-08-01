import torch
import numpy as np

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def train_model(epochs , train_loader , test_loader , model , optimizer, loss_fun, device=device,  scheduler=None , scheduler_position=None  ):
    train_losses=[]
    test_losses=[]
    train_acc=[]
    test_acc=[]

    model.to(device)

    for epoch_i in range(epochs):
        batch_acc=[]
        batch_loss=[]

        model.train()
        for x,y in train_loader:
            x=x.to(device)
            y=y.to(device)

            yhat=model(x)
            loss=loss_fun(yhat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                if scheduler_position=='batch':
                    scheduler.step()

            batch_loss.append(loss.item())
            batch_acc.append(100*torch.mean((torch.argmax(yhat , dim=1)==y).float()).item())

        if scheduler is not None:
            if scheduler_position=='epoch':
                scheduler.step()
        
        train_losses.append(np.mean(batch_loss))
        train_acc.append(np.mean(batch_acc))

        model.eval()
        with torch.no_grad():
            batch_test_acc=[]
            batch_test_loss=[]
            for x,y in test_loader:
                x=x.to(device)
                y=y.to(device)

                y_pred=model(x)
                loss=loss_fun(y_pred,y)
                batch_test_acc.append(100*torch.mean((torch.argmax(y_pred , dim=1)==y).float()).item())
                batch_test_loss.append(loss.item())

        test_acc.append(np.mean(batch_test_acc))
        test_losses.append(np.mean(batch_test_loss))

        print(f'Epoch:{epoch_i+1}/{epochs}')
        print(f' | Train Loss:{train_losses[-1]:.3f}')
        print(f' | Test Loss:{test_losses[-1]:.3f}')
        print(f' | Train Acc:{train_acc[-1]:.2f}%')
        print(f' | Test Acc:{test_acc[-1]:.2f}%')

        
    history={'train_loss':train_losses,'test_loss':test_losses,
             'train_acc':train_acc,'test_acc':test_acc}

  
    return model, history , device