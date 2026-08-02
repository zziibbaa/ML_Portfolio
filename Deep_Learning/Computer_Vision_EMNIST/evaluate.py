import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report , ConfusionMatrixDisplay
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')


def plot_loss(history):
    # loss & accuracy
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(history['train_loss'] , label='train')
    ax.plot(history['test_loss'] , label='test')
    ax.set_title('losses')
    ax.set_xlabel('epochs')
    ax.legend()
    plt.grid(alpha=.3)




    
def plot_accuracy(history):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(history['train_acc'] , label='train')
    ax.plot(history['test_acc'] , label='test')
    ax.set_title(f"Train:{history['train_acc'][-1]:.2f}%" f" | Test:{history['test_acc'][-1]:.2f}%")
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.legend()

def predict_model(test_loader , model , device):
    y_true=[]
    y_predict=[]
    images=[]
    model.eval()
    with torch.no_grad():
        
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)
            y_pred=model(x)
        
            y_true.append(y.cpu())
            y_predict.append(y_pred.cpu())
            images.append(x.cpu())

        y_true=torch.cat(y_true)
        y_predict=torch.cat(y_predict)
        pred=torch.argmax(y_predict,dim=1)
        images=torch.cat(images)
    return images , y_true, pred


def classification_report_plot(class_name , y_true , pred):
    # classification_report
    report= classification_report(y_true.cpu(), pred, target_names=class_name , output_dict=True)
    df=pd.DataFrame(report).T.loc[class_name,
                                 ['precision' , 'recall' , 'f1-score']]

    df.plot(kind='bar',figsize=(15,8))
    plt.grid(axis='y')
    plt.ylim([.75,1])


def plot_confusion_matrix(y_true , pred,class_name):
    #  ConfusionMatrixDisplay
    print("Confusion Matrix")
    fig,ax=plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay.from_predictions(y_true.detach().cpu() , pred.detach() , ax=ax , cmap='Blues' ,normalize='true'  , include_values=False)
    ax.set_xticks(range(len(class_name)))
    ax.set_xticklabels(class_name)

    ax.set_yticks(range(len(class_name)))
    ax.set_yticklabels(class_name)
    plt.title('TEST confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


    

def show_wrong_predictions(images , y_true, pred,class_name):   
    # mismatch classification
    idx_err=torch.where(y_true.cpu() != pred)[0]

    fig,axis=plt.subplots(3,8,figsize=(15,6))
    for i, ax in enumerate(axis.flatten()):
    
        random_idx=np.random.choice(len(idx_err))
        err_idx=idx_err[random_idx]
        img=images[err_idx].detach().cpu()
    
        ax.imshow(torch.squeeze(img).T , cmap='gray')

        true_label=class_name[y_true[err_idx]]
        pred_label=class_name[pred[err_idx]]
        ax.set_title(f'True:"{true_label}" , Pred:"{pred_label}"')

        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('true label VS false predict')
    plt.tight_layout()



     

def show_common_errors(class_name , y_true , pred):
    # error analysis

    print(f'false predict:{torch.sum(pred!=y_true.cpu())} number from {len(y_true)} sampel')

    
    # Analysis of most frequent errors
    idx_err = pred!=y_true.cpu()

    true_labels = y_true[idx_err].cpu().numpy()
    pred_labels = pred[idx_err].cpu().numpy()


    errors = Counter(zip(true_labels,pred_labels))

    result=[]

    for (true,pred),count in errors.most_common():
        result.append([class_name[true],class_name[pred],count])

    df=pd.DataFrame(result,columns=['True','Predict','Count'])
    return df.head(10)

def evaluate_model(model, history, test_loader, device, class_name):
    images , y_true, pred=predict_model(test_loader , model , device)
    show_wrong_predictions(images , y_true, pred,class_name)
    plot_loss(history)
    plot_accuracy(history)
    classification_report_plot(class_name , y_true,pred)
    plot_confusion_matrix(y_true , pred,class_name)
    error_df=show_common_errors(class_name , y_true , pred)

    return error_df



