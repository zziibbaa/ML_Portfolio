from training import training
from preprocessing import preprocess_data
from data_loader import load_data

import torch
from sklearn.metrics import ConfusionMatrixDisplay , classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

# ------------------------
# load data
# ------------------------
data=load_data()
# ------------------------
# Preprocessing
# ------------------------
train_loader , test_loader , preprocesser=preprocess_data(test_size=.1 , random_state=101 , batch_size=8)
# ------------------------
# training model
# ------------------------
train_acc , test_acc , losses , model , best_model=training(train_loader , test_loader)



# ------------------------
# ploting of Losses & train-test accuracy
# ------------------------
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(losses)
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
ax[0].set_title('losses')

ax[1].plot(train_acc , label='train')
ax[1].plot(test_acc , label='test')
ax[1].legend()
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
ax[1].set_title(f'train_test accuracy for Heart_Disease dataset with test accuracy {np.mean(test_acc[-10:]):.2f}%')
plt.savefig('loss_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()


# ------------------------
# showing best model
# ------------------------
print(f"Best Validation Accuracy: " f"{best_model['accuracy']:.2f}%")

print(f"Best Epoch: " f"{best_model['epoch']}")


# ------------------------
# Predictions
# ------------------------
model.eval()
with torch.no_grad():
    train_prediction=model(train_loader.dataset.tensors[0])
    train_prediction=torch.argmax(train_prediction , dim=1)

    test_prediction=model(test_loader.dataset.tensors[0])
    test_prediction=torch.argmax(test_prediction , dim=1)

# ------------------------
# Confusion Matrix
# ------------------------

ConfusionMatrixDisplay.from_predictions(train_loader.dataset.tensors[1] , train_prediction)
plt.title("Train Confusion Matrix")
plt.show()
ConfusionMatrixDisplay.from_predictions(test_loader.dataset.tensors[1] , test_prediction)
plt.title("Test Confusion Matrix")
plt.show()

# ------------------------
# Classification Report
# ------------------------
print(classification_report(test_loader.dataset.tensors[1] , test_prediction))

# ------------------------
# Precision / Recall / F1 Plot
# ------------------------
report=classification_report(test_loader.dataset.tensors[1] , test_prediction , output_dict=True)
df=pd.DataFrame(report).T.loc[['0','1'] , 
                              ['precision' , 'recall' , 'f1-score']]

df.plot(kind='bar')
plt.ylim(.72,1)
plt.grid(axis='y')
plt.savefig('classification_metrics.png',dpi=300,bbox_inches='tight')
plt.show()

# ------------------------
# save of best model
# ------------------------
torch.save(model.state_dict(),'best_model.pth')

if __name__ == "__main__":
    main()