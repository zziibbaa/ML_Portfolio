from training import training
from preprocessing import preprocess_data
from data_loader import load_data

import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay , classification_report , f1_score , RocCurveDisplay

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

# ------------------------
# load data
# ------------------------
data=load_data()
# ------------------------
# Preprocessing
# ------------------------
train_loader , val_loader , test_loader , preprocesser=preprocess_data()
# ------------------------
# training model
# ------------------------
train_acc , val_acc , losses , model , best_model=training(train_loader , val_loader)



# ------------------------
# ploting of Losses & train-test accuracy
# ------------------------
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(losses)
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
ax[0].set_title('losses')

ax[1].plot(train_acc , label='train')
ax[1].plot(val_acc , label='validation')
ax[1].legend()
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')
ax[1].set_title(f'train_validation accuracy for Heart_Disease dataset with test accuracy {np.mean(val_acc[-10:]):.2f}%')
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
# Extract validation features and labels
x_val=val_loader.dataset.tensors[0]
y_val=val_loader.dataset.tensors[1]

# Generate prediction probabilities
# for the validation set
model.eval()
with torch.no_grad():
    y_predict=model(x_val)
    # Convert logits to probabilities
    # and keep probability of positive class
    # (Disease = 1)
    proba=F.softmax(y_predict , dim=1)[:,1]

# Convert tensors to numpy arrays
proba=proba.detach().numpy()
y_val=y_val.detach().numpy()
#####################################################
# Search for the optimal classification
# threshold based on validation F1-score
threshold=np.linspace(0,1,200)
f_score=[]

for thresh_i in threshold:
    # Convert probabilities to class labels
    pred=(proba>=thresh_i).astype(int)
    # Compute F1-score for current threshold
    f_score.append(f1_score(y_val,pred))

# Select threshold with highest F1-score
best_threshold=threshold[np.argmax(f_score)]

print(f"Best Threshold from Validation: "f"{best_threshold:.3f}")
print(f"Best Validation F1score: {np.max(f_score):.3f}")

# ----------------------------------------
# Extract test features and labels
# ----------------------------------------
x_test=test_loader.dataset.tensors[0]
y_test=test_loader.dataset.tensors[1]

# ----------------------------------------
# Put model in evaluation mode
# Disable gradient calculation
# ----------------------------------------
model.eval()
with torch.no_grad():
    # Forward pass
    logits=model(x_test)
    # Convert logits to class probabilities
    # Keep only probability of positive class (Disease = 1)
    y_proba=F.softmax(logits , dim=1)[:,1]
# ----------------------------------------
# Convert tensors to numpy arrays
# ----------------------------------------
y_proba=y_proba.detach().numpy()
y_test=y_test.detach().numpy()
# ----------------------------------------
# Apply optimal threshold obtained
# from validation set
# ----------------------------------------
pred=(y_proba>=best_threshold).astype(int)
# ----------------------------------------
# Calculate F1-score on test set
# ----------------------------------------
test_f1=f1_score(y_test , pred)
print(f"Test F1-score: {test_f1:.4f}")
# ------------------------
# Confusion Matrix
# ------------------------
# ------------------------
# Train Confusion Matrix
# ------------------------

with torch.no_grad():

    train_logits = model(train_loader.dataset.tensors[0])

    train_proba = F.softmax(train_logits, dim=1)[:,1]

train_proba = train_proba.detach().numpy()
y_train = train_loader.dataset.tensors[1].detach().numpy()

train_prediction = (train_proba >= best_threshold).astype(int)

ConfusionMatrixDisplay.from_predictions(y_train,train_prediction)
plt.title(f"Train Confusion Matrix\nThreshold={best_threshold:.3f}")
plt.show()


# ------------------------
# Test Confusion Matrix
# ------------------------

ConfusionMatrixDisplay.from_predictions(y_test,pred)
plt.title(f"Test Confusion Matrix\nThreshold={best_threshold:.3f}")
plt.show()


# ------------------------
# Classification Report
# ------------------------
print(classification_report(test_loader.dataset.tensors[1] , pred))

# ------------------------
# Precision / Recall / F1 Plot
# ------------------------
report=classification_report(test_loader.dataset.tensors[1] , pred , output_dict=True)
df=pd.DataFrame(report).T.loc[['0','1'] , 
                              ['precision' , 'recall' , 'f1-score']]

df.plot(kind='bar')
plt.ylim(.72,1)
plt.grid(axis='y')
plt.savefig('classification_metrics.png',dpi=300,bbox_inches='tight')
plt.show()
RocCurveDisplay.from_predictions(test_loader.dataset.tensors[1] , y_proba)
plt.savefig('RocCurveDisplay.png',dpi=300,bbox_inches='tight')
plt.show()

# ------------------------
# save of best model
# ------------------------
torch.save(model.state_dict(),'best_model.pth')

print(f"Best Validation Accuracy: {best_model['accuracy']:.2f}")

print(f"Train Accuracy (last epoch): {train_acc[-1]:.2f}")

print(f"Validation Accuracy (last epoch): {val_acc[-1]:.2f}")

print(f"Best Threshold: {best_threshold:.2f}")

print(f"Test F1: {test_f1:.2f}")
