# evaluation.py

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score , RocCurveDisplay
)


def evaluate_model(model, train_loader, val_loader , test_loader):
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

    with torch.no_grad():

        # Train probabilities
        train_logits = model(train_loader.dataset.tensors[0])

        train_proba = F.softmax(train_logits,dim=1)[:,1]

    # Convert to numpy
    train_proba = train_proba.detach().numpy()
    y_train = train_loader.dataset.tensors[1].detach().numpy()

    # Apply best threshold
    train_prediction = (train_proba >= best_threshold).astype(int)
    # ------------------------
    # Confusion Matrix
    # ------------------------
    ConfusionMatrixDisplay.from_predictions(y_train , train_prediction)
    plt.title(f"Train Confusion Matrix\nThreshold={best_threshold:.3f}")
    plt.show()
    ConfusionMatrixDisplay.from_predictions(y_test , pred)
    plt.title(f"Test Confusion Matrix\nThreshold={best_threshold:.3f}")
    plt.show()

    # ------------------------
    # Classification Report
    # ------------------------
    print(classification_report(y_test , pred))

    # ------------------------
    # Precision / Recall / F1 Plot
    # ------------------------
    report_dict=classification_report(test_loader.dataset.tensors[1] , pred , output_dict=True)
    df=pd.DataFrame(report_dict).T.loc[['0','1'] , 
                                  ['precision' , 'recall' , 'f1-score']]

    df.plot(kind='bar')
    plt.ylim(.72,1)
    plt.grid(axis='y')
    plt.savefig('classification_metrics.png',dpi=300,bbox_inches='tight')
    plt.show()
    RocCurveDisplay.from_predictions(test_loader.dataset.tensors[1] , y_proba)
    plt.show()

 

    print(f"Best Threshold: {best_threshold:.2f}")

    print(f"Test F1: {test_f1:.2f}")

    return {
    'best_threshold': best_threshold,
    'test_f1': test_f1,
    'report': report_dict}