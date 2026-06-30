import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay , classification_report , f1_score , RocCurveDisplay , roc_auc_score , PrecisionRecallDisplay

def evaluate_model(pipe_model, val_data , val_label, test_data , test_label):
    y_proba=pipe_model.predict_proba(val_data)[:,1]
    
    #####################################################
    # Search for the optimal classification
    # threshold based on validation F1-score
    threshold=np.linspace(0,1,200)
    f_score=[]

    for thresh_i in threshold:
        # Convert probabilities to class labels
        pred=(y_proba>=thresh_i).astype(int)
        # Compute F1-score for current threshold
        f_score.append(f1_score(val_label , pred))

    # Select threshold with highest F1-score
    best_threshold=threshold[np.argmax(f_score)]

    # Keep only probability of positive class (Disease = 1)
    test_proba=pipe_model.predict_proba(test_data)[:,1]
    # ----------------------------------------
    # Apply optimal threshold obtained
    # from validation set
    # ----------------------------------------
    prediction=(test_proba>=best_threshold).astype(int)
    # ----------------------------------------
    # Calculate F1-score on test set
    # ----------------------------------------
    test_f1=f1_score(test_label , prediction)


    
    #-----------------------------------------
    auc = roc_auc_score(test_label, test_proba)
    # ------------------------
    # ConfusionMatrixDisplay
    # ------------------------
    fig,ax=plt.subplots()
    ConfusionMatrixDisplay.from_predictions(test_label , prediction , ax=ax)
    ax.set_title(f"Test Confusion Matrix\nThreshold={best_threshold:.3f}")
    cm_fig = fig
    cm_fig.tight_layout()
    #------------------------
    #RocCurveDisplay
    #------------------------
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(test_label , test_proba , ax=ax)
    ax.set_title("ROC Curve")
    roc_fig = fig
    roc_fig.tight_layout()
    
    #------------------------
    #PrecisionRecallDisplay
    #------------------------
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(test_label , test_proba , ax=ax)
    ax.set_title("Precision Recall Curve")
    pr_fig = fig
    pr_fig.tight_layout()
    # ------------------------
    # Precision / Recall / F1 Plot
    # ------------------------
    report_dict=classification_report(test_label , prediction , output_dict=True)
    df=pd.DataFrame(report_dict).T.loc[['0','1'] , 
                                       ['precision' , 'recall' , 'f1-score']]
    
    fig, ax = plt.subplots()
    df.plot(kind='bar' , ax=ax)
    ax.set_title("Classification Metrics")
    metrics_fig=fig
    metrics_fig.tight_layout()
 
    print(report_dict)

    return {"best_threshold": best_threshold,

            "best_validation_f1": max(f_score),

            "test_f1": test_f1,

            "roc_auc": auc,

            "roc_fig": roc_fig,

            "pr_fig": pr_fig,

            "cm_fig": cm_fig,

            "metrics_fig": metrics_fig,

            "classification_report": report_dict,

            "predictions": prediction,

            "probabilities": test_proba}