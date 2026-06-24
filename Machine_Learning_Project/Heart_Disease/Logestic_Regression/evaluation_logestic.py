from training_logestic import training_logestic
from preprocessing import preprocess_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , ConfusionMatrixDisplay , f1_score , RocCurveDisplay , PrecisionRecallDisplay , roc_auc_score

train_data , train_label , val_data , val_label ,  test_data , test_label , preprocesser=preprocess_data()
log_model ,coefs=training_logestic()

def evaluation_logestic_model(log_model=log_model, val_data=val_data, val_label=val_label,  test_data=test_data, test_label=test_label):
    # Validation probabilities
    y_proba=log_model.predict_proba(val_data)[:,1]
    treshold=np.linspace(0,1,200)
    f_score=[]
    for t in treshold:
        y_val_pred=(y_proba>=t).astype(int)
        f_score.append(f1_score(val_label , y_val_pred))

    best_treshold=treshold[np.argmax(f_score)]
    print('best threshold is:' , best_treshold)

    # Test evaluation
    y_test_proba=log_model.predict_proba(test_data)[:,1]
    y_pred=(y_test_proba>=best_treshold).astype(int)

    test_f1 = f1_score(test_label,y_pred)

    print(f"Test F1-score: {test_f1:.3f}")

    print(classification_report(test_label , y_pred))

    ConfusionMatrixDisplay.from_predictions(test_label , y_pred)
    plt.title(f"Test Confusion Matrix\nThreshold={best_treshold:.3f}")
    plt.show()

    RocCurveDisplay.from_predictions(test_label , y_test_proba )
    plt.savefig('RocCurveDisplay_logestic.png',dpi=300,bbox_inches='tight')
    plt.title('RocCurveDisplay')
    plt.show()
    
    PrecisionRecallDisplay.from_predictions(test_label , y_test_proba)
    plt.title('PrecisionRecallDisplay')
    plt.show()

    auc = roc_auc_score(test_label, y_test_proba)

    #print(f"ROC-AUC: {auc:.3f}")
    
    return {
        'best_threshold': round(best_treshold, 2),
        'test_f1': round(test_f1, 2),
        'predictions': y_pred,
        'probabilities': np.round(y_test_proba, 2),
        'roc_auc': round(auc, 2)
}