import pandas as pd
import joblib
from preprocessing import preprocess_data
from training_logestic import training_logestic
from evaluation_logestic import evaluation_logestic_model


def main():
    train_data,train_label,val_data,val_label,test_data,test_label,preprocesser = preprocess_data()

    log_model, coefs = training_logestic( train_data, train_label, preprocesser )

    results = evaluation_logestic_model()
    joblib.dump(preprocesser, "preprocesser.pkl")
    joblib.dump(log_model, "best_logistic_model.pkl")
    
    print("\n========== FINAL RESULTS ==========")
    print(f"Best Threshold : {results['best_threshold']:.3f}")
    print(f"Test F1 Score  : {results['test_f1']:.3f}")
    print(f"ROC-AUC        : {results['roc_auc']:.3f}")

  


if __name__ == "__main__":
    main()