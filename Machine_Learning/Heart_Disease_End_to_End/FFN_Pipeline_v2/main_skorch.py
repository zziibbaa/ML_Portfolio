from data_loader import load_data
from preprocessing import build_preprocessor
from training_skorch import training
from evaluation_skorch import evaluate_model

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

from pathlib import Path


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('Heart_Disease_Experiment')

def main_FFN_skorch():
    with mlflow.start_run(run_name='FFN_Skorch'):
        # load data
        data=load_data()
   
        # preprocessing
        # Features & Target
        data['DISEASE']=(data['DISEASE']>0).astype(int)
        x=data.drop('DISEASE' ,axis=1)
        y=data['DISEASE']
    
        # train_val_test_split
        train_data , temp_data , train_label , temp_label=train_test_split(x,y , test_size=.2 , random_state=101 , stratify=y)
        val_data , test_data , val_label , test_label=train_test_split(temp_data , temp_label , test_size=.5 , random_state=101 , stratify=temp_label)
        # ------------------------
        # Preprocessing
        # ------------------------
        preprocessor=build_preprocessor()
    
        preprocessor.fit(train_data)
        input_dim = preprocessor.transform(train_data).shape[1]
        # ------------------------
        # training model
        # ------------------------
        pipeline_model=training(input_dim , preprocessor)
        pipeline_model.fit(train_data,train_label)
    
        classifier=pipeline_model.named_steps['classifier']
        mlflow.log_param('model_type' , type(classifier).__name__)
        mlflow.log_params(classifier.get_params())

        
        predictions = pipeline_model.predict(train_data)
        signature = infer_signature(train_data,predictions)
        input_example = train_data.iloc[:1]
       
        # ------------------------
        # evaluating model
        # ------------------------

        results = evaluate_model(pipeline_model,val_data,val_label,test_data,test_label)

        mlflow.log_metric("best_validation_f1",results["best_validation_f1"])
        mlflow.log_metric("best_threshold",results["best_threshold"])
        mlflow.log_metric("test_f1",results["test_f1"])
        mlflow.log_metric("roc_auc", results["roc_auc"])
        mlflow.log_metric("precision_class1" , results["classification_report"]["1"]["precision"])
        mlflow.log_metric("recall_class1", results["classification_report"]["1"]["recall"])

        mlflow.log_figure(results["roc_fig"],"roc_curve.png")
        mlflow.log_figure(results["pr_fig"],"precision_recall.png")
        mlflow.log_figure(results["cm_fig"], "confusion_matrix.png")
        mlflow.log_figure(results["metrics_fig"], "classification_metrics.png")


        
        joblib.dump(pipeline_model, "heart_pipeline.pkl")
        mlflow.log_artifact("heart_pipeline.pkl")
       
        
        print("="*40)
        print(f"Best Threshold : {results['best_threshold']:.3f}")
        print(f"Test F1        : {results['test_f1']:.3f}")
        print(f"ROC AUC        : {results['roc_auc']:.3f}")




if __name__=="__main__":
    main_FFN_skorch()