import pandas as pd
import joblib
from preprocessing_logistic import build_preprocessor
from training_logistic import training_logistic
from evaluation_logistic import evaluation_logistic_model
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_loader import load_data


mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_experiment('Heart_Disease_Experiment')
REGISTERED_MODEL_NAME = "HeartDiseasePipeline"

def main():
    with mlflow.start_run(run_name="LogisticRegressionCV"):
        data=load_data()
        # preprocessing
        # Features & Target
        data['DISEASE']=(data['DISEASE']>0).astype(int)
        x=data.drop('DISEASE' ,axis=1)
        y=data['DISEASE']

        # train_val_test_split
        train_data , temp_data , train_label , temp_label=train_test_split(x,y , test_size=.2 , random_state=101, stratify=y)
        val_data , test_data , val_label , test_label=train_test_split(temp_data , temp_label , test_size=.5 , random_state=101 , stratify=temp_label)
     
        # preprocess
        preprocessor = build_preprocessor()
        # training model
        pipeline, coefs, fig = training_logistic( train_data, train_label, preprocessor )
        classifier = pipeline.named_steps["classifier"]
        
        mlflow.log_param('model_type' , type(classifier).__name__)
        mlflow.log_params(classifier.get_params())

        # define signature of model
        predictions=pipeline.predict(train_data)
        signature=infer_signature(train_data , predictions)
        input_example = train_data[:1]
        # evaluating model
        results = evaluation_logistic_model(pipeline, val_data, val_label,  test_data, test_label)
        mlflow.log_metric('best_threshold' , results['best_threshold'])
        mlflow.log_metric('test_f1' , results['test_f1'])
        mlflow.log_metric('roc_auc' , results['roc_auc'])

        # save artifact

        mlflow.log_figure(fig , "Feature_Importance.png")
        mlflow.log_figure(results['roc_fig'] , "roc_curve.png")
        mlflow.log_figure(results["cm_fig"], "confusion_matrix.png")
        model_info = mlflow.sklearn.log_model(sk_model=pipeline , name='Logistic_Model' , signature=signature,input_example=input_example , registered_model_name=REGISTERED_MODEL_NAME)
        print(f"Model URI: {model_info.model_uri}")
        
        print("\n========== FINAL RESULTS ==========")
        print(f"Best Threshold : {results['best_threshold']:.3f}")
        print(f"Test F1 Score  : {results['test_f1']:.3f}")
        print(f"ROC-AUC        : {results['roc_auc']:.3f}")

  


if __name__ == "__main__":
    main()