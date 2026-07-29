import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor



df= pd.read_csv('../pythonProject/Supervize_Learning/04.Linear_Regression/Advertising.csv')



from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



mlflow.set_tracking_uri("http://host.docker.internal:5555")
print('tracking to :' , mlflow.get_tracking_uri())
#set experiment name
mlflow.set_experiment("first_project")
# create and log a run
with mlflow.start_run():
    X=df.drop('sales' , axis=1)
    y=df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=101)
    scale=StandardScaler()
    model=RandomForestRegressor(n_estimators=50 , random_state=101)
    pipe_model=Pipeline([('scale' , scale),('model' , model)])
    print("Fitting model...")
    pipe_model.fit(X_train , y_train)
    print("Predicting...")
    pred=pipe_model.predict(X_val)
    rmse=np.sqrt(mean_squared_error(y_val , pred))

    # infer model signature and input example
    signature=infer_signature(X , pred)
    input_example = X[:5] #a small batch as sample input
    # log parametrs and metric
    mlflow.log_params({'model_type' : 'RandomForestRegressor',
                       'n_estimators' : model.n_estimators , 
                       'random_state' : model.random_state , 
                       'data_path' : 'Advertising.csv'})
    mlflow.log_metric('rmse' , rmse)
    #log model with signature and example
    mlflow.sklearn.log_model(pipe_model,
                             artifact_path='pipe_model',
                             signature=signature,
                             input_example=input_example)
    print(f'run loged with rmse {rmse:.2f}')



from joblib import dump , load
dump_model=dump(pipe_model , 'final_model.pkl')
dump_column=dump(list(X.columns) , 'column_name.pkl')