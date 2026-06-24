import numpy as np
import pandas as pd

def load_data(path='../processed.cleveland.data' ):
    data=pd.read_csv(path , sep=',' , header=None)
    data.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','DISEASE']

    data=data.replace('?' , np.nan)
    Data=data.dropna()

    return data