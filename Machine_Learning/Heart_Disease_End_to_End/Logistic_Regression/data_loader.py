import numpy as np
import pandas as pd

def load_data(path='../data/processed.cleveland.data' ):
    data=pd.read_csv(path , sep=',' , header=None)
    data.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','DISEASE']

    data = data.replace("?", np.nan)
    data = data.dropna()

    data["ca"] = pd.to_numeric(data["ca"], errors="coerce")
    data["thal"] = pd.to_numeric(data["thal"], errors="coerce")  

    data = data.astype({ "ca": int, "thal": int, "DISEASE": int})
    return data