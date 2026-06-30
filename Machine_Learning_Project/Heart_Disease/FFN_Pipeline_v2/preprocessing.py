import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder , FunctionTransformer
from sklearn.pipeline import Pipeline


def to_float32(x):
    return x.astype(np.float32)

def build_preprocessor():
    # Column Groups
    continus_col=['age' , 'trestbps', 'chol','thalach','oldpeak']
    categorical_col=['sex' , 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'thal', 'ca']

    # Preprocessing
    preprocessor = Pipeline([("transform", ColumnTransformer(transformers=[("num", StandardScaler(), continus_col),
                                                                           ("cat", OneHotEncoder(drop="first",handle_unknown="ignore"), categorical_col)])),
                             ("astype",FunctionTransformer(to_float32, accept_sparse=True))])
    return  preprocessor


   
