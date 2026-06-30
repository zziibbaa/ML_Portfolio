from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor():
 
    
    # Column Groups
    countinus_col=['age' , 'trestbps', 'chol','thalach','oldpeak']
    categorical_col=['sex' , 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'thal', 'ca']

    # Preprocessing
    preprocessor=ColumnTransformer(transformers=[('num' , StandardScaler(), countinus_col),
                                                 ('cat' , OneHotEncoder(drop='first' , handle_unknown='ignore') , categorical_col)])


    return  preprocessor
