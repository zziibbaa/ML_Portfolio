import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset , DataLoader
import torch

from data_loader import load_data
data=load_data()

def preprocess_data(data=data , random_state=101 ):
    # preprocessing
    # Features & Target
    data['DISEASE']=(data['DISEASE']>0).astype(int)
    x=data.drop('DISEASE' ,axis=1)
    y=data['DISEASE']
    

    train_data , temp_data , train_label , temp_label=train_test_split(x,y , test_size=.2 , random_state=random_state , stratify=y)
    val_data , test_data , val_label , test_label=train_test_split(temp_data , temp_label , test_size=.5 , random_state=101 , stratify=temp_label)
    # Column Groups
    countinus_col=['age' , 'trestbps', 'chol','thalach','oldpeak']
    categorical_col=['sex' , 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'thal', 'ca']

    # Preprocessing
    preprocesser=ColumnTransformer(transformers=[('num' , StandardScaler(), countinus_col),
                                                 ('cat' , OneHotEncoder(drop='first' , handle_unknown='ignore') , categorical_col)])

    train_data=preprocesser.fit_transform(train_data)
    val_data=preprocesser.transform(val_data)
    test_data=preprocesser.transform(test_data)


    return train_data , train_label , val_data , val_label ,  test_data , test_label , preprocesser