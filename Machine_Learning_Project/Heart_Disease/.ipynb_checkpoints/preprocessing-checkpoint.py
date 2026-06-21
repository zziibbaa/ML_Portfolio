import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset , DataLoader
import torch

from data_loader import load_data
data=load_data()

def preprocess_data(data=data , test_size=.1 , random_state=101 , batch_size=8):
    # preprocessing
    # Features & Target
    data = data.copy()
    data['DISEASE'] = (data['DISEASE'] > 0).astype(int)

    x=data.drop('DISEASE' ,axis=1)
    y=data['DISEASE']

    train_data , test_data , train_label , test_label=train_test_split(x,y , test_size=test_size , random_state=random_state , stratify=y)

    # Column Groups
    countinus_col=['age' , 'trestbps', 'chol','thalach','oldpeak']
    categorical_col=['sex' , 'cp' , 'fbs' , 'restecg' , 'exang' , 'slope' , 'thal', 'ca']

    # Preprocessing
    preprocesser=ColumnTransformer(transformers=[('num' , StandardScaler(), countinus_col),
                                             ('cat' , OneHotEncoder(drop='first' , handle_unknown='ignore') , categorical_col)])

    train_data=preprocesser.fit_transform(train_data)
    test_data=preprocesser.transform(test_data)


    # convert to torch.tensor
    train_data=torch.tensor(train_data , dtype=torch.float32)
    test_data=torch.tensor(test_data , dtype=torch.float32)

    train_label=torch.tensor(train_label.values , dtype=torch.long)
    test_label=torch.tensor(test_label.values , dtype=torch.long)

    # convert to TensorDataset
    train_dataset=TensorDataset(train_data , train_label)
    test_dataset=TensorDataset(test_data , test_label)

    # batching of data
    train_loader=DataLoader(train_dataset , batch_size=batch_size , drop_last=True , shuffle=True)
    test_loader=DataLoader(test_dataset , batch_size=len(test_dataset))

    return train_loader , test_loader , preprocesser