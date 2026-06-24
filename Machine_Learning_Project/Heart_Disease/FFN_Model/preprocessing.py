import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset , DataLoader
import torch

from data_loader import load_data
data=load_data()

def preprocess_data(data=data , random_state=101 , batch_size=16):
    # preprocessing
    # Features & Target
    x=data.drop('DISEASE' ,axis=1)
    data['DISEASE']=(data['DISEASE']>0).astype(int)
    y=data['DISEASE']
    

    train_data , temp_data , train_label , temp_label=train_test_split(x,y , test_size=.2 , random_state=101 , stratify=y)
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


   # convert to torch.tensor
    train_data=torch.tensor(train_data , dtype=torch.float32)
    val_data=torch.tensor(val_data , dtype=torch.float32)
    test_data=torch.tensor(test_data , dtype=torch.float32)

    train_label=torch.tensor(train_label.values , dtype=torch.long)
    val_label=torch.tensor(val_label.values , dtype=torch.long)
    test_label=torch.tensor(test_label.values , dtype=torch.long)

    # convert to TensorDataset
    train_dataset=TensorDataset(train_data , train_label)
    val_dataset=TensorDataset(val_data , val_label)
    test_dataset=TensorDataset(test_data , test_label)

    # batching of data
    train_loader=DataLoader(train_dataset , batch_size=16 , drop_last=True , shuffle=True)
    val_loader=DataLoader(val_dataset , batch_size=16 , shuffle=False , drop_last=False)
    test_loader=DataLoader(test_dataset , batch_size=len(test_dataset))

    return train_loader , val_loader ,  test_loader , preprocesser