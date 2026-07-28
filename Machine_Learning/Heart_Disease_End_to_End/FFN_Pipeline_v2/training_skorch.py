from build_model import HeartFFN

from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
import torch.nn as nn
import torch


def training(input_dim , preprocessor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net=NeuralNetClassifier(module=HeartFFN ,
                            module__input_dim=input_dim , 
                            max_epochs=50,
                            lr=0.001,

                            optimizer=torch.optim.Adam,
                            criterion=nn.CrossEntropyLoss,

                            batch_size=16 ,
                            train_split=False,
                            verbose=0,
                            device=device)
    
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", net) ])

    return pipeline
