## conda install -c conda-forge fastapi

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from joblib import dump , load
import pandas as pd


# create fastapi app
app=FastAPI()

# define the input data model
class PredictionInput(BaseModel):
    TV : float
    radio:float
    newspaper : float


# load model , columns at startup

model= load('final_model.pkl')
col_names = load('column_name.pkl')


@app.post('/predict')
async def predict(data : List[PredictionInput]):

    # convert input to DataFrame
    df = pd.DataFrame([item.dict() for item in data])

    # match column names
    df = df.reindex(columns= col_names)

    # get predictions
    prediction= list(model.predict(df))

    return {'prediction' : str(prediction)} 
    
    