import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fastapi import FastAPI
from pydantic import BaseModel , Field
import joblib

from build_model import build_model


app=FastAPI(title='Heart Disease Prediction API',
            description='Pytorch FFN Model For Heart Disease Prediction',
            version='1.0.0')

# ------------------
# Load Artifacts
# ------------------

preprocessor=joblib.load('preprocessor.pkl')
threshold=joblib.load('threshold.pkl')
model , _ , _ = build_model(input_dim=22)
model.load_state_dict(torch.load('best_model.pt' ,
                                 map_location='cpu'))


model.eval()

# ------------------
# Request Schema
# ------------------    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    result: str

    
class PatientData(BaseModel):
    age: float = Field(..., description="Patient age in years")
    
    sex: int = Field(..., description="Gender: 0=female, 1=male")
    
    cp: int = Field(..., description="Chest pain type: 1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic")
    
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)")
    
    chol: float = Field(..., description="Serum cholesterol level (mg/dl)")
    
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl: 1=true, 0=false")
    
    restecg: int = Field(..., description="Resting ECG results: 0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy")
    
    thalach: float = Field(..., description="Maximum heart rate achieved")
    
    exang: int = Field(..., description="Exercise induced angina: 1=yes, 0=no")
    
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    
    slope: int = Field(..., description="Slope of peak exercise ST segment: 1=upsloping, 2=flat, 3=downsloping")
    
    thal: int = Field(..., description="Thalassemia status: 3=normal, 6=fixed defect, 7=reversible defect")
    
    ca: int = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")

# ------------------
# Health Check
# ------------------

@app.get('/health')
def health():
    return {'status':'Healthy'}

# ------------------
# Prediction Endpoint
# ------------------

@app.post('/predict' , response_model=PredictionResponse)
def predict(data : PatientData):
    df=pd.DataFrame([data.model_dump()])
    
    x=preprocessor.transform(df)
    x=torch.tensor(x , dtype=torch.float32)

    with torch.no_grad():
        logits=model(x)
        probability=F.softmax(logits , dim=1)[:,1].item()
        prediction= int(probability>=threshold)

        return PredictionResponse(prediction=prediction,
                                  probability=round(probability, 3),
                                  threshold=round(float(threshold), 3),
                                  result=("Heart Disease" if prediction == 1 else "No Heart Disease"))