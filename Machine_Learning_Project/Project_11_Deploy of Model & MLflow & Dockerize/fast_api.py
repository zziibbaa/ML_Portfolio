from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Union
from joblib import load
import pandas as pd

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(
    title="ML Advertising Sales Predictor",
    description="Predict advertising sales using a trained regression model via FastAPI",
    version="1.0.0"
)

# -----------------------------
# Define input data model with Pydantic
# -----------------------------
class PredictionInput(BaseModel):
    TV: float = Field(..., ge=0, description="TV advertising budget (non-negative)")
    radio: float = Field(..., ge=0, description="Radio advertising budget (non-negative)")
    newspaper: float = Field(..., ge=0, description="Newspaper advertising budget (non-negative)")

# -----------------------------
# Load model and column names at startup
# -----------------------------
model = None
col_names = None

@app.on_event("startup")
def load_artifacts():
    global model, col_names
    model = load("final_model.pkl")      # Load the trained model
    col_names = load("column_name.pkl")  # Load expected input columns

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(
    data: Union[PredictionInput, List[PredictionInput]]
):
    """
    Predict sales for one or multiple input records.
    """
    # If single input is provided, convert it to a list
    if isinstance(data, PredictionInput):
        data = [data]
    
    # Convert input data to DataFrame
    df = pd.DataFrame([item.dict() for item in data])
    
    # Reindex columns to match model training columns, fill missing columns with 0
    df = df.reindex(columns=col_names, fill_value=0)
    
    # Get predictions from the model
    predictions = model.predict(df)
    
    # Return predictions as a list of floats
    return {"predictions": [float(p) for p in predictions]}
