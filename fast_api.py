from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List

# Create FastAPI app
app = FastAPI()

# Load the trained model from the file
model_pipeline = joblib.load('ml_pipeline.pkl')

class PredictionRequest(BaseModel):
    X: List[List[float]]  # Adjust the type according to your data structure

@app.post('/predict')
async def predict(request: PredictionRequest):
    # Convert request data to pandas DataFrame
    X_new = pd.DataFrame(request.X)
    
    # Normalize and make predictions
    predictions = model_pipeline.predict(X_new)
    
    # Send predictions back as JSON
    return {'predictions': predictions.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)