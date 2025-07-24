from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from src.predict_utils import predict_from_dict
import uvicorn


class HeartFeatures(BaseModel):
    age: float
    sex: int
    trestbps: float
    chol: float
    fbs: int
    thalach: float
    exang: int
    oldpeak: float
    ca: int
    cp: int
    restecg: int
    slope: int
    thal: int


app = FastAPI(title="Heart Disease Predictor")


@app.post("/predict")
def predict(data: HeartFeatures):
    try:
        result = predict_from_dict(data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"has_disease": result}


@app.get("/report")
def get_report():
    report_path = os.path.join(os.path.dirname(__file__), '..', 'report.html')
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(report_path, media_type="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
