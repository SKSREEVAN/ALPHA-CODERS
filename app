# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Create FastAPI app at the global level
app = FastAPI(title="PulseGuard AI Server")

# 2️⃣ Sample data
data = pd.DataFrame({
    "age":[45,52,38,60,40,55],
    "bmi":[26,30,22,32,24,29],
    "systolic":[130,150,118,170,120,145],
    "diastolic":[85,95,78,110,80,90],
    "cholesterol":[200,240,180,260,190,220],
    "glucose":[110,130,95,140,100,120],
    "smoking":[0,1,0,1,0,1],
    "alcohol":[0,1,0,1,0,0],
    "activity":[1,0,1,0,1,0],
    "hypertension_stage":[1,2,0,3,0,2]
})

X = data.drop("hypertension_stage", axis=1)
y = data["hypertension_stage"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

stage_labels = {0:"Normal", 1:"Prehypertension", 2:"Stage 1 Hypertension", 3:"Stage 2 Hypertension"}

# 3️⃣ Input model
class PatientData(BaseModel):
    age:int
    bmi:float
    systolic:int
    diastolic:int
    cholesterol:int
    glucose:int
    smoking:int
    alcohol:int
    activity:int

# 4️⃣ Root endpoint
@app.get("/")
def read_root():
    return {"message":"Welcome to PulseGuard AI API"}

# 5️⃣ Prediction endpoint
@app.post("/predict")
def predict(data:PatientData):
    input_df = pd.DataFrame([data.dict()])
    input_scaled = scaler.transform(input_df)
    pred_class = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    return {
        "predicted_stage": stage_labels.get(pred_class,"Unknown"),
        "probabilities": {stage_labels[i]: float(pred_proba[i]) for i in range(len(pred_proba))}
    }

# 6️⃣ Uvicorn run for Mac
if __name__=="__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
