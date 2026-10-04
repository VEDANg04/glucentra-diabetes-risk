# api.py
# Glucentra — FastAPI Backend
# Connects the static frontend to the diabetes prediction model.
#
# Run with:
#   pip install fastapi uvicorn
#   uvicorn api:app --reload --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import traceback

# Import prediction function from model_predict.py (same folder)
from model_predict import predict_diabetes_risk

app = FastAPI(
    title="Glucentra Diabetes Risk API",
    description="Predicts diabetes/prediabetes risk from lifestyle and hereditary inputs.",
    version="1.0.0"
)

# Allow the static frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],  # local dev only
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Request schema ─────────────────────────────────────────────────────────────
# FamilyHistory comes in as a string from the frontend dropdown
FAMILY_HISTORY_MAP = {
    "none"        : 0,
    "one_parent"  : 1,
    "both_parents": 2,
    "sibling"     : 3,
    "extended"    : 4,
}

class PredictRequest(BaseModel):
    BMI                  : float = Field(..., ge=10, le=80,  description="Body mass index")
    Smoker               : int   = Field(..., ge=0,  le=1)
    Stroke               : int   = Field(..., ge=0,  le=1)
    HeartDiseaseorAttack : int   = Field(..., ge=0,  le=1)
    PhysActivity         : int   = Field(..., ge=0,  le=1)
    Fruits               : int   = Field(..., ge=0,  le=1)
    Veggies              : int   = Field(..., ge=0,  le=1)
    HvyAlcoholConsump    : int   = Field(..., ge=0,  le=1)
    GenHlth              : int   = Field(..., ge=1,  le=5)
    MentHlth             : int   = Field(..., ge=0,  le=30)
    PhysHlth             : int   = Field(..., ge=0,  le=30)
    DiffWalk             : int   = Field(..., ge=0,  le=1)
    Sex                  : int   = Field(..., ge=0,  le=1)
    Age                  : int   = Field(..., ge=1,  le=13)
    Education            : int   = Field(..., ge=1,  le=6)
    Income               : int   = Field(..., ge=1,  le=8)
    FamilyHistory        : Literal["none", "one_parent", "both_parents", "sibling", "extended"] = "none"


# ── Response schema ────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    lifestyle_risk   : float        # 0–100, model output before hereditary
    combined_risk    : float        # 0–100, after hereditary multiplier
    final_risk       : float        # same as combined_risk (alias for frontend gauge)
    risk_label       : str          # "Low" / "Moderate" / "High"
    risk_level       : int          # 1 / 2 / 3 (used by frontend colour coding)
    hereditary_mult  : float        # multiplier applied (e.g. 1.90)
    hereditary_label : str          # human-readable label
    suggestions      : list[str]    # personalised tips


# ── POST /predict ──────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Convert FamilyHistory string → integer category
        hereditary_cat = FAMILY_HISTORY_MAP.get(req.FamilyHistory, 0)

        # Build input dict for model_predict.py
        user_input = {
            "BMI"                  : req.BMI,
            "Smoker"               : req.Smoker,
            "Stroke"               : req.Stroke,
            "HeartDiseaseorAttack" : req.HeartDiseaseorAttack,
            "PhysActivity"         : req.PhysActivity,
            "Fruits"               : req.Fruits,
            "Veggies"              : req.Veggies,
            "HvyAlcoholConsump"    : req.HvyAlcoholConsump,
            "GenHlth"              : req.GenHlth,
            "MentHlth"             : req.MentHlth,
            "PhysHlth"             : req.PhysHlth,
            "DiffWalk"             : req.DiffWalk,
            "Sex"                  : req.Sex,
            "Age"                  : req.Age,
            "Education"            : req.Education,
            "Income"               : req.Income,
            "hereditary_category"  : hereditary_cat,
        }

        result = predict_diabetes_risk(user_input)

        # Map risk_label → numeric level for frontend colour coding
        label_to_level = {"Low": 1, "Moderate": 2, "High": 3}
        risk_level = label_to_level.get(result["risk_label"], 1)

        return PredictResponse(
            lifestyle_risk   = result["lifestyle_risk"],
            combined_risk    = result["final_risk"],
            final_risk       = result["final_risk"],
            risk_label       = result["risk_label"],
            risk_level       = risk_level,
            hereditary_mult  = result["hereditary_multiplier"],
            hereditary_label = result["hereditary_label"],
            suggestions      = result["suggestions"],
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# ── GET /health ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "Stacking Ensemble — Glucentra v1"}
