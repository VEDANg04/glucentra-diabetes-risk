import sys
from pathlib import Path

import pytest

# Make backend/ importable when pytest runs from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

try:
    from model_predict import predict_diabetes_risk
except FileNotFoundError:
    pytest.skip("Model not trained yet: run backend/model_train.py",
                allow_module_level=True)

HIGH = {
    "BMI": 35.0, "Smoker": 1, "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
    "HvyAlcoholConsump": 0, "GenHlth": 5, "MentHlth": 10, "PhysHlth": 15,
    "DiffWalk": 1, "Sex": 1, "Age": 10, "Education": 3, "Income": 2,
    "Stroke": 0, "HeartDiseaseorAttack": 1, "hereditary_category": 0,
}

LOW = {
    "BMI": 22.0, "Smoker": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0,
    "DiffWalk": 0, "Sex": 0, "Age": 4, "Education": 6, "Income": 7,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "hereditary_category": 0,
}


def test_high_risk_scores_above_low_risk():
    high = predict_diabetes_risk(HIGH)["final_risk"]
    low = predict_diabetes_risk(LOW)["final_risk"]
    assert high > low


def test_family_history_raises_risk():
    base = predict_diabetes_risk(LOW)["final_risk"]
    both_parents = predict_diabetes_risk({**LOW, "hereditary_category": 2})["final_risk"]
    assert both_parents > base


def test_no_family_history_leaves_risk_unchanged():
    r = predict_diabetes_risk(LOW)
    assert r["final_risk"] == pytest.approx(r["lifestyle_risk"], abs=0.1)


def test_risk_stays_within_bounds():
    r = predict_diabetes_risk({**HIGH, "hereditary_category": 2})
    assert 0 <= r["final_risk"] <= 99


def test_multiplier_is_returned():
    r = predict_diabetes_risk({**LOW, "hereditary_category": 1})
    assert r["hereditary_multiplier"] == 1.90