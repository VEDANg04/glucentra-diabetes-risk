# model_predict.py
# Diabetes Risk Prediction — Prediction Function
# Used by the UI / app to get risk scores for a user's inputs.
#
# Hereditary Risk Multipliers (from published literature):
# ─────────────────────────────────────────────────────────
# | Category               | Relative Risk | Source                              |
# |------------------------|---------------|-------------------------------------|
# | No family history      | 1.00          | Baseline                            |
# | One parent             | 1.90          | CDC/NHANES 2005 (avg father + mother)|
# | Both parents           | 3.40          | Bisquera et al., ScienceDirect 2019 |
# | Sibling                | 2.77          | Shanghai High-Risk Diabetic Study   |
# | Grandparent/Uncle/Aunt | 1.10          | PMC Family History Study 2024       |

import numpy as np
import pandas as pd
import joblib

# ── Load Model ─────────────────────────────────────────────────────────────
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "diabetes_model.pkl"
_obj     = joblib.load(MODEL_PATH)
_model   = _obj["model"]
FEATURES = _obj["features"]

# ── Hereditary Risk Configuration ─────────────────────────────────────────────
HEREDITARY_CATEGORIES = {
    0: {"label": "No family history",       "multiplier": 1.00},
    1: {"label": "One parent",              "multiplier": 1.90},
    2: {"label": "Both parents",            "multiplier": 3.40},
    3: {"label": "Sibling",                 "multiplier": 2.77},
    4: {"label": "Grandparent/Uncle/Aunt",  "multiplier": 1.10},
}


def predict_diabetes_risk(user_input: dict) -> dict:
    """
    Predict diabetes risk for a single user.

    Parameters
    ----------
    user_input : dict
        Keys:
          BMI, Smoker, Stroke, HeartDiseaseorAttack,
          PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
          GenHlth, MentHlth, PhysHlth, DiffWalk,
          Sex, Age, Education, Income,
          hereditary_category (0–4, default 0)

    Returns
    -------
    dict with:
        lifestyle_risk   — model output before hereditary adjustment (0–100%)
        final_risk       — after hereditary adjustment (0–100%)
        risk_label       — "Low" / "Moderate" / "High"
        hereditary_label — which hereditary category was used
        suggestions      — list of personalised tip strings
    """

    # ── Extract inputs ─────────────────────────────────────────────────────────
    bmi         = float(user_input["BMI"])
    gen_hlth    = int(user_input["GenHlth"])
    smoker      = int(user_input["Smoker"])
    hvy_alcohol = int(user_input["HvyAlcoholConsump"])
    phys_act    = int(user_input["PhysActivity"])
    age         = int(user_input["Age"])

    hereditary_cat  = int(user_input.get("hereditary_category", 0))
    hereditary_info = HEREDITARY_CATEGORIES.get(hereditary_cat, HEREDITARY_CATEGORIES[0])
    multiplier      = hereditary_info["multiplier"]

    # ── Build Feature Vector ───────────────────────────────────────────────────
    feature_vector = {
        "BMI"                  : bmi,
        "Smoker"               : smoker,
        "Stroke"               : int(user_input.get("Stroke", 0)),
        "HeartDiseaseorAttack" : int(user_input.get("HeartDiseaseorAttack", 0)),
        "PhysActivity"         : phys_act,
        "Fruits"               : int(user_input.get("Fruits", 0)),
        "Veggies"              : int(user_input.get("Veggies", 0)),
        "HvyAlcoholConsump"    : hvy_alcohol,
        "GenHlth"              : gen_hlth,
        "MentHlth"             : int(user_input.get("MentHlth", 0)),
        "PhysHlth"             : int(user_input.get("PhysHlth", 0)),
        "DiffWalk"             : int(user_input.get("DiffWalk", 0)),
        "Sex"                  : int(user_input.get("Sex", 0)),
        "Age"                  : age,
        "Education"            : int(user_input.get("Education", 4)),
        "Income"               : int(user_input.get("Income", 5)),
        # Engineered features — must match training exactly
        "Metabolic_Risk"       : bmi * gen_hlth,
        "Lifestyle_Burden"     : smoker + hvy_alcohol + (1 - phys_act),
        "Age_BMI"              : age * bmi,
    }

    input_df = pd.DataFrame([feature_vector])[FEATURES]

    # ── Lifestyle Risk (model output) ──────────────────────────────────────────
    lifestyle_prob = float(_model.predict_proba(input_df)[0][1])

    # ── Apply Hereditary Multiplier ────────────────────────────────────────────
    # Convert probability to odds, scale by multiplier, convert back.
    # This prevents overflow (probability always stays in 0–1 range).
    if lifestyle_prob >= 1.0:
        final_prob = 0.99
    else:
        odds         = lifestyle_prob / (1.0 - lifestyle_prob)
        adjusted_odds = odds * multiplier
        final_prob   = adjusted_odds / (1.0 + adjusted_odds)
        final_prob   = min(final_prob, 0.99)

    # ── Risk Label ─────────────────────────────────────────────────────────────
    # Illustrative risk bands for display only. They are not clinically validated
    # cutoffs and are independent of the F1-optimal threshold used in evaluation.

    if final_prob >= 0.50:
        risk_label = "High"
    elif final_prob >= 0.25:
        risk_label = "Moderate"
    else:
        risk_label = "Low"

    # ── Personalised Suggestions ───────────────────────────────────────────────
    suggestions = _generate_suggestions(user_input, bmi, final_prob)

    return {
        "lifestyle_risk"       : round(lifestyle_prob * 100, 1),
        "final_risk"           : round(final_prob * 100, 1),
        "risk_label"           : risk_label,
        "hereditary_label"     : hereditary_info["label"],
        "hereditary_multiplier": multiplier,
        "suggestions"          : suggestions,
    }


def _generate_suggestions(user_input: dict, bmi: float, risk: float) -> list:
    """
    Generate personalised suggestions based on the user's specific inputs.
    Returns a list of suggestion strings (max 5).
    """
    tips = []

    if bmi >= 30:
        tips.append(
            f"Your BMI is {bmi:.1f} (obese range). Even a 5–10% reduction in body "
            "weight can significantly lower diabetes risk."
        )
    elif bmi >= 25:
        tips.append(
            f"Your BMI is {bmi:.1f} (overweight range). Moving towards a BMI under "
            "25 through diet and activity can reduce your risk."
        )

    if int(user_input.get("PhysActivity", 1)) == 0:
        tips.append(
            "You reported no physical activity in the past 30 days. "
            "Even 30 minutes of walking 5 days a week can lower diabetes risk by up to 30%."
        )

    if int(user_input.get("Smoker", 0)) == 1:
        tips.append(
            "Smokers are 30–40% more likely to develop type 2 diabetes. "
            "Quitting smoking is one of the highest-impact lifestyle changes you can make."
        )

    if int(user_input.get("HvyAlcoholConsump", 0)) == 1:
        tips.append(
            "Heavy alcohol consumption affects insulin sensitivity. "
            "Reducing alcohol intake can improve blood sugar regulation."
        )

    gen_hlth = int(user_input.get("GenHlth", 3))
    if gen_hlth >= 4:
        tips.append(
            "You rated your general health as Fair or Poor. "
            "Consider scheduling a health check-up — early detection of pre-diabetes is reversible."
        )

    if int(user_input.get("Fruits", 1)) == 0 and int(user_input.get("Veggies", 1)) == 0:
        tips.append(
            "Including fruits and vegetables daily supports blood sugar stability "
            "and overall metabolic health."
        )

    if int(user_input.get("HeartDiseaseorAttack", 0)) == 1:
        tips.append(
            "A history of heart disease increases diabetes risk. "
            "Regular monitoring of blood glucose levels is strongly advised."
        )

    if risk < 0.25 and not tips:
        tips.append(
            "Your risk is currently low. Keep up the healthy lifestyle — "
            "regular activity and a balanced diet are your best long-term protection."
        )

    return tips[:5]   # cap at 5 suggestions


# ── Test Cases ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("TEST 1: High risk person")
    print("=" * 55)
    test1 = {
        "BMI": 35.0, "Smoker": 1, "PhysActivity": 0,
        "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0,
        "GenHlth": 5, "MentHlth": 10, "PhysHlth": 15,
        "DiffWalk": 1, "Sex": 1, "Age": 10,
        "Education": 3, "Income": 2,
        "Stroke": 0, "HeartDiseaseorAttack": 1,
        "hereditary_category": 2   # both parents
    }
    r1 = predict_diabetes_risk(test1)
    print(f"Lifestyle Risk : {r1['lifestyle_risk']}%")
    print(f"Final Risk     : {r1['final_risk']}%  ({r1['risk_label']})")
    print(f"Hereditary     : {r1['hereditary_label']}")
    print("Suggestions:")
    for s in r1["suggestions"]:
        print(f"  • {s}")

    print("\n" + "=" * 55)
    print("TEST 2: Low risk person")
    print("=" * 55)
    test2 = {
        "BMI": 22.0, "Smoker": 0, "PhysActivity": 1,
        "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0,
        "DiffWalk": 0, "Sex": 0, "Age": 4,
        "Education": 6, "Income": 7,
        "Stroke": 0, "HeartDiseaseorAttack": 0,
        "hereditary_category": 0   # no family history
    }
    r2 = predict_diabetes_risk(test2)
    print(f"Lifestyle Risk : {r2['lifestyle_risk']}%")
    print(f"Final Risk     : {r2['final_risk']}%  ({r2['risk_label']})")
    print(f"Hereditary     : {r2['hereditary_label']}")
    print("Suggestions:")
    for s in r2["suggestions"]:
        print(f"  • {s}")
