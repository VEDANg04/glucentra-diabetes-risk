# model_train.py
# Diabetes Risk Prediction — Training Pipeline
# Dataset  : BRFSS 2015 (diabetes_012_health_indicators_BRFSS2015.csv)
# Approach : Stacking Ensemble (XGBoost + LightGBM + RandomForest + LR meta)
#            Hereditary risk is applied at prediction time only (not trained on fake data)

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.pkl"

np.random.seed(42)

# ── STEP 1: Load Dataset ───────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Binarise target: 0 = no diabetes, 1 = pre-diabetes or diabetes
df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)

print(f"Total records : {len(df)}")
print(f"No Diabetes   : {sum(df['Diabetes_binary'] == 0)}")
print(f"Diabetes      : {sum(df['Diabetes_binary'] == 1)}")


# ── STEP 2: Remove Clinical Features ──────────────────────────────────────────
# These require a prior doctor visit — not available from lifestyle survey alone
clinical_features = ["HighBP", "HighChol", "CholCheck"]
df = df.drop(columns=[c for c in clinical_features if c in df.columns])
print(f"\nClinical features removed: {clinical_features}")


# ── STEP 3: Feature Engineering ───────────────────────────────────────────────
# Interaction features based on domain knowledge

# Metabolic Risk: high BMI + poor general health = compounding effect
df["Metabolic_Risk"] = df["BMI"] * df["GenHlth"]

# Lifestyle Burden: smoking + heavy alcohol + no exercise
df["Lifestyle_Burden"] = df["Smoker"] + df["HvyAlcoholConsump"] + (1 - df["PhysActivity"])

# Age × BMI: older age amplifies BMI-related risk
df["Age_BMI"] = df["Age"] * df["BMI"]

print("Interaction features added: Metabolic_Risk, Lifestyle_Burden, Age_BMI")


# ── STEP 4: Define Features ────────────────────────────────────────────────────
# NOTE: Hereditary_Multiplier is intentionally excluded from training.
# The dataset has no real family history data — assigning random values would
# teach the model noise, not signal. Hereditary risk is applied at prediction
# time as a literature-based multiplier (see model_predict.py).

FEATURES = [
    "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income",
    "Metabolic_Risk", "Lifestyle_Burden", "Age_BMI"
]

X = df[FEATURES]
y = df["Diabetes_binary"]

print(f"\nFeatures used : {len(FEATURES)}")
print(f"Feature list  : {FEATURES}")


# ── STEP 5: Train/Test Split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# Class imbalance ratio — used for XGBoost/LightGBM scale_pos_weight
neg = sum(y_train == 0)
pos = sum(y_train == 1)
scale = round(neg / pos, 2)
print(f"Class imbalance ratio (neg/pos): {scale} — used for XGB/LGBM")


# ── STEP 6: Train Individual Models ───────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_and_evaluate(name, model):
    print(f"\nTraining {name}...")
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC (5-fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model"    : model,
        "accuracy" : accuracy_score(y_test, y_pred),
        "auc"      : roc_auc_score(y_test, y_prob),
        "f1"       : f1_score(y_test, y_pred),
        "cv_auc"   : cv_auc.mean()
    }
    print(f"  Test Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  Test ROC-AUC  : {results[name]['auc']:.4f}")
    print(f"  Test F1       : {results[name]['f1']:.4f}")
    return model


lr = train_and_evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
)

rf = train_and_evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=10,
                           class_weight="balanced", random_state=42, n_jobs=-1)
)

xgb = train_and_evaluate(
    "XGBoost",
    XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=scale,          # handles class imbalance
                  eval_metric="logloss", random_state=42, n_jobs=-1)
)

lgbm = train_and_evaluate(
    "LightGBM",
    LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                   is_unbalance=True,              # handles class imbalance
                   random_state=42, n_jobs=-1, verbose=-1)
)


# ── STEP 7: Train Stacking Ensemble ───────────────────────────────────────────
print("\nTraining Stacking Ensemble (this takes a few minutes)...")

base_learners = [
    ("xgb",  XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale,
                            eval_metric="logloss", random_state=42, n_jobs=-1)),
    ("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                             is_unbalance=True,
                             random_state=42, n_jobs=-1, verbose=-1)),
    ("rf",   RandomForestClassifier(n_estimators=200, max_depth=10,
                                    class_weight="balanced",
                                    random_state=42, n_jobs=-1))
]

stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

stacking.fit(X_train, y_train)
stack_pred = stacking.predict(X_test)
stack_prob = stacking.predict_proba(X_test)[:, 1]

results["Stacking Ensemble"] = {
    "model"    : stacking,
    "accuracy" : accuracy_score(y_test, stack_pred),
    "auc"      : roc_auc_score(y_test, stack_prob),
    "f1"       : f1_score(y_test, stack_pred),
    "cv_auc"   : 0  # stacking CV is done internally
}

print(f"  Accuracy  : {results['Stacking Ensemble']['accuracy']:.4f}")
print(f"  ROC-AUC   : {results['Stacking Ensemble']['auc']:.4f}")
print(f"  F1 Score  : {results['Stacking Ensemble']['f1']:.4f}")


# ── STEP 8: Model Comparison Table ────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Model':<25} {'CV AUC':>10} {'Test Acc':>10} {'Test AUC':>10} {'F1':>8}")
print("=" * 65)
for name, r in results.items():
    cv_str = f"{r['cv_auc']:.4f}" if r["cv_auc"] > 0 else "  (internal)"
    marker = " ★" if name == "Stacking Ensemble" else ""
    print(f"{name:<25} {cv_str:>10} {r['accuracy']:>10.4f} {r['auc']:>10.4f} {r['f1']:>8.4f}{marker}")
print("=" * 65)


# ── STEP 9: Save Model & Artifacts ────────────────────────────────────────────
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model"      : stacking,
    "features"   : FEATURES,
    "model_name" : "Stacking Ensemble",
    "xgb_model"  : xgb
}, MODEL_PATH)

print(f"\nSaved: {MODEL_PATH}")
print("Run model_evaluate.py next to see detailed evaluation and plots.")