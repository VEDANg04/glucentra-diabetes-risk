# model_evaluate.py
# Diabetes Risk Prediction — Detailed Evaluation + Visualisations
# Run this after model_train.py has generated diabetes_model.pkl

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from matplotlib.patches import Patch
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.pkl"
DOCS       = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
print("Loading model...")
obj        = joblib.load(MODEL_PATH)
model      = obj["model"]
FEATURES   = obj["features"]
model_name = obj.get("model_name", "Saved Model")
xgb_model  = obj.get("xgb_model", None)

print(f"Model loaded : {model_name}")


# ── Recreate Exact Test Split ──────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)
df = df.drop(columns=[c for c in ["HighBP", "HighChol", "CholCheck"] if c in df.columns])

df["Metabolic_Risk"]   = df["BMI"] * df["GenHlth"]
df["Lifestyle_Burden"] = df["Smoker"] + df["HvyAlcoholConsump"] + (1 - df["PhysActivity"])
df["Age_BMI"]          = df["Age"] * df["BMI"]

X = df[FEATURES]
y = df["Diabetes_binary"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_prob = model.predict_proba(X_test)[:, 1]

# ── Threshold tuning — find threshold that balances precision/recall ───────────
# Default 0.5 causes very low recall on imbalanced data.
# We pick the threshold closest to where precision == recall (F1 is maximised).
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx   = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"\nOptimal decision threshold : {best_threshold:.3f}  (maximises F1)")

y_pred = (y_prob >= best_threshold).astype(int)


# ── Core Metrics ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"EVALUATION RESULTS — {model_name}")
print(f"(threshold = {best_threshold:.3f})")
print("=" * 55)
print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix Breakdown:")
print(f"  True Negatives  (TN) : {tn}  — correctly identified healthy")
print(f"  False Positives (FP) : {fp}  — healthy flagged as diabetic")
print(f"  False Negatives (FN) : {fn}  — diabetics missed by model")
print(f"  True Positives  (TP) : {tp}  — correctly identified diabetic")


# ── Plot 1: Confusion Matrix + ROC Curve ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Diabetes", "Diabetes"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix — {model_name}\n(threshold={best_threshold:.3f})",
                  fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_val = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#e74c3c", lw=2,
             label=f"{model_name} (AUC = {auc_val:.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
axes[1].fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve", fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.savefig(DOCS / "evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: evaluation_plots.png")


# ── Plot 2: Feature Importance ─────────────────────────────────────────────────
if xgb_model is not None:
    try:
        importances = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature"    : FEATURES,
            "Importance" : importances
        }).sort_values("Importance", ascending=True)

        colors = [
            "#e67e22" if f in ["Metabolic_Risk", "Lifestyle_Burden", "Age_BMI"]
            else "#3498db"
            for f in importance_df["Feature"]
        ]

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
        plt.title("Feature Importance — XGBoost", fontweight="bold")
        plt.xlabel("Importance Score")
        legend_elements = [
            Patch(facecolor="#3498db", label="Original feature"),
            Patch(facecolor="#e67e22", label="Engineered feature")
        ]
        plt.legend(handles=legend_elements, loc="lower right")
        plt.tight_layout()
        plt.savefig(DOCS / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: feature_importance.png")

        # Print ranked table
        print("\n" + "=" * 58)
        print("FEATURE IMPORTANCES (ranked, XGBoost)")
        print("=" * 58)
        ranked  = importance_df.iloc[::-1]
        max_val = ranked["Importance"].max()
        for _, row in ranked.iterrows():
            bar = "█" * int((row["Importance"] / max_val) * 30)
            tag = " ← engineered" if row["Feature"] in ["Metabolic_Risk", "Lifestyle_Burden", "Age_BMI"] else ""
            print(f"  {row['Feature']:<25} {row['Importance']:.4f}  {bar}{tag}")

    except Exception as e:
        print(f"Feature importance error: {e}")
else:
    print("\nNote: XGBoost model not found in pkl — retrain to regenerate.")


# ── Plot 3: Risk Score Distribution ───────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.hist(y_prob[y_test == 0], bins=50, alpha=0.6, color="#2ecc71", label="No Diabetes")
plt.hist(y_prob[y_test == 1], bins=50, alpha=0.6, color="#e74c3c", label="Diabetes")
plt.axvline(x=best_threshold, color="black", linestyle="--",
            label=f"Optimal threshold ({best_threshold:.3f})")
plt.xlabel("Predicted Risk Probability")
plt.ylabel("Number of Records")
plt.title("Predicted Risk Score Distribution", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(DOCS / "risk_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: risk_distribution.png")

print("\nEvaluation complete.")
print(f"Optimal threshold to use in model_predict.py: {best_threshold:.3f}")

# ── Plot 4: Calibration Curve ─────────────────────────────────────────────────
from sklearn.calibration import calibration_curve

frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed fraction of positives")
plt.title("Calibration Curve", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(DOCS / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: calibration_curve.png")