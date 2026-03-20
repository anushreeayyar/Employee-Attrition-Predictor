"""
Employee Attrition Predictor — Model Training Pipeline
=======================================================
Trains an XGBoost classifier + a Random Forest for comparison.
Generates SHAP explainability values and saves all artefacts.

Usage:
    python src/train_model.py

Outputs:
    models/attrition_model.pkl      — trained XGBoost pipeline
    models/feature_importance.csv   — SHAP-based feature importance
    models/model_metrics.json       — accuracy, AUC, F1, precision, recall
    models/shap_values.npy          — SHAP matrix (for dashboard)
    models/X_test.csv               — held-out features (for dashboard)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
)
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "employee_data.csv")
MODELS_DIR  = "models"
PLOTS_DIR   = os.path.join("models", "plots")
SEED        = 42
TEST_SIZE   = 0.20

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────────
DROP_COLS = [
    "employee_id", "full_name", "email", "address", "supervisor_name",
    "supervisor_email", "talent_business_partner", "business_title",
    "cost_center", "successor_name", "exit_reason", "termination_date",
    "attrition", "attrition_risk_score", "current_start_date",
    "last_day_working", "paid_or_unpaid_leave", "job_code",
]

CATEGORICAL_COLS = [
    "gender", "marital_status", "state", "business_unit", "division",
    "department", "cost_center_name", "job_level", "performance_review",
    "overtime",
]

NUMERIC_COLS = [
    "age", "distance_from_home_miles", "commute_time_minutes",
    "num_direct_reports", "team_size", "tenure_years", "years_at_company",
    "salary", "bonus_percentage", "stock_options", "last_raise_percentage",
    "remote_work_percentage", "monthly_hours_worked", "num_companies_before",
    "num_promotions", "years_since_last_promotion", "training_hours_last_year",
    "engagement_score", "job_satisfaction_score", "work_life_balance_score",
    "relationship_satisfaction", "learning_score", "nine_box_talent_readiness",
    "healthcare_satisfaction",
]

BOOL_COLS = ["is_manager", "has_successor"]


def load_and_prepare(path: str):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Target
    y = (df["attrition"] == "Yes").astype(int)

    # Features
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()

    # Booleans → int
    for c in BOOL_COLS:
        if c in X.columns:
            X[c] = X[c].astype(int)

    return X, y, feature_cols


def build_preprocessor(X: pd.DataFrame):
    cat_cols  = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols  = [c for c in NUMERIC_COLS    if c in X.columns]
    bool_cols = [c for c in BOOL_COLS       if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(),                                   num_cols + bool_cols),
            ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor, cat_cols, num_cols, bool_cols


def train(X_tr, y_tr, preprocessor):
    xgb_model = xgb.XGBClassifier(
        n_estimators     = 400,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum(),
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = SEED,
        n_jobs           = -1,
    )
    pipeline = Pipeline([
        ("prep",  preprocessor),
        ("model", xgb_model),
    ])
    print("Training XGBoost …")
    pipeline.fit(X_tr, y_tr)
    return pipeline


def evaluate(pipeline, X_te, y_te):
    y_pred  = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_te, y_proba), 4),
        "f1":        round(f1_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall":    round(recall_score(y_te, y_pred), 4),
    }
    cm = confusion_matrix(y_te, y_pred)

    print("\n── Model Metrics ─────────────────────────────────")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=["Stayed","Left"]))
    return metrics, y_proba


def compute_shap(pipeline, X_te, X_tr):
    print("\nComputing SHAP values …")
    prep     = pipeline.named_steps["prep"]
    model    = pipeline.named_steps["model"]
    X_te_t   = prep.transform(X_te)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te_t)

    # Feature names after encoding
    feature_names = list(prep.get_feature_names_out())

    # Mean |SHAP| importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "mean_shap":  mean_shap,
    }).sort_values("mean_shap", ascending=False)

    print("\nTop 15 attrition drivers (SHAP):")
    print(importance_df.head(15).to_string(index=False))

    # Save SHAP summary plot
    shap.summary_plot(shap_values, X_te_t, feature_names=feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved → {PLOTS_DIR}/shap_summary.png")

    return shap_values, importance_df, feature_names


def main():
    print("=" * 60)
    print("  Employee Attrition Predictor — Training Pipeline")
    print("=" * 60)

    # 1. Load
    X, y, feature_cols = load_and_prepare(DATA_PATH)
    print(f"Class distribution  → Stayed:{(y==0).sum():,}  Left:{(y==1).sum():,}")

    # 2. Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 3. Preprocessor
    preprocessor, cat_cols, num_cols, bool_cols = build_preprocessor(X_tr)

    # 4. Train
    pipeline = train(X_tr, y_tr, preprocessor)

    # 5. Evaluate
    metrics, y_proba = evaluate(pipeline, X_te, y_te)

    # 6. SHAP
    shap_values, importance_df, feature_names = compute_shap(pipeline, X_te, X_tr)

    # 7. Save artefacts
    with open(os.path.join(MODELS_DIR, "attrition_model.pkl"), "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved → {MODELS_DIR}/attrition_model.pkl")

    importance_df.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)

    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(MODELS_DIR, "shap_values.npy"), shap_values)

    X_te.to_csv(os.path.join(MODELS_DIR, "X_test.csv"), index=False)

    # Save feature names for dashboard
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save test predictions for dashboard
    results = X_te.copy()
    results["attrition_actual"]     = y_te.values
    results["attrition_predicted"]  = pipeline.predict(X_te)
    results["flight_risk_score"]    = y_proba
    results.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)

    print("\n✅ All artefacts saved to ./models/")
    print(f"\n🎯 Final XGBoost AUC: {metrics['roc_auc']:.4f}  |  Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
