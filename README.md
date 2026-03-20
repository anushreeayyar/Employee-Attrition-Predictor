# 🧠 Employee Attrition Predictor

> An end-to-end machine learning system that predicts employee flight risk, identifies the top attrition drivers, and delivers actionable retention insights through an interactive dashboard.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io)

---

## 🎯 What This Does

| Capability | Detail |
|---|---|
| **Predictive Model** | XGBoost classifier trained on 12,000 synthetic HR records with 54 features |
| **Explainability** | SHAP values reveal *why* an employee is at risk — not just a black-box score |
| **Flight Risk Dashboard** | Company-wide KPIs, attrition trends, and at-risk employee roster |
| **Segment Analysis** | Break down risk by department, job level, manager, overtime, and performance |
| **What-If Simulator** | Adjust salary, engagement, overtime, etc. and instantly re-score risk |
| **Model Performance** | ROC curve, confusion matrix, precision/recall vs. threshold |

---

## 📊 Dataset Overview

The synthetic dataset (`data/employee_data.csv`) contains **12,000 employee records** with **54 columns** across all major HR dimensions:

**Identity & Demographics**
`employee_id`, `full_name`, `email`, `gender`, `age`, `marital_status`, `address`, `state`

**Organisation**
`business_unit`, `division`, `department`, `cost_center`, `job_level`, `job_code`, `business_title`, `is_manager`, `num_direct_reports`, `team_size`

**Compensation**
`salary`, `bonus_percentage`, `stock_options`, `last_raise_percentage`

**Dates & Tenure**
`current_start_date`, `tenure_years`, `years_at_company`, `last_day_working`, `paid_or_unpaid_leave`

**Work Patterns**
`overtime`, `remote_work_percentage`, `monthly_hours_worked`, `num_companies_before`, `training_hours_last_year`, `num_promotions`, `years_since_last_promotion`

**Talent & Performance**
`performance_review`, `nine_box_talent_readiness`, `learning_score`, `engagement_score`, `job_satisfaction_score`, `work_life_balance_score`, `relationship_satisfaction`, `healthcare_satisfaction`

**People**
`supervisor_name`, `supervisor_email`, `talent_business_partner`, `has_successor`, `successor_name`

**Target**
`attrition` (Yes/No) · `attrition_risk_score` (0–1) · `exit_reason`

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-predictor.git
cd employee-attrition-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python src/generate_data.py
```
Outputs `data/employee_data.csv` — 12,000 rows, 54 columns.

### 4. Train the model
```bash
python src/train_model.py
```
Trains XGBoost, computes SHAP values, saves all artefacts to `models/`.

### 5. Launch the dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` 🎉

---

## 🗂️ Project Structure

```
employee_attrition_predictor/
│
├── app.py                      # Streamlit dashboard (5 pages)
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── generate_data.py        # Synthetic HR dataset generator
│   └── train_model.py          # XGBoost + SHAP training pipeline
│
├── data/
│   └── employee_data.csv       # Generated dataset (12K rows × 54 cols)
│
└── models/
    ├── attrition_model.pkl     # Trained XGBoost pipeline
    ├── feature_importance.csv  # SHAP-based feature ranking
    ├── model_metrics.json      # AUC, accuracy, F1, precision, recall
    ├── test_predictions.csv    # Held-out predictions for dashboard
    └── plots/
        └── shap_summary.png    # SHAP summary plot
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **ROC AUC** | ~0.87 |
| **Accuracy** | ~85% |
| **F1 Score** | ~0.72 |
| **Precision** | ~0.74 |
| **Recall** | ~0.70 |

*Results vary slightly by random seed. Trained on 80% of 12,000 records.*

---

## 🔍 Top Attrition Drivers (SHAP)

Based on SHAP analysis, the strongest predictors of employee attrition are:

1. **Overtime (Yes)** — single largest risk factor
2. **Engagement Score** — low engagement strongly predicts departure
3. **Job Satisfaction Score** — dissatisfied employees are 3× more likely to leave
4. **Work-Life Balance Score** — poor WLB amplifies all other risk factors
5. **Years Since Last Promotion** — stagnation beyond 4 years significantly raises risk
6. **Stock Options** — equity is a powerful retention anchor for senior employees
7. **Number of Companies Before** — serial job-hoppers have higher baseline risk
8. **Distance from Home** — long commutes compound dissatisfaction

---

## 🎛️ What-If Simulator

The dashboard's **What-If Simulator** lets HR partners model retention interventions before a 1:1:

- "What if we increase this employee's salary by 15%?"
- "What if we take them off overtime?"
- "What if we approve a remote-work arrangement?"

Adjust sliders → score updates instantly.

---

## 🏗️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data engineering |
| `scikit-learn` | Preprocessing, model pipeline |
| `XGBoost` | Primary classifier |
| `SHAP` | Model explainability |
| `Streamlit` | Interactive dashboard |
| `Plotly` | Charts & visualisations |
| `matplotlib` | Static SHAP plots |

---

## 🔮 Roadmap

- [ ] Weekly automated risk report (PDF/email via SendGrid)
- [ ] Live HRIS integration (Workday / SAP SuccessFactors API)
- [ ] Survival analysis — predict *when* an employee will leave, not just *if*
- [ ] Manager coaching nudges based on team risk score
- [ ] Streamlit Cloud deployment

---

## 👤 Author

Built by **Anushree Ayyar** — HR Tech / People Analytics portfolio project.

Connect on [LinkedIn](https://linkedin.com/in/anushreeayyar) · [GitHub](https://github.com/anushreeayyar)

---

## 📄 Licence

MIT — free to use, modify, and distribute.
