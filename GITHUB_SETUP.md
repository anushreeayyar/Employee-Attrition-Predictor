# 🚀 How to Push This Project to GitHub

Follow these steps to get `employee-attrition-predictor` live on your GitHub profile.

---

## Step 1 — Create a new repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `employee-attrition-predictor`
3. Description: `AI-powered HR attrition predictor with XGBoost, SHAP explainability, and an interactive Streamlit dashboard`
4. Set to **Public** (so it shows on your portfolio)
5. ✅ Check **"Add a README file"** — **NO** (we already have one)
6. Click **Create repository**

---

## Step 2 — Open a terminal in this project folder

```bash
cd path/to/employee_attrition_predictor
```

---

## Step 3 — Initialise git and push

```bash
# Initialise git
git init

# Stage all files
git add .

# First commit
git commit -m "feat: initial commit — Employee Attrition Predictor

- 12,000-row synthetic HR dataset (54 columns)
- XGBoost + SHAP model training pipeline
- 5-page Streamlit dashboard: Overview, At-Risk, Segments, SHAP, What-If"

# Connect to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/employee-attrition-predictor.git

# Push
git branch -M main
git push -u origin main
```

---

## Step 4 — Add a GitHub Topics (makes it discoverable)

On your repo page → click the ⚙️ gear next to **About** → add topics:
```
machine-learning, hr-analytics, xgboost, shap, streamlit, people-analytics,
attrition-prediction, python, data-science, portfolio
```

---

## Step 5 — Deploy dashboard on Streamlit Cloud (free!)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub account
4. Select: `YOUR_USERNAME / employee-attrition-predictor` → branch `main` → file `app.py`
5. Click **Deploy** — you'll get a public URL like:
   `https://your-username-employee-attrition-predictor.streamlit.app`
6. Add that URL to your GitHub repo's **About** section and your portfolio/LinkedIn

> **Note:** On Streamlit Cloud, run the data and model generation as a one-time step first,
> or commit the generated `data/` and `models/` folders (remove them from `.gitignore` if needed).

---

## Step 6 — Add the live demo link to your LinkedIn

In your LinkedIn **Featured** section:
> 🧠 **Employee Attrition Predictor** — ML model that identifies flight-risk employees with ~87% AUC. Built with XGBoost, SHAP, and Streamlit.
> 🔗 [Live Demo] | [GitHub Repo]
