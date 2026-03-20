"""
Employee Attrition Predictor — Streamlit Dashboard
====================================================
Interactive dashboard featuring:
  • Flight Risk Overview        — company-wide KPIs & risk distribution
  • At-Risk Employee Table      — searchable/sortable high-risk roster
  • Segment Analysis            — attrition by dept, tenure, manager, BU
  • SHAP Explainability         — top drivers with waterfall charts
  • What-If Simulator           — adjust employee attributes & re-score
  • Model Performance           — confusion matrix, AUC, metrics

Usage:
    streamlit run app.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12); text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { color: #6b7280; font-size: 0.85rem; margin-top: 0.2rem; }
    .risk-high   { color: #ef4444; }
    .risk-medium { color: #f59e0b; }
    .risk-low    { color: #10b981; }
    div[data-testid="metric-container"] { background: white; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join("data", "employee_data.csv"))
    return df


@st.cache_resource
def load_model():
    path = os.path.join("models", "attrition_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_predictions():
    path = os.path.join("models", "test_predictions.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    path = os.path.join("models", "model_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_importance():
    path = os.path.join("models", "feature_importance.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def risk_label(score: float) -> str:
    if score >= 0.65:  return "🔴 High"
    if score >= 0.35:  return "🟡 Medium"
    return "🟢 Low"


def risk_color(score: float) -> str:
    if score >= 0.65:  return "#ef4444"
    if score >= 0.35:  return "#f59e0b"
    return "#10b981"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.markdown("## 🧠 Attrition Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🚨 At-Risk Employees", "📊 Segment Analysis",
         "🔍 SHAP Explainability", "🎛️ What-If Simulator", "📈 Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")

    df_raw = load_data()

    bu_filter    = st.multiselect("Business Unit",  sorted(df_raw["business_unit"].unique()), default=[])
    dept_filter  = st.multiselect("Department",     sorted(df_raw["department"].unique()),    default=[])
    level_filter = st.multiselect("Job Level",      sorted(df_raw["job_level"].unique()),     default=[])

    risk_threshold = st.slider("Flight Risk Threshold", 0.0, 1.0, 0.50, 0.05,
                               help="Employees with predicted risk ≥ this value are flagged")

    st.markdown("---")
    st.markdown(
        "<small>Built with XGBoost + SHAP + Streamlit</small>",
        unsafe_allow_html=True,
    )


# ── Load everything ───────────────────────────────────────────────────────────
df       = load_data()
model    = load_model()
preds    = load_predictions()
metrics  = load_metrics()
imp_df   = load_importance()

# Apply sidebar filters to main dataframe
df_f = df.copy()
if bu_filter:    df_f = df_f[df_f["business_unit"].isin(bu_filter)]
if dept_filter:  df_f = df_f[df_f["department"].isin(dept_filter)]
if level_filter: df_f = df_f[df_f["job_level"].isin(level_filter)]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="main-header">Employee Attrition Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered flight risk detection & retention intelligence</div>', unsafe_allow_html=True)

    total       = len(df_f)
    left        = (df_f["attrition"] == "Yes").sum()
    attrition_r = left / total * 100
    managers    = df_f["is_manager"].sum()
    avg_tenure  = df_f["tenure_years"].mean()
    avg_engage  = df_f["engagement_score"].mean()
    avg_salary  = df_f["salary"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees",    f"{total:,}")
    c2.metric("Attrition Rate",     f"{attrition_r:.1f}%",  delta=f"-{left:,} left")
    c3.metric("Avg Tenure",         f"{avg_tenure:.1f} yrs")
    c4.metric("Avg Engagement",     f"{avg_engage:.0f}/100")
    c5.metric("Avg Salary",         f"${avg_salary:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Attrition by Business Unit
        bu_atr = df_f.groupby("business_unit")["attrition"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index(name="Attrition Rate (%)")
        bu_atr = bu_atr.sort_values("Attrition Rate (%)", ascending=True)
        fig = px.bar(bu_atr, x="Attrition Rate (%)", y="business_unit", orientation="h",
                     title="Attrition Rate by Business Unit",
                     color="Attrition Rate (%)", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk score distribution (from actual attrition_risk_score)
        fig = px.histogram(df_f, x="attrition_risk_score", nbins=40,
                           title="Attrition Risk Score Distribution",
                           color_discrete_sequence=["#667eea"])
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold: {risk_threshold:.2f}")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Attrition by tenure bracket
        df_f2 = df_f.copy()
        df_f2["tenure_bracket"] = pd.cut(
            df_f2["tenure_years"],
            bins=[0,1,3,5,10,50],
            labels=["<1yr","1–3yr","3–5yr","5–10yr","10+yr"],
        )
        ten_atr = df_f2.groupby("tenure_bracket", observed=True)["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reset_index(name="Attrition %")
        fig = px.bar(ten_atr, x="tenure_bracket", y="Attrition %",
                     title="Attrition Rate by Tenure Bracket",
                     color="Attrition %", color_continuous_scale="Reds")
        fig.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Engagement vs Salary scatter (sample 1000)
        sample = df_f.sample(min(1000, len(df_f)), random_state=42)
        fig = px.scatter(sample, x="engagement_score", y="salary",
                         color="attrition", opacity=0.6,
                         title="Engagement vs Salary (coloured by Attrition)",
                         color_discrete_map={"Yes":"#ef4444","No":"#10b981"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AT-RISK EMPLOYEES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 At-Risk Employees":
    st.markdown("## 🚨 At-Risk Employees")
    st.markdown("Employees whose flight risk score exceeds the threshold you set in the sidebar.")

    high_risk = df_f[df_f["attrition_risk_score"] >= risk_threshold].copy()
    high_risk["risk_category"] = high_risk["attrition_risk_score"].apply(risk_label)
    high_risk = high_risk.sort_values("attrition_risk_score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("At-Risk Employees", f"{len(high_risk):,}")
    c2.metric("% of Workforce",    f"{len(high_risk)/len(df_f)*100:.1f}%")
    c3.metric("High Risk (≥0.65)", f"{(high_risk['attrition_risk_score'] >= 0.65).sum():,}")

    # Search
    search = st.text_input("🔍 Search by name, department, or supervisor", "")

    display_cols = [
        "employee_id","full_name","department","job_level","tenure_years",
        "salary","engagement_score","job_satisfaction_score","overtime",
        "years_since_last_promotion","attrition_risk_score","risk_category",
    ]
    display_df = high_risk[display_cols].copy()
    display_df["attrition_risk_score"] = display_df["attrition_risk_score"].map("{:.2%}".format)

    if search:
        mask = (
            high_risk["full_name"].str.contains(search, case=False, na=False) |
            high_risk["department"].str.contains(search, case=False, na=False) |
            high_risk["supervisor_name"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    st.dataframe(
        display_df.style.applymap(
            lambda v: "color: #ef4444; font-weight: bold" if "🔴" in str(v) else
                      "color: #f59e0b; font-weight: bold" if "🟡" in str(v) else "",
            subset=["risk_category"],
        ),
        use_container_width=True,
        height=450,
    )

    # Download
    csv = high_risk.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download At-Risk List (CSV)", csv,
                       "at_risk_employees.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Segment Analysis":
    st.markdown("## 📊 Segment Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "By Department", "By Job Level", "By Manager Status",
        "By Overtime", "By Performance"
    ])

    def attrition_bar(group_col: str, title: str, df_input=None):
        if df_input is None:
            df_input = df_f
        g = df_input.groupby(group_col)["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reset_index(name="Attrition %")
        g = g.sort_values("Attrition %", ascending=False)
        fig = px.bar(g, x=group_col, y="Attrition %", title=title,
                     color="Attrition %", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False, height=380)
        return fig

    with tab1:
        top_dept = df_f.groupby("department")["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reset_index(name="Attrition %").sort_values("Attrition %", ascending=False).head(15)
        fig = px.bar(top_dept, x="Attrition %", y="department", orientation="h",
                     title="Top 15 Departments by Attrition Rate",
                     color="Attrition %", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        level_order = ["IC1","IC2","IC3","IC4","IC5","M1","M2","M3","M4","Director","VP","C-Suite"]
        g = df_f.groupby("job_level")["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reindex(level_order).dropna().reset_index(name="Attrition %")
        fig = px.bar(g, x="job_level", y="Attrition %",
                     title="Attrition Rate by Job Level",
                     color="Attrition %", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        mgr = df_f.groupby("is_manager")["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reset_index(name="Attrition %")
        mgr["is_manager"] = mgr["is_manager"].map({True:"Manager", False:"Individual Contributor"})
        fig = px.bar(mgr, x="is_manager", y="Attrition %",
                     title="Attrition: Manager vs Individual Contributor",
                     color="is_manager",
                     color_discrete_map={"Manager":"#667eea","Individual Contributor":"#f093fb"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Top supervisors by team attrition
        st.markdown("#### Supervisors with Highest Team Attrition")
        sup_atr = df_f.groupby("supervisor_name")["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100 if len(x) >= 5 else np.nan
        ).dropna().sort_values(ascending=False).head(10).reset_index(name="Team Attrition %")
        st.dataframe(sup_atr, use_container_width=True)

    with tab4:
        ot = df_f.groupby("overtime")["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reset_index(name="Attrition %")
        fig = px.bar(ot, x="overtime", y="Attrition %",
                     title="Attrition by Overtime Status",
                     color="overtime",
                     color_discrete_map={"Yes":"#ef4444","No":"#10b981"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.box(df_f, x="overtime", y="engagement_score",
                          color="overtime", title="Engagement Score by Overtime",
                          color_discrete_map={"Yes":"#ef4444","No":"#10b981"})
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.box(df_f, x="overtime", y="work_life_balance_score",
                          color="overtime", title="Work-Life Balance by Overtime",
                          color_discrete_map={"Yes":"#ef4444","No":"#10b981"})
            st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        pr = df_f.groupby("performance_review")["attrition"].apply(
            lambda x: (x=="Yes").sum()/len(x)*100
        ).reset_index(name="Attrition %")
        fig = px.bar(pr, x="performance_review", y="Attrition %",
                     title="Attrition by Performance Rating",
                     color="Attrition %", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 SHAP Explainability":
    st.markdown("## 🔍 SHAP Feature Importance")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows which features drive each prediction. "
        "Higher mean |SHAP| = stronger influence on attrition probability."
    )

    if imp_df is not None:
        # Clean up feature names (remove sklearn prefix)
        imp_df["feature_clean"] = imp_df["feature"].str.replace("num__","").str.replace("cat__","")

        top_n = st.slider("Number of features to display", 5, 30, 15)
        top   = imp_df.head(top_n).copy()

        fig = px.bar(top.sort_values("mean_shap"), x="mean_shap", y="feature_clean",
                     orientation="h", title=f"Top {top_n} Attrition Drivers (Mean |SHAP|)",
                     color="mean_shap", color_continuous_scale="Viridis")
        fig.update_layout(coloraxis_showscale=False, height=max(350, top_n * 28))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Full Feature Importance Table")
        st.dataframe(
            imp_df[["feature_clean","mean_shap"]].rename(
                columns={"feature_clean":"Feature","mean_shap":"Mean |SHAP|"}
            ).style.format({"Mean |SHAP|": "{:.4f}"}),
            use_container_width=True,
            height=350,
        )

        shap_img = os.path.join("models","plots","shap_summary.png")
        if os.path.exists(shap_img):
            st.markdown("---")
            st.markdown("#### SHAP Summary Plot")
            st.image(shap_img, use_column_width=True)
    else:
        st.info("⚠️ Run `python src/train_model.py` first to generate SHAP values.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎛️ What-If Simulator":
    st.markdown("## 🎛️ What-If Simulator")
    st.markdown(
        "Adjust any employee attribute and instantly see how it affects their predicted flight risk. "
        "Ideal for testing retention levers before a 1:1 conversation."
    )

    if model is None:
        st.info("⚠️ Run `python src/train_model.py` first to load the model.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Employee Profile")
            gender   = st.selectbox("Gender", ["Male","Female","Non-Binary","Prefer Not to Say"])
            age      = st.slider("Age", 21, 65, 32)
            marital  = st.selectbox("Marital Status", ["Single","Married","Divorced","Widowed"])
            state    = st.selectbox("State", sorted(df["state"].unique()), index=0)
            dist     = st.slider("Distance from Home (miles)", 1, 90, 15)
            commute  = st.slider("Commute Time (mins)", 5, 200, 30)

            st.markdown("#### Role & Compensation")
            bu       = st.selectbox("Business Unit", sorted(df["business_unit"].unique()))
            division = st.selectbox("Division", sorted(df["division"].unique()))
            dept     = st.selectbox("Department", sorted(df["department"].unique()))
            jlevel   = st.selectbox("Job Level", ["IC1","IC2","IC3","IC4","IC5","M1","M2","M3","M4","Director","VP","C-Suite"])
            is_mgr   = jlevel in {"M1","M2","M3","M4","Director","VP","C-Suite"}
            salary   = st.number_input("Salary ($)", 30000, 600000, 85000, step=5000)
            bonus    = st.slider("Bonus %", 0.0, 40.0, 10.0)
            stock    = st.slider("Stock Options", 0, 5000, 500)
            last_r   = st.slider("Last Raise %", 0.0, 20.0, 5.0)

        with col_right:
            st.markdown("#### Work Patterns")
            overtime  = st.selectbox("Overtime", ["No","Yes"])
            remote    = st.select_slider("Remote Work %", [0,25,50,75,100], value=50)
            monthly_h = st.slider("Monthly Hours Worked", 100, 280, 175)
            training  = st.slider("Training Hours (last year)", 0, 120, 30)
            prev_co   = st.slider("Companies Worked Before", 0, 6, 2)
            promotions= st.slider("Number of Promotions", 0, 4, 1)
            yrs_promo = st.slider("Years Since Last Promotion", 0.0, 8.0, 2.0)

            st.markdown("#### Sentiment Scores")
            engage    = st.slider("Engagement Score", 0, 100, 65)
            job_sat   = st.slider("Job Satisfaction", 0, 100, 65)
            wlb       = st.slider("Work-Life Balance", 0, 100, 60)
            rel_sat   = st.slider("Relationship Satisfaction", 0, 100, 70)
            learning  = st.slider("Learning Score", 0, 100, 72)
            nine_box  = st.slider("9-Box Readiness (1–9)", 1, 9, 5)
            health    = st.slider("Healthcare Satisfaction", 0, 100, 65)
            perf      = st.selectbox("Performance Review", [
                "Outstanding","Exceeds Expectations","Meets Expectations",
                "Needs Improvement","Below Expectations"
            ])

        # Build input row
        input_dict = {
            "gender": gender, "age": age, "marital_status": marital,
            "state": state, "distance_from_home_miles": dist,
            "commute_time_minutes": commute, "business_unit": bu,
            "division": division, "department": dept, "cost_center_name": f"{bu} – {division}",
            "job_level": jlevel, "is_manager": int(is_mgr),
            "num_direct_reports": 4 if is_mgr else 0,
            "team_size": 8 if is_mgr else 5,
            "tenure_years": 3.0, "years_at_company": 3,
            "salary": salary, "bonus_percentage": bonus,
            "stock_options": stock, "last_raise_percentage": last_r,
            "overtime": overtime, "remote_work_percentage": remote,
            "monthly_hours_worked": monthly_h,
            "num_companies_before": prev_co, "num_promotions": promotions,
            "years_since_last_promotion": yrs_promo,
            "training_hours_last_year": training,
            "performance_review": perf,
            "engagement_score": engage, "job_satisfaction_score": job_sat,
            "work_life_balance_score": wlb, "relationship_satisfaction": rel_sat,
            "learning_score": learning, "nine_box_talent_readiness": nine_box,
            "healthcare_satisfaction": health, "has_successor": int(is_mgr and nine_box >= 7),
        }

        X_in = pd.DataFrame([input_dict])

        try:
            prob = model.predict_proba(X_in)[0][1]
        except Exception as e:
            prob = None
            st.error(f"Prediction error: {e}")

        if prob is not None:
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            gauge_color = "#ef4444" if prob >= 0.65 else "#f59e0b" if prob >= 0.35 else "#10b981"
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                domain= {"x": [0, 1], "y": [0, 1]},
                title = {"text": "Flight Risk Score (%)"},
                delta = {"reference": risk_threshold * 100},
                gauge = {
                    "axis":  {"range": [0, 100], "tickwidth": 1},
                    "bar":   {"color": gauge_color},
                    "steps": [
                        {"range": [0, 35],  "color": "#d1fae5"},
                        {"range": [35, 65], "color": "#fef3c7"},
                        {"range": [65, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "black","width": 3},
                        "thickness": 0.75,
                        "value": risk_threshold * 100,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            label = risk_label(prob)
            st.markdown(
                f"<h3 style='text-align:center; color:{gauge_color}'>"
                f"Risk Category: {label} &nbsp;|&nbsp; Score: {prob:.1%}</h3>",
                unsafe_allow_html=True,
            )

            # Retention suggestions
            suggestions = []
            if overtime == "Yes":
                suggestions.append("📌 **Reduce overtime** — strongest single predictor of attrition")
            if engage < 55:
                suggestions.append("📌 **Run an engagement pulse survey** and address gaps")
            if job_sat < 55:
                suggestions.append("📌 **Schedule a career development 1:1** to surface concerns")
            if yrs_promo > 4:
                suggestions.append("📌 **Consider a promotion or role expansion** — stagnation risk")
            if stock == 0 and jlevel in ["IC4","IC5","M1","M2","M3","M4"]:
                suggestions.append("📌 **Offer stock options** — retention anchoring for senior ICs/managers")
            if training < 20:
                suggestions.append("📌 **Enrol in a learning programme** — low L&D correlates with flight risk")

            if suggestions:
                st.markdown("#### 💡 Retention Recommendations")
                for s in suggestions:
                    st.markdown(s)
            else:
                st.success("✅ This employee looks well-retained. Keep up the good work!")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{metrics.get('accuracy',0):.1%}")
        c2.metric("ROC AUC",   f"{metrics.get('roc_auc',0):.4f}")
        c3.metric("F1 Score",  f"{metrics.get('f1',0):.4f}")
        c4.metric("Precision", f"{metrics.get('precision',0):.4f}")
        c5.metric("Recall",    f"{metrics.get('recall',0):.4f}")

    if preds is not None:
        from sklearn.metrics import roc_curve, confusion_matrix
        import plotly.figure_factory as ff

        y_true  = preds["attrition_actual"]
        y_pred  = preds["attrition_predicted"]
        y_proba = preds["flight_risk_score"]

        col1, col2 = st.columns(2)

        with col1:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = metrics.get("roc_auc", 0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"XGBoost (AUC={auc_val:.3f})",
                                     line=dict(color="#667eea", width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                     line=dict(dash="dash", color="gray")))
            fig.update_layout(
                title="ROC Curve", xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate", height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = ff.create_annotated_heatmap(
                z=cm, x=["Predicted Stayed","Predicted Left"],
                y=["Actual Stayed","Actual Left"],
                colorscale="Blues", showscale=False,
            )
            fig.update_layout(title="Confusion Matrix", height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Precision-Recall at various thresholds
        st.markdown("---")
        st.markdown("#### Precision / Recall at Different Thresholds")
        thresholds = np.arange(0.1, 0.9, 0.05)
        pr_data = []
        for t in thresholds:
            yp = (y_proba >= t).astype(int)
            tp_ = ((yp==1)&(y_true==1)).sum()
            fp_ = ((yp==1)&(y_true==0)).sum()
            fn_ = ((yp==0)&(y_true==1)).sum()
            p   = tp_/(tp_+fp_) if (tp_+fp_) > 0 else 0
            r   = tp_/(tp_+fn_) if (tp_+fn_) > 0 else 0
            pr_data.append({"Threshold": round(t,2), "Precision": round(p,3), "Recall": round(r,3)})
        pr_df = pd.DataFrame(pr_data)
        fig = px.line(pr_df, x="Threshold", y=["Precision","Recall"],
                      title="Precision & Recall vs. Decision Threshold",
                      color_discrete_map={"Precision":"#667eea","Recall":"#ef4444"})
        fig.add_vline(x=risk_threshold, line_dash="dash", annotation_text=f"Current: {risk_threshold:.2f}")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Run `python src/train_model.py` to generate model artefacts.")
