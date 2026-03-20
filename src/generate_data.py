"""
Employee Attrition Predictor — Synthetic Dataset Generator
============================================================
Generates a realistic 12,000-row HR dataset with 55 columns.
Uses only pandas + numpy (no external dependencies).

Usage:
    python src/generate_data.py

Output:
    data/employee_data.csv
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
rng = np.random.default_rng(SEED)

N = 12_000

# ── Name pools ────────────────────────────────────────────────────────────────
FIRST_NAMES_M = [
    "James","John","Robert","Michael","William","David","Richard","Charles","Joseph","Thomas",
    "Christopher","Daniel","Paul","Mark","Donald","George","Kenneth","Steven","Edward","Brian",
    "Ronald","Anthony","Kevin","Jason","Jeffrey","Ryan","Jacob","Gary","Nicholas","Eric",
    "Jonathan","Stephen","Larry","Justin","Scott","Brandon","Benjamin","Samuel","Frank","Gregory",
    "Raymond","Alexander","Patrick","Jack","Dennis","Jerry","Tyler","Aaron","Jose","Adam",
    "Henry","Nathan","Douglas","Zachary","Peter","Kyle","Walter","Ethan","Jeremy","Harold",
]
FIRST_NAMES_F = [
    "Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth","Susan","Jessica","Sarah","Karen",
    "Lisa","Nancy","Betty","Margaret","Sandra","Ashley","Dorothy","Kimberly","Emily","Donna",
    "Michelle","Carol","Amanda","Melissa","Deborah","Stephanie","Rebecca","Sharon","Laura","Cynthia",
    "Kathleen","Amy","Angela","Shirley","Anna","Brenda","Pamela","Emma","Nicole","Helen",
    "Samantha","Katherine","Christine","Debra","Rachel","Carolyn","Janet","Catherine","Maria","Heather",
    "Diane","Julie","Joyce","Victoria","Kelly","Christina","Joan","Evelyn","Lauren","Judith",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
    "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
    "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts",
    "Turner","Phillips","Evans","Parker","Edwards","Collins","Stewart","Morris","Morales","Murphy",
    "Cook","Rogers","Gutierrez","Ortiz","Morgan","Cooper","Peterson","Bailey","Reed","Kelly",
    "Howard","Ramos","Kim","Cox","Ward","Richardson","Watson","Brooks","Chavez","Wood",
    "James","Bennett","Gray","Mendoza","Ruiz","Hughes","Price","Alvarez","Castillo","Sanders",
    "Patel","Myers","Long","Ross","Foster","Jimenez","Powell","Jenkins","Perry","Russell",
]

STREETS = [
    "Main St","Oak Ave","Maple Dr","Cedar Ln","Elm St","Pine Rd","Lake Blvd","Park Ave",
    "River Rd","Hill Dr","Sunset Blvd","Valley Rd","Forest Ave","Meadow Ln","Summit Dr",
    "Harbor View","Willow Way","Spring St","Autumn Ct","Canyon Rd",
]
CITIES = [
    "New York","Los Angeles","Chicago","Houston","Phoenix","Philadelphia","San Antonio",
    "San Diego","Dallas","San Jose","Austin","Jacksonville","Fort Worth","Columbus","Charlotte",
    "Indianapolis","San Francisco","Seattle","Denver","Nashville","Portland","Las Vegas",
    "Boston","Detroit","Louisville","Memphis","Baltimore","Milwaukee","Albuquerque","Tucson",
    "Atlanta","Raleigh","Minneapolis","Colorado Springs","Omaha","Miami","Cleveland","Tulsa",
    "Oakland","Tampa","Arlington","New Orleans","Wichita","Bakersfield","Aurora","Anaheim",
]
STATES = [
    "CA","NY","TX","FL","IL","WA","MA","CO","GA","NC",
    "AZ","NJ","OH","MI","MN","PA","VA","TN","OR","MD",
    "NV","IN","MO","WI","SC","AL","KY","LA","CT","OK",
]

BUSINESS_UNITS = [
    "Technology","Finance","Sales & Marketing","Operations","Human Resources",
    "Legal & Compliance","Customer Success","Research & Development","Supply Chain","Corporate Strategy",
]
DIVISIONS = {
    "Technology":             ["Engineering","Infrastructure","Product","Cybersecurity","Data & Analytics"],
    "Finance":                ["Accounting","FP&A","Treasury","Tax","Internal Audit"],
    "Sales & Marketing":      ["Enterprise Sales","SMB Sales","Digital Marketing","Brand","Partnerships"],
    "Operations":             ["Facilities","Procurement","Business Operations","Quality Assurance"],
    "Human Resources":        ["Talent Acquisition","HR Business Partners","L&D","Total Rewards","HR Ops"],
    "Legal & Compliance":     ["Corporate Legal","Regulatory","Privacy","Contracts"],
    "Customer Success":       ["Onboarding","Support Tier 1","Support Tier 2","Account Management"],
    "Research & Development": ["Applied Research","Product Research","Innovation Lab"],
    "Supply Chain":           ["Logistics","Inventory","Vendor Management","Distribution"],
    "Corporate Strategy":     ["M&A","Strategic Initiatives","Investor Relations"],
}
DEPARTMENTS = [
    "Software Engineering","Data Science","DevOps","Product Management","UX Design",
    "Financial Planning","Accounts Payable","Accounts Receivable","Sales Operations",
    "Field Sales","Inside Sales","Content Marketing","Performance Marketing","HR Operations",
    "Talent Development","IT Support","Network Infrastructure","Legal Affairs","Compliance",
    "Customer Onboarding","Technical Support","Account Management","Procurement","Logistics",
    "Quality Control","R&D Engineering","Strategic Planning","Corporate Communications","Payroll","Benefits",
]
JOB_LEVELS  = ["IC1","IC2","IC3","IC4","IC5","M1","M2","M3","M4","Director","VP","C-Suite"]
JOB_LEVEL_W = [15,   18,   14,   10,   7,    8,   6,   4,   3,    3,         2,   1    ]
JOB_CODES   = [f"JC-{1000+i}" for i in range(60)]

TITLE_PREFIXES   = ["Senior","Lead","Principal","Staff","Associate","Junior",""]
TITLE_ROLES_IC   = ["Analyst","Specialist","Engineer","Consultant","Coordinator","Advisor","Developer","Designer"]
TITLE_ROLES_MG   = ["Manager","Director","Head","Leader","Partner","VP"]

PERFORMANCE_RATINGS = ["Outstanding","Exceeds Expectations","Meets Expectations","Needs Improvement","Below Expectations"]
PERF_W              = [15, 20, 50, 10, 5]

EXIT_REASONS = [
    "Better Opportunity","Compensation","Work-Life Balance","Manager Relationship",
    "Career Growth","Relocation","Personal Reasons","Retirement",
    "Involuntary Termination","Contract End","Company Culture","Family Reasons",
]

TALENT_PARTNERS = [
    f"{random.choice(FIRST_NAMES_F+FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
    for _ in range(20)
]

SUPERVISOR_POOL = []  # built below


def _name(gender=None):
    if gender == "Male":
        return random.choice(FIRST_NAMES_M), random.choice(LAST_NAMES)
    elif gender == "Female":
        return random.choice(FIRST_NAMES_F), random.choice(LAST_NAMES)
    else:
        pool = FIRST_NAMES_M + FIRST_NAMES_F
        return random.choice(pool), random.choice(LAST_NAMES)


def _build_supervisor_pool(n=250):
    pool = []
    for _ in range(n):
        fn, ln = _name()
        pool.append({
            "name":  f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}@company.com",
        })
    return pool


def _address():
    num   = random.randint(100, 9999)
    st    = random.choice(STREETS)
    city  = random.choice(CITIES)
    state = random.choice(STATES)
    zipcode = f"{random.randint(10000,99999)}"
    return f"{num} {st}, {city}, {state} {zipcode}", state


def generate_record(emp_id: int) -> dict:
    # ── Demographics ──────────────────────────────────────────────────────
    gender = random.choices(["Male","Female","Non-Binary","Prefer Not to Say"],[48,46,4,2])[0]
    age    = int(np.clip(rng.normal(38, 10), 21, 65))
    fn, ln = _name(gender)
    name   = f"{fn} {ln}"

    # ── Org ───────────────────────────────────────────────────────────────
    bu         = random.choice(BUSINESS_UNITS)
    division   = random.choice(DIVISIONS[bu])
    department = random.choice(DEPARTMENTS)
    job_level  = random.choices(JOB_LEVELS, weights=JOB_LEVEL_W)[0]
    is_manager = job_level in {"M1","M2","M3","M4","Director","VP","C-Suite"}
    job_code   = random.choice(JOB_CODES)

    if is_manager:
        title_role   = random.choice(TITLE_ROLES_MG)
        title_prefix = random.choice(["Senior","Head of","","Global"])
    else:
        title_role   = random.choice(TITLE_ROLES_IC)
        title_prefix = random.choice(TITLE_PREFIXES)
    business_title = f"{title_prefix} {department} {title_role}".strip().replace("  "," ")

    cost_center      = f"CC-{random.randint(1000,9999)}"
    cost_center_name = f"{bu} – {division}"

    # ── Dates ─────────────────────────────────────────────────────────────
    today          = datetime(2025, 6, 30)
    max_days       = (today - datetime(2000, 1, 1)).days
    days_employed  = int(np.clip(rng.exponential(1300), 30, max_days))
    start_date     = (today - timedelta(days=days_employed)).strftime("%Y-%m-%d")
    tenure_years   = round(days_employed / 365.25, 2)
    years_at_co    = int(tenure_years)

    # ── Compensation ──────────────────────────────────────────────────────
    base_map = {
        "IC1":55000,"IC2":72000,"IC3":90000,"IC4":115000,"IC5":140000,
        "M1":110000,"M2":130000,"M3":155000,"M4":180000,
        "Director":200000,"VP":250000,"C-Suite":350000,
    }
    base      = base_map[job_level]
    salary    = int(np.clip(rng.normal(base, base*0.12), 38000, 650000))
    bonus_pct = round(random.uniform(8,40) if is_manager else random.uniform(0,20), 1)
    stock     = random.randint(500,5000) if job_level in {"IC4","IC5","M1","M2","M3","M4","Director","VP","C-Suite"} else 0
    last_raise = round(random.uniform(0, 20), 1)

    # ── Performance / Talent ──────────────────────────────────────────────
    perf_rating   = random.choices(PERFORMANCE_RATINGS, weights=PERF_W)[0]
    learning_sc   = round(float(np.clip(rng.normal(72, 15), 0, 100)), 1)
    engagement_sc = round(float(np.clip(rng.normal(68, 18), 0, 100)), 1)
    job_sat       = round(float(np.clip(rng.normal(65, 20), 0, 100)), 1)
    wlb_sc        = round(float(np.clip(rng.normal(60, 22), 0, 100)), 1)
    rel_sat       = round(float(np.clip(rng.normal(70, 15), 0, 100)), 1)
    nine_box      = random.randint(1, 9)
    healthcare_sc = round(float(np.clip(rng.normal(65, 20), 0, 100)), 1)
    has_successor = bool(random.random() < 0.25) if is_manager else False
    successor_nm  = f"{random.choice(FIRST_NAMES_M+FIRST_NAMES_F)} {random.choice(LAST_NAMES)}" if has_successor else ""

    # ── Work patterns ─────────────────────────────────────────────────────
    overtime       = random.choices(["Yes","No"], weights=[35,65])[0]
    monthly_hrs    = int(np.clip(rng.normal(210 if overtime=="Yes" else 175, 20), 100, 280))
    remote_pct     = random.choices([0,25,50,75,100], weights=[15,10,30,20,25])[0]
    training_hrs   = int(np.clip(rng.normal(32,18), 0, 120))
    num_promotions = random.choices([0,1,2,3,4], weights=[40,30,18,8,4])[0]
    yrs_since_promo= round(random.uniform(0, min(tenure_years, 8)), 1)
    prev_companies = random.choices([0,1,2,3,4,5,6], weights=[10,20,25,20,12,8,5])[0]

    # ── Leave ─────────────────────────────────────────────────────────────
    last_day_w = ""
    paid_unpaid = ""
    if random.random() < 0.07:
        leave_start = today - timedelta(days=random.randint(10,300))
        last_day_w  = (leave_start - timedelta(days=1)).strftime("%Y-%m-%d")
        paid_unpaid = random.choice(["Paid","Unpaid"])

    # ── Personal ──────────────────────────────────────────────────────────
    address, state = _address()
    marital = random.choices(["Single","Married","Divorced","Widowed"],[35,50,12,3])[0]
    dist    = int(np.clip(rng.exponential(18), 1, 90))
    commute = int(dist * random.uniform(1.2, 3.5))

    # ── Supervisor ────────────────────────────────────────────────────────
    sup = random.choice(SUPERVISOR_POOL)

    # ── Manager-specific ─────────────────────────────────────────────────
    num_direct   = random.randint(2,12) if is_manager else 0
    team_size_n  = num_direct + random.randint(0,5) if is_manager else random.randint(3,15)

    # ── Contact ───────────────────────────────────────────────────────────
    email   = f"{fn.lower()}.{ln.lower()}{emp_id % 100}@company.com"
    tbp     = random.choice(TALENT_PARTNERS)

    # ── Attrition label (engineered correlations) ─────────────────────────
    score = 0.0
    score += 0.30 if overtime == "Yes" else 0.0
    score += 0.20 if engagement_sc < 40 else (0.10 if engagement_sc < 55 else 0.0)
    score += 0.20 if job_sat < 40 else (0.10 if job_sat < 55 else 0.0)
    score += 0.15 if wlb_sc < 40 else 0.0
    score += 0.15 if yrs_since_promo > 4 else 0.0
    score += 0.12 if prev_companies >= 4 else 0.0
    score += 0.10 if dist > 50 else 0.0
    score += 0.10 if perf_rating in {"Needs Improvement","Below Expectations"} else 0.0
    score += 0.10 if tenure_years < 2 else 0.0
    score += 0.05 if age < 28 else 0.0
    score -= 0.15 if stock > 2000 else 0.0
    score -= 0.10 if marital == "Married" else 0.0
    score -= 0.10 if num_promotions >= 2 else 0.0
    score -= 0.08 if training_hrs > 60 else 0.0
    score -= 0.05 if is_manager else 0.0
    score -= 0.08 if nine_box >= 7 else 0.0
    score  = float(np.clip(score + rng.normal(0, 0.10), 0.0, 1.0))
    attrition = "Yes" if score > 0.45 else "No"

    exit_reason = random.choice(EXIT_REASONS) if attrition == "Yes" else ""
    term_date = ""
    if attrition == "Yes":
        t = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days_employed + random.randint(30,365))
        if t <= today:
            term_date = t.strftime("%Y-%m-%d")

    return {
        "employee_id":                f"EMP-{100000+emp_id}",
        "full_name":                  name,
        "email":                      email,
        "gender":                     gender,
        "age":                        age,
        "marital_status":             marital,
        "address":                    address,
        "state":                      state,
        "distance_from_home_miles":   dist,
        "commute_time_minutes":       commute,
        "business_title":             business_title,
        "business_unit":              bu,
        "division":                   division,
        "department":                 department,
        "cost_center":                cost_center,
        "cost_center_name":           cost_center_name,
        "job_level":                  job_level,
        "job_code":                   job_code,
        "is_manager":                 is_manager,
        "num_direct_reports":         num_direct,
        "team_size":                  team_size_n,
        "current_start_date":         start_date,
        "tenure_years":               tenure_years,
        "years_at_company":           years_at_co,
        "last_day_working":           last_day_w,
        "paid_or_unpaid_leave":       paid_unpaid,
        "salary":                     salary,
        "bonus_percentage":           bonus_pct,
        "stock_options":              stock,
        "last_raise_percentage":      last_raise,
        "overtime":                   overtime,
        "remote_work_percentage":     remote_pct,
        "monthly_hours_worked":       monthly_hrs,
        "num_companies_before":       prev_companies,
        "num_promotions":             num_promotions,
        "years_since_last_promotion": yrs_since_promo,
        "training_hours_last_year":   training_hrs,
        "performance_review":         perf_rating,
        "engagement_score":           engagement_sc,
        "job_satisfaction_score":     job_sat,
        "work_life_balance_score":    wlb_sc,
        "relationship_satisfaction":  rel_sat,
        "learning_score":             learning_sc,
        "nine_box_talent_readiness":  nine_box,
        "healthcare_satisfaction":    healthcare_sc,
        "supervisor_name":            sup["name"],
        "supervisor_email":           sup["email"],
        "talent_business_partner":    tbp,
        "has_successor":              has_successor,
        "successor_name":             successor_nm,
        "attrition":                  attrition,
        "exit_reason":                exit_reason,
        "termination_date":           term_date,
        "attrition_risk_score":       round(score, 4),
    }


def main():
    global SUPERVISOR_POOL
    print(f"🔄 Generating {N:,} employee records …")
    SUPERVISOR_POOL = _build_supervisor_pool(250)

    records = [generate_record(i) for i in range(1, N + 1)]
    df = pd.DataFrame(records)

    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "employee_data.csv")
    df.to_csv(out, index=False)

    rate = df["attrition"].eq("Yes").mean() * 100
    print(f"✅ Saved → {out}")
    print(f"   Rows      : {len(df):,}")
    print(f"   Columns   : {len(df.columns)}")
    print(f"   Attrition : {rate:.1f}%  ({df['attrition'].eq('Yes').sum():,} employees left)")
    print(f"\nColumns ({len(df.columns)}):")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:>2}. {c}")


if __name__ == "__main__":
    main()
