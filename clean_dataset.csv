import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise · Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #18181f;
    --border: #26263a;
    --accent: #6c63ff;
    --accent2: #ff6b6b;
    --accent3: #43e97b;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --green: #43e97b;
    --red: #ff6b6b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Force sidebar always visible */
[data-testid="stSidebar"] {
    transform: none !important;
    width: 300px !important;
    min-width: 300px !important;
}

/* Fix sidebar toggle button - always visible */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: white !important;
    background: #6c63ff !important;
    border-radius: 0 8px 8px 0 !important;
    z-index: 999 !important;
}

/* Main heading */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 50%, #43e97b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6c63ff, #43e97b);
}
.metric-card:hover { border-color: var(--accent); }
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.25rem;
}

/* Result card */
.result-approved {
    background: linear-gradient(135deg, rgba(67,233,123,0.12), rgba(67,233,123,0.04));
    border: 1.5px solid rgba(67,233,123,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.04));
    border: 1.5px solid rgba(255,107,107,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
}
.result-title.approved { color: var(--green); }
.result-title.rejected { color: var(--red); }
.result-prob {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.25rem;
}

/* Section headings */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.25rem 0;
}
.section-divider {
    height: 1px;
    background: var(--border);
    margin: 0.75rem 0 1.25rem 0;
}

/* Factor bar */
.factor-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.factor-label {
    font-size: 0.82rem;
    color: var(--muted);
    width: 160px;
    flex-shrink: 0;
}
.factor-bar-wrap {
    flex: 1;
    background: var(--border);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.factor-bar-fill {
    height: 100%;
    border-radius: 999px;
}
.factor-val {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* Streamlit widget overrides */
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }
div[data-testid="stNumberInput"] input { background: var(--surface2) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 10px !important; }
.stSelectbox [data-baseweb="select"] > div { background: var(--surface2) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 10px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 12px; padding: 4px; border: 1px solid var(--border); gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: white !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(108,99,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(108,99,255,0.5) !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--border) !important; }

/* Sidebar nav */
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    border-radius: 10px;
    cursor: pointer;
    margin-bottom: 4px;
    font-size: 0.9rem;
    font-weight: 500;
    transition: background 0.15s;
}
.nav-item.active {
    background: rgba(108,99,255,0.2);
    color: #a78bfa;
}
.nav-item:hover { background: rgba(255,255,255,0.05); }

/* Alert banner */
.tip-box {
    background: rgba(108,99,255,0.1);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    color: #c4b5fd;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA & MODEL LOADING ────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")

    # Imputation
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # Drop ID
    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)

    # Encode
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    # One-hot
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Feature engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

    X = df.drop(columns=["Loan_Approved", "DTI_Ratio", "Credit_Score"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        pred = m.predict(X_test_s)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, pred) * 100, 1),
            "Precision": round(precision_score(y_test, pred) * 100, 1),
            "Recall": round(recall_score(y_test, pred) * 100, 1),
            "F1": round(f1_score(y_test, pred) * 100, 1),
            "CM": confusion_matrix(y_test, pred),
        }
        trained[name] = m

    return df, X_train, X_test, y_train, y_test, scaler, trained, results, X.columns.tolist()

df_raw = pd.read_csv("loan_approval_data.csv")
df, X_train, X_test, y_train, y_test, scaler, trained_models, results, feature_cols = load_and_train()


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 800;
                    background: linear-gradient(135deg, #6c63ff, #43e97b);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            💳 CreditWise
        </div>
        <div style='font-size: 0.78rem; color: #6b6b80; margin-top: 2px;'>ML Loan Approval System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 Dataset Explorer", "🤖 Model Performance", "🔮 Loan Predictor"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #6b6b80; line-height: 1.6;'>
        <b style='color: #9090a8;'>Dataset</b><br>
        1,000 loan applications<br>
        19 raw features → 28 engineered<br><br>
        <b style='color: #9090a8;'>Best Model</b><br>
        Logistic Regression · 88.0% Accuracy · F1 80.9%
    </div>
    """, unsafe_allow_html=True)

page_name = page.split(" ", 1)[1]


# ─── DASHBOARD ───────────────────────────────────────────────────────────────
if page_name == "Dashboard":
    st.markdown('<div class="hero-title">Loan Intelligence<br>Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML-powered credit risk assessment · Real-time probability scoring</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    approved = (df_raw["Loan_Approved"] == "Yes").sum() if "Loan_Approved" in df_raw.columns else (df["Loan_Approved"] == 1).sum()
    total = len(df_raw)
    approval_rate = round(approved / total * 100, 1)
    avg_income = round(df_raw["Applicant_Income"].mean())
    avg_loan = round(df_raw["Loan_Amount"].mean()) if "Loan_Amount" in df_raw.columns else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Applications</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-sub">Loan applications</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Approval Rate</div>
            <div class="metric-value" style="color:#43e97b">{approval_rate}%</div>
            <div class="metric-sub">{approved} approved</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg. Applicant Income</div>
            <div class="metric-value">₹{avg_income:,}</div>
            <div class="metric-sub">Monthly</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        best_acc = max(v["Accuracy"] for v in results.values())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Model F1 Score</div>
            <div class="metric-value" style="color:#a78bfa">80.9%</div>
            <div class="metric-sub">Logistic Regression</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">Income Distribution by Approval</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        fig = px.histogram(
            df_raw, x="Applicant_Income", color="Loan_Approved",
            nbins=30, barmode="overlay", opacity=0.8,
            color_discrete_map={"Yes": "#43e97b", "No": "#ff6b6b"},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#9090a8"),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            legend=dict(title="", orientation="h", y=1.05),
            xaxis=dict(gridcolor="#26263a"), yaxis=dict(gridcolor="#26263a"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Employment Status Mix</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        emp_counts = df_raw["Employment_Status"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=emp_counts.index,
            values=emp_counts.values,
            hole=0.6,
            marker_colors=["#6c63ff", "#43e97b", "#ff6b6b", "#ffd166", "#a78bfa"],
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#9090a8"),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            showlegend=True, legend=dict(orientation="v"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Credit score vs approval
    st.markdown('<div class="section-title">Credit Score vs Approval Rate</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    df_raw["Credit_Bin"] = pd.cut(df_raw["Credit_Score"], bins=5)
    agg = df_raw.groupby("Credit_Bin").apply(
        lambda g: (g["Loan_Approved"] == "Yes").mean() * 100
    ).reset_index()
    agg.columns = ["Credit Range", "Approval %"]
    agg["Credit Range"] = agg["Credit Range"].astype(str)
    fig3 = px.bar(agg, x="Credit Range", y="Approval %",
                  color="Approval %", color_continuous_scale=["#ff6b6b", "#ffd166", "#43e97b"],
                  template="plotly_dark")
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#9090a8"),
        margin=dict(l=0, r=0, t=10, b=0), height=260,
        xaxis=dict(gridcolor="#26263a"), yaxis=dict(gridcolor="#26263a"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)


# ─── DATASET EXPLORER ────────────────────────────────────────────────────────
elif page_name == "Dataset Explorer":
    st.markdown('<div class="hero-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Explore and filter the raw loan application data</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        emp_filter = st.multiselect("Employment Status", df_raw["Employment_Status"].dropna().unique(), default=list(df_raw["Employment_Status"].dropna().unique()))
    with col2:
        approval_filter = st.multiselect("Loan Approved", df_raw["Loan_Approved"].dropna().unique(), default=list(df_raw["Loan_Approved"].dropna().unique()))
    with col3:
        income_range = st.slider("Applicant Income Range",
                                  int(df_raw["Applicant_Income"].min()),
                                  int(df_raw["Applicant_Income"].max()),
                                  (int(df_raw["Applicant_Income"].min()), int(df_raw["Applicant_Income"].max())))

    filtered = df_raw[
        df_raw["Employment_Status"].isin(emp_filter) &
        df_raw["Loan_Approved"].isin(approval_filter) &
        df_raw["Applicant_Income"].between(*income_range)
    ]

    st.markdown(f'<div class="tip-box">Showing <b>{len(filtered):,}</b> of {len(df_raw):,} records</div>', unsafe_allow_html=True)
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=400)

    st.markdown('<div class="section-title">Feature Correlations with Loan Approval</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    num_cols_raw = df_raw.select_dtypes(include="number").columns.tolist()
    corr_data = []
    for col in num_cols_raw:
        if col not in ["Applicant_ID"]:
            corr_val = df[[col] + [c for c in df.columns if "Loan_Approved" in c]].corr().iloc[0, -1] if col in df.columns else 0
            corr_data.append({"Feature": col, "Correlation": corr_val})
    corr_df = pd.DataFrame(corr_data).sort_values("Correlation")

    fig = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                 color="Correlation", color_continuous_scale=["#ff6b6b", "#26263a", "#43e97b"],
                 template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#9090a8"),
        margin=dict(l=0, r=0, t=10, b=0), height=350,
        xaxis=dict(gridcolor="#26263a"), yaxis=dict(gridcolor="#26263a"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── MODEL PERFORMANCE ───────────────────────────────────────────────────────
elif page_name == "Model Performance":
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Compare Logistic Regression, Naive Bayes, and KNN classifiers</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Model comparison cards
    cols = st.columns(3)
    icons = {"Logistic Regression": "📈", "Naive Bayes": "🧮", "KNN": "🔵"}
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            border_col = "#6c63ff" if name == "Logistic Regression" else "#26263a"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {border_col};">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icons[name]}</div>
                <div class="metric-label">{name}</div>
                <div class="metric-value">{res['Accuracy']}%</div>
                <div class="metric-sub">Accuracy</div>
                <div style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div style="font-size: 0.78rem;">
                        <div style="color: #6b6b80;">Precision</div>
                        <div style="color: #e8e8f0; font-weight: 600;">{res['Precision']}%</div>
                    </div>
                    <div style="font-size: 0.78rem;">
                        <div style="color: #6b6b80;">Recall</div>
                        <div style="color: #e8e8f0; font-weight: 600;">{res['Recall']}%</div>
                    </div>
                    <div style="font-size: 0.78rem;">
                        <div style="color: #6b6b80;">F1 Score</div>
                        <div style="color: #e8e8f0; font-weight: 600;">{res['F1']}%</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">Metrics Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1"]]
        fig = px.bar(metrics_df, barmode="group",
                     color_discrete_sequence=["#6c63ff", "#43e97b", "#ff6b6b", "#ffd166"],
                     template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#9090a8"),
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            xaxis=dict(gridcolor="#26263a"), yaxis=dict(gridcolor="#26263a", range=[0, 100]),
            legend=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        selected_model = st.selectbox("Select Model", list(results.keys()), label_visibility="collapsed")
        cm = results[selected_model]["CM"]
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted", y="Actual"),
                           x=["Rejected", "Approved"], y=["Rejected", "Approved"],
                           color_continuous_scale=["#111118", "#6c63ff"],
                           template="plotly_dark")
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#9090a8"),
            margin=dict(l=0, r=0, t=10, b=0), height=280,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ─── LOAN PREDICTOR ──────────────────────────────────────────────────────────
elif page_name == "Loan Predictor":
    st.markdown('<div class="hero-title">Loan Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter applicant details to get an instant eligibility decision</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-title">💰 Financial Info</div><div class="section-divider"></div>', unsafe_allow_html=True)
            applicant_income = st.number_input("Applicant Income (₹)", min_value=0, max_value=100000, value=15000, step=500)
            coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, max_value=50000, value=5000, step=500)
            savings = st.number_input("Savings (₹)", min_value=0, max_value=100000, value=10000, step=1000)
            collateral_value = st.number_input("Collateral Value (₹)", min_value=0, max_value=100000, value=25000, step=1000)
            dti_ratio = st.slider("DTI Ratio", 0.1, 0.6, 0.3, 0.01)

        with col2:
            st.markdown('<div class="section-title">📋 Loan Details</div><div class="section-divider"></div>', unsafe_allow_html=True)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, max_value=100000, value=20000, step=1000)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
            loan_purpose = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
            credit_score = st.slider("Credit Score", 550, 800, 680)
            existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3, 4])

        with col3:
            st.markdown('<div class="section-title">👤 Personal Info</div><div class="section-divider"></div>', unsafe_allow_html=True)
            age = st.slider("Age", 21, 59, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
            employer_category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Self-employed", "Unemployed"])
            dependents = st.selectbox("Dependents", [0, 1, 2, 3])
            property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Check Loan Eligibility")

    if submitted:
        # Build feature vector matching training
        input_data = {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age": age,
            "Dependents": dependents,
            "Existing_Loans": existing_loans,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Education_Level": 1 if education_level == "Graduate" else 0,
            "Employment_Status_Self-employed": 1 if employment_status == "Self-employed" else 0,
            "Employment_Status_Salaried": 1 if employment_status == "Salaried" else 0,
            "Employment_Status_Unemployed": 1 if employment_status == "Unemployed" else 0,
            "Marital_Status_Single": 1 if marital_status == "Single" else 0,
            "Loan_Purpose_Car": 1 if loan_purpose == "Car" else 0,
            "Loan_Purpose_Education": 1 if loan_purpose == "Education" else 0,
            "Loan_Purpose_Home": 1 if loan_purpose == "Home" else 0,
            "Loan_Purpose_Personal": 1 if loan_purpose == "Personal" else 0,
            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
            "Property_Area_Urban": 1 if property_area == "Urban" else 0,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Employer_Category_Government": 1 if employer_category == "Government" else 0,
            "Employer_Category_MNC": 1 if employer_category == "MNC" else 0,
            "Employer_Category_Private": 1 if employer_category == "Private" else 0,
            "Employer_Category_Unemployed": 1 if employer_category == "Unemployed" else 0,
            "DTI_Ratio_sq": dti_ratio ** 2,
            "Credit_Score_sq": credit_score ** 2,
            "Applicant_Income_log": np.log1p(applicant_income),
        }

        # Align with training columns
        input_df = pd.DataFrame([input_data])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        input_scaled = scaler.transform(input_df)

        st.markdown("<br>", unsafe_allow_html=True)
        col_r, col_d = st.columns([1, 1.5])

        with col_r:
            # Best model is Logistic Regression (F1: 80.9%, Accuracy: 88%)
            model = trained_models["Logistic Regression"]
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            approval_prob = prob[1] * 100
            rejection_prob = prob[0] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="result-approved">
                    <div class="result-emoji">✅</div>
                    <div class="result-title approved">APPROVED</div>
                    <div class="result-prob">Approval probability: <b style="color:#43e97b">{approval_prob:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    <div class="result-emoji">❌</div>
                    <div class="result-title rejected">NOT APPROVED</div>
                    <div class="result-prob">Rejection probability: <b style="color:#ff6b6b">{rejection_prob:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=approval_prob,
                number={"suffix": "%", "font": {"family": "Syne", "size": 28, "color": "#e8e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b6b80"},
                    "bar": {"color": "#43e97b" if pred == 1 else "#ff6b6b", "thickness": 0.35},
                    "bgcolor": "#18181f",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(255,107,107,0.1)"},
                        {"range": [40, 70], "color": "rgba(255,209,102,0.1)"},
                        {"range": [70, 100], "color": "rgba(67,233,123,0.1)"},
                    ],
                    "threshold": {"line": {"color": "#6c63ff", "width": 3}, "thickness": 0.9, "value": 50},
                },
                title={"text": "Approval Score", "font": {"family": "DM Sans", "color": "#6b6b80", "size": 14}},
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=220,
                margin=dict(l=20, r=20, t=30, b=0),
                font=dict(family="DM Sans"),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with col_d:
            st.markdown('<div class="section-title">Key Risk Factors</div><div class="section-divider"></div>', unsafe_allow_html=True)

            # Display key factors with visual bars
            factors = {
                "Credit Score": (credit_score - 550) / (800 - 550) * 100,
                "DTI Ratio (lower=better)": (1 - dti_ratio / 0.6) * 100,
                "Income": min((applicant_income + coapplicant_income) / 30000 * 100, 100),
                "Savings": min(savings / 20000 * 100, 100),
                "Loan-to-Income": max(0, 100 - (loan_amount / max(applicant_income, 1) * 10)),
                "Collateral Coverage": min(collateral_value / max(loan_amount, 1) * 50, 100),
            }

            for label, val in factors.items():
                color = "#43e97b" if val >= 60 else "#ffd166" if val >= 40 else "#ff6b6b"
                st.markdown(f"""
                <div class="factor-row">
                    <div class="factor-label">{label}</div>
                    <div class="factor-bar-wrap">
                        <div class="factor-bar-fill" style="width:{val:.0f}%; background:{color};"></div>
                    </div>
                    <div class="factor-val" style="color:{color};">{val:.0f}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # All models comparison
            st.markdown('<div class="section-title">All Models Prediction</div><div class="section-divider"></div>', unsafe_allow_html=True)
            model_preds = {}
            for mname, m in trained_models.items():
                p = m.predict(input_scaled)[0]
                pr = m.predict_proba(input_scaled)[0][1] * 100
                model_preds[mname] = {"result": "✅ Approved" if p == 1 else "❌ Rejected", "prob": pr}

            for mname, v in model_preds.items():
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"<span style='font-size:0.9rem; font-weight:500;'>{mname}</span>", unsafe_allow_html=True)
                with col_b:
                    color = "#43e97b" if "Approved" in v["result"] else "#ff6b6b"
                    st.markdown(f"<span style='color:{color}; font-size:0.88rem; font-weight:600;'>{v['result']} ({v['prob']:.1f}%)</span>", unsafe_allow_html=True)

        # Tips
        st.markdown("<br>", unsafe_allow_html=True)
        tips = []
        if credit_score < 650:
            tips.append("📈 Improve your credit score above 650 — it's the strongest approval predictor.")
        if dti_ratio > 0.4:
            tips.append("💸 Your DTI ratio is high. Reducing existing debt before applying could help.")
        if savings < 5000:
            tips.append("🏦 Build up savings — it signals financial stability to lenders.")
        if existing_loans >= 3:
            tips.append("⚠️ Having 3+ existing loans significantly reduces approval chances.")
        if tips:
            st.markdown(f'<div class="tip-box">💡 <b>Improvement Tips:</b><br>' + "<br>".join(tips) + '</div>', unsafe_allow_html=True)
