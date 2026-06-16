<div align="center">

# 💳 CreditWise — ML Loan Approval System
## 📸 Preview

| Dashboard | Loan Predictor |
|-----------|---------------|
| KPI cards, charts, approval trends | Real-time eligibility with risk factor bars |

---

## 🎯 Project Overview

CreditWise is a complete ML pipeline that takes raw loan application data, processes and engineers features, trains multiple classification models, and serves predictions through a polished interactive web interface.

The system evaluates applicants across 28 engineered features and outputs an approval decision with a probability score, helping understand the key factors driving the result.

---

## 🚀 Features

- **4-page interactive dashboard** built with Streamlit
- **Real-time loan prediction** with approval probability gauge
- **Risk factor analysis** with visual bar indicators per applicant
- **Model performance comparison** — Logistic Regression vs Naive Bayes vs KNN
- **Dataset explorer** with filters and correlation analysis
- **Improvement tips** shown when a loan is rejected

---

## 🧠 ML Pipeline

### Dataset
- 1,000 loan applications, 19 raw features
- Features: income, credit score, DTI ratio, savings, collateral, loan amount/term, employment, demographics

### Preprocessing
- Missing value imputation (mean for numerical, mode for categorical)
- Label encoding for ordinal features
- One-hot encoding for nominal features
- StandardScaler for feature normalization

### Feature Engineering
- `DTI_Ratio_sq` — squared DTI ratio to capture non-linear risk
- `Credit_Score_sq` — squared credit score for non-linear reward
- `Applicant_Income_log` — log transform to reduce income skew
- Total: **28 engineered features**

### Models Trained

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| **Logistic Regression** ⭐ | 78.5% | 83.6% | **80.9%** | **88.0%** |
| Naive Bayes | 81.1% | 70.5% | 75.4% | 86.0% |
| KNN (k=5) | 67.3% | 57.4% | 61.9% | 78.5% |

**Best Model: Logistic Regression** — chosen based on F1 Score, which balances both Precision and Recall. In a loan approval system, both false approvals and missed good applicants carry real costs, making F1 the most appropriate metric.

---

## 🗂️ Project Structure

```
credit-risk-streamlit-app/
│
├── app.py                          # Main Streamlit application
├── loan_approval_data.csv          # Raw dataset
├── CreditWise - Loan Approval System.ipynb  # Full ML notebook
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## 🖥️ App Pages

### 🏠 Dashboard
- Total applications, approval rate, average income, best model F1 score
- Income distribution by approval status
- Employment status breakdown (pie chart)
- Credit score vs approval rate bar chart

### 📊 Dataset Explorer
- Filter by employment status, approval status, income range
- Full interactive data table
- Feature correlation with loan approval

### 🤖 Model Performance
- Side-by-side model metric cards
- Grouped bar chart comparing all metrics
- Interactive confusion matrix per model

### 🔮 Loan Predictor
- Full applicant form (financial info, loan details, personal info)
- Instant approval/rejection result
- Probability gauge chart
- Key risk factor visual bars
- All-models prediction comparison
- Personalized improvement tips

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/paramramit305-a11y/credit-risk-streamlit-app.git
cd credit-risk-streamlit-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.13.0
```

---

## 📊 Key Insights from EDA

- **Credit Score** is the strongest predictor of approval (correlation: +0.45)
- **DTI Ratio** is the strongest negative predictor (correlation: -0.44)
- **Loan Amount** and **Loan Term** also negatively impact approval chances
- Applicants with credit scores above 700 have significantly higher approval rates

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | ML models & preprocessing |
| Plotly | Interactive charts |
| Streamlit | Web app framework & deployment |
| GitHub | Version control |
| Streamlit Cloud | Free hosting & deployment |

---

## 🔮 Future Improvements

- [ ] Add XGBoost / Random Forest for improved accuracy
- [ ] Integrate SHAP values for explainability
- [ ] Add user authentication for personalized history
- [ ] REST API with FastAPI for programmatic access
- [ ] Deploy on AWS / GCP for production scale

---

