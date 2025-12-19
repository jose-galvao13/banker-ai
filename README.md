# 🏦 Banker AI: Credit Risk Scoring Engine

### 📋 Overview
A Machine Learning-powered application designed to automate **credit risk assessment** and streamline loan approval workflows.
Built to bridge the gap between **Traditional Banking Policies** and **Data Science**, this tool combines predictive modeling with dynamic business logic to evaluate borrower creditworthiness in real-time.

### 🚀 Key Features
* **Predictive Risk Engine:** Utilizes a **Random Forest Classifier** (trained on historical credit data) to predict the Probability of Default (PD) for new applicants.
* **Dynamic Business Logic Layer:** Implements hard-coded compliance rules ("Hard Limits") that act as a gateway before the AI model, enforcing lending limits based on **Age**, **Job Qualification**, and **Loan Amount**.
* **Real-Time Decisioning:** Reactive interface where credit limits and risk warnings update instantly as the user modifies input parameters (e.g., age penalizations for applicants < 21).
* **Interpretable Scoring:** Converts raw probabilistic output into a standardized **Credit Score (300-850)** and provides clear "Approved/Declined" actionable verdicts.

### 🛠️ Tech Stack
* **Core:** Python 3.10+
* **Machine Learning:** Scikit-Learn (Random Forest), Joblib
* **Data Processing:** Pandas, NumPy
* **Frontend/UI:** Streamlit (Reactive State Management)
* **Dataset:** German Credit Data (UCI Machine Learning Repository)

---
*Developed by a Management & Finance student at ISCAC - Coimbra Business School.*
