import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Banker AI - Credit Scoring", 
    page_icon="🏦",
    layout="centered"
)

# --- LOAD MODEL FUNCTION ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('credit_risk_model.pkl')
        encoder = joblib.load('sex_encoder.pkl')
        return model, encoder
    except:
        return None, None

model, sex_encoder = load_model()

# --- HEADER ---
st.title("🏦 AI Credit Risk Scoring")
st.markdown("""
This system uses **Machine Learning** to assess creditworthiness based on applicant data.
Values update in real-time based on your profile.
""")
st.divider()

st.subheader("📝 Applicant Details")

# --- INTERFACE REATIVA (SEM st.form) ---
# Usamos colunas, mas sem o bloqueio do formulário para permitir atualizações dinâmicas

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    
    job_display = {
        0: "0 - Unskilled and Non-resident",
        1: "1 - Unskilled Resident",
        2: "2 - Skilled",
        3: "3 - Highly Qualified"
    }
    
    # O user escolhe aqui e o Python lê IMEDIATAMENTE
    job = st.selectbox(
        "Job Skill Level", 
        options=[0, 1, 2, 3], 
        format_func=lambda x: job_display[x]
    )

# --- CÁLCULO DINÂMICO DO LIMITE (EM TEMPO REAL) ---
# O código corre isto sempre que mudas a idade ou job
if job <= 1:
    current_limit = 10000
elif job == 2:
    current_limit = 50000
else:
    current_limit = 100000

# Aplica penalização de idade
if age < 21:
    current_limit = current_limit * 0.5
    limit_msg = f"€ {current_limit:,.0f} (Age < 21 Limit Applied)"
else:
    limit_msg = f"€ {current_limit:,.0f}"

# --- CONTINUAÇÃO DOS INPUTS ---
with col1:
    amount = st.number_input("Credit Amount (€)", min_value=500, value=2000, step=100)
    
    # AQUI ESTÁ A MAGIA: O texto muda sozinho
    if amount > current_limit:
        st.caption(f"⚠️ **Limit Exceeded!** Max for this profile: :red[{limit_msg}]")
    else:
        st.caption(f"ℹ️ Max allowed limit: :green[{limit_msg}]")

with col1:
    duration = st.slider("Duration (Months)", 6, 72, 24)

st.markdown("<br>", unsafe_allow_html=True)

# Botão normal (fora de form)
submit_button = st.button("Analyze Risk Profile", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if submit_button:
    
    # 1. VALIDAÇÃO DO PEDIDO (HARD RULES)
    # Usamos o 'current_limit' que já foi calculado lá em cima
    if amount > current_limit:
        st.error("❌ AUTO-DECLINED (Business Policy)")
        
        st.markdown(f"""
        The requested amount (**€{amount:,.0f}**) exceeds the maximum limit for this profile.
        
        Based on the applicant's **Job Level** and **Age**, the maximum allowed is:
        # **€ {current_limit:,.0f}**
        """)
        
        # Para o código aqui
        st.stop() 
    
    # 2. VERIFICAÇÕES TÉCNICAS
    if model is None:
        st.error("🚨 Error: Model file not found.")
        st.stop()
        
    # 3. INTELIGÊNCIA ARTIFICIAL
    sex_encoded = sex_encoder.transform([sex])[0]
    features = pd.DataFrame([[age, amount, duration, sex_encoded, job]], 
                            columns=['Age', 'Credit amount', 'Duration', 'Sex', 'Job'])
    
    with st.spinner("Analyzing risk profile..."):
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
    credit_score = int(850 - (probability * 550))
    
    # 4. MOSTRAR RESULTADOS
    st.divider()
    st.subheader("📊 Analysis Results")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Credit Score", credit_score)
    m2.metric("Default Probability", f"{probability:.1%}")
    
    if prediction == 1:
        m3.error("Declined")
        st.error("❌ CREDIT DECLINED")
        st.write(f"The model detected high risk patterns ({probability:.1%}).")
    else:
        m3.success("Approved")
        st.success("✅ CREDIT APPROVED")
        st.write("The applicant fits the risk profile.")
  