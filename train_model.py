# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. CARREGAR DADOS
print("⚠️ A gerar dados simulados para teste...")
data = {
    'Age': np.random.randint(18, 70, 1000),
    'Credit amount': np.random.randint(500, 10000, 1000),
    'Duration': np.random.randint(6, 48, 1000), # Meses
    'Sex': np.random.choice(['male', 'female'], 1000),
    'Job': np.random.choice([0, 1, 2, 3], 1000), # Nível de qualificação
    'Risk': np.random.choice([0, 1], 1000) # 0 = Bom, 1 = Mau 
}
df = pd.DataFrame(data)

print("✅ Dados carregados. A processar...")

# 2. PRÉ-PROCESSAMENTO (Limpeza)
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

X = df[['Age', 'Credit amount', 'Duration', 'Sex', 'Job']]
y = df['Risk']

# 3. DIVIDIR EM TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TREINAR O MODELO (Random Forest)
print("🤖 A treinar o modelo de Risco...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. AVALIAR
accuracy = model.score(X_test, y_test)
print(f"🎯 Precisão do Modelo: {accuracy:.2%}")

# 6. GUARDAR O MODELO (Para usar na App)
joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(le_sex, 'sex_encoder.pkl')

print("💾 Modelo guardado como 'credit_risk_model.pkl'. Pronto para a App!")