import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Carregar dados
csv_path = 'dados_treinamento.csv'
dados = pd.read_csv(csv_path)

X = dados[[
    'angulo_braco_direito', 'y_punho_direito', 'y_ombro_direito',
    'angulo_braco_esquerdo', 'y_punho_esquerdo', 'y_ombro_esquerdo']]
y = dados['rotulo']

# Dividir dados para treino e validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Salvar modelo treinado
joblib.dump(modelo, 'modelo_treinado.pkl')
print("Modelo treinado e salvo como modelo_treinado.pkl")