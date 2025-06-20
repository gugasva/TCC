import pandas as pd
from db_conexao import conexao
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Carrega dados do banco
query = "SELECT * FROM dados_treinamento"
df = pd.read_sql(query, conexao)

# Prepara os dados
X = df.drop(columns=['id', 'rotulo'])
y = df['rotulo']

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avalia
y_pred = modelo.predict(X_test)
print("\n--- Avaliação ---")
print(classification_report(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

# Salva o modelo
joblib.dump(modelo, 'modelo_treinado.pkl')
print("\n✅ Modelo salvo como modelo_treinado.pkl")
