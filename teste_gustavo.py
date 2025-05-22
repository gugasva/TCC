import pandas as pd
df = pd.read_csv("dados_treinamento.csv")
print(df['rotulo'].value_counts())
