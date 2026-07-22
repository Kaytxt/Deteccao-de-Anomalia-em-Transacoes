import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    verificarAusentes,
    proporcaoFraude,
    preparar_dados,
    treinar_isolation_forest,
    prever_anomalias,
    avaliar_modelo,
    simular_triagem_genai
)

sns.set_theme(style="whitegrid")

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print('Analise Exploratoria:')
verificarAusentes(df)
proporcaoFraude(df)

print('\nPré Processando os Dados:')
X_train, X_test, Y_train, Y_test = preparar_dados(df)
print(f"Dados de Treino: {X_train.shape[0]}, transações")
print(f"Dados de Teste: {X_test.shape[0]} transações")

print('\nTreinando o Isolation Forest:')
modelo = treinar_isolation_forest(X_train, Y_train)
print("Modelo Treinado")

print('\nExecutando Predições no Conjunto de Teste')
Y_pred = prever_anomalias(modelo, X_test)

print(f'Total de anomalias detectadas pelo modelo no teste: {sum(Y_pred)}')

print('\nAvaliação de Desempenho')
avaliar_modelo(Y_test, Y_pred)

simular_triagem_genai(X_test, Y_pred, modelo, quantidade=2)