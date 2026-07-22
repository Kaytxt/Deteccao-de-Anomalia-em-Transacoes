import numpy as np
from sklearn.ensemble import IsolationForest

def treinar_isolation_forest(X_train, Y_train):
    """
    Treina um algoritmo Isolation Forest ajustado com base na taxa de contaminação dos dados
    """
    
    # Calcula a proporção de anomalias no conjunto de treino
    taxa_contaminacao = Y_train.mean()
    
    # Instancia o Isolation Forest
    modelo = IsolationForest(
        n_estimators=150,
        contamination=taxa_contaminacao,
        random_state=42,
        n_jobs=1
    )
    
    # treina o modelo
    modelo.fit(X_train)
    
    return modelo

def prever_anomalias(modelo, X_test):
    """
    Faz as predições no conjunto de teste e converte a saida do Isolation Forest
    para o padrão do dataset: 1 para Anomalia/Fraude e 0 para o Normal.
    """
    # O isolation Fortes retorna -1 para anomalia e 1 para normal
    predicoes_brutas = modelo.predict(X_test)
    
    # se for -1 vira 1, se for 1 vira 0 
    Y_pred = np.where(predicoes_brutas == -1, 1, 0)
    
    return Y_pred