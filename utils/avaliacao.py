import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def avaliar_modelo(Y_test, Y_pred):
    """
    Exibe o relatório de classificação e gera o gráfico da matriz de confusão.
    """
    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
    print(classification_report(Y_test, Y_pred, target_names=['Normal', 'Fraude/Anomalia']))

    # Gerando a Matriz de Confusão
    cm = confusion_matrix(Y_test, Y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Reds',
        xticklabels=['Normal', 'Anomalia'],
        yticklabels=['Normal', 'Anomalia']
    )
    plt.title('Matriz de Confusão - Isolation Forest')
    plt.xlabel('Predição do Modelo')
    plt.ylabel('Realidade (Ground Truth)')
    plt.tight_layout()
    plt.show()