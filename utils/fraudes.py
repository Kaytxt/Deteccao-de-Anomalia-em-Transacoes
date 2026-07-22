def proporcaoFraude(df):
    """
    Função utilizada para verificar a proporção real das fraudes
    """
    contagem_classes = df['Class'].value_counts()
    proporcao = df['Class'].value_counts(normalize=True)*100

    print(f"Transações Normais (0): {contagem_classes[0]} ({proporcao[0]:.2f}%)")
    print(f"Transações Anômalas (1): {contagem_classes[1]} ({proporcao[1]:.2f}%)")