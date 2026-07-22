def verificarAusentes(df):
    """
    Essa função ira te retornar quantos valores na tabela são nulos ou estão ausentes
    """
    print("Valores ausentes:", df.isnull().sum().sum())
