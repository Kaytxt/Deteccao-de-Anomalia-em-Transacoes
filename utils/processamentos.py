from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preparar_dados(df):
    """
    Escalona Time e Amount e divide os dados em X_train, X_test, y_train, y_test.
    """
    df_clean = df.copy()
    scaler = StandardScaler()
    
    # Ajusta a escala das colunas amount e time
    df_clean['scaled_amount'] = scaler.fit_transform(df_clean[['Amount']])
    df_clean['scaled_time'] = scaler.fit_transform(df_clean[['Time']])
    
    # Remove as colunas originais sem escala
    df_clean = df_clean.drop(['Time', 'Amount'], axis=1)
    
    # Separa os recursos X do rotulo Y
    X = df_clean.drop('Class', axis=1)
    Y = df_clean['Class']
    
    # Divide o treino e o teste mantendo a proporção exata de fraudes (treino 80% teste 20%)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    return X_train, X_test, Y_train, Y_test