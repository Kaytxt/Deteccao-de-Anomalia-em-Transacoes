def gerar_prompt_explicabilidade(transacao_id, dados_transacao, score_anomalia):
    """
    Constrói um prompt estruturado para ser enviado a um modelo de GenAI (ex: Gemini / GPT),
    transformando dados técnicos de uma anomalia em um relatório legível para triagem de segurança.
    """
    
    prompt = f"""
    ======================================================================
    [RELATÓRIO DE TRIAGEM AUTOMÁTICA DE CIBERSEGURANÇA E FRAUDES - SOC]
    ======================================================================
    
    ALERT: Transação Suspeita Detectada!
    ID da Transação: #{transacao_id}
    Score de Anomalia (Isolation Forest): {score_anomalia:.4f}
    
    DADOS COMPORTAMENTAIS DA TRANSAÇÃO (Features Normalizadas):
    - Valor da Transação (Amount Normalizado): {dados_transacao.get('scaled_amount', 'N/A'):.4f}
    - Horário Relativo (Time Normalizado): {dados_transacao.get('scaled_time', 'N/A'):.4f}
    - Componentes Principais com maior desvio:
      * V1: {dados_transacao.get('V1', 0):.4f}
      * V2: {dados_transacao.get('V2', 0):.4f}
      * V3: {dados_transacao.get('V3', 0):.4f}
      * V4: {dados_transacao.get('V4', 0):.4f}
    
    ----------------------------------------------------------------------
    INSTRUÇÕES PARA O AGENTE DE IA GENERATIVA (LLM):
    Atue como um Especialista Sênior em Prevenção a Fraudes e Cibersegurança.
    Com base nos dados fornecidos acima:
    1. Forneça uma explicação concisa de 2 parágrafos sobre o porquê desta transação ter comportamento atípico.
    2. Defina o nível de severidade (BAIXO, MÉDIO, ALTO ou CRÍTICO).
    3. Recomende 2 ações imediatas para a equipe de operações (ex: Bloqueio temporário do cartão, envio de token biométrico ao aplicativo do cliente).
    ======================================================================
    """
    return prompt


def simular_triagem_genai(X_test, y_pred, modelo, quantidade=2):
    """
    Seleciona as principais anomalias detectadas pelo modelo e gera
    os prompts estruturados para triagem com GenAI.
    """
    # Filtra os índices que o modelo classificou como Anomalia (1)
    indices_anomalias = X_test[y_pred == 1].index
    
    prompts_gerados = []
    
    print(f"\n=== GENERATING GENAI AGENT PROMPTS ({quantidade} EXEMPLOS SELECIONADOS) ===")
    
    # Pega as N primeiras anomalias para demonstração
    for i in range(min(quantidade, len(indices_anomalias))):
        idx = indices_anomalias[i]
        
        # Obtém o score de anomalia do Isolation Forest
        score = modelo.score_samples(X_test.loc[[idx]])[0]
        
        # Extrai os dados da transação em dicionário
        dados_transacao = X_test.loc[idx].to_dict()
        
        # Gera o prompt estruturado
        prompt = gerar_prompt_explicabilidade(idx, dados_transacao, score)
        prompts_gerados.append(prompt)
        
        print(prompt)
        
    return prompts_gerados