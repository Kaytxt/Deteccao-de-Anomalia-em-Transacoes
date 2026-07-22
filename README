# Detecção de Anomalias em Transações Financeiras com ML e GenAI

Projeto feito para o bootcamp de GenAI, Dados & Cyber da DIO. A ideia é simples de explicar e chata de fazer bem: pegar um dataset real de transações de cartão de crédito, treinar um modelo não-supervisionado pra separar o normal do atípico, e depois usar um LLM pra transformar essas detecções em algo que um analista de SOC leia sem precisar decorar o que é matriz de confusão.

O dataset é o "Credit Card Fraud Detection", disponibilizado pela ULB (Université Libre de Bruxelles) em parceria com a Worldline e hospedado no Kaggle. As variáveis já vêm anonimizadas via PCA (`V1` a `V28`), além de `Amount`, `Time` e o rótulo `Class`. Esse mesmo dataset também aparece no tutorial oficial de classificação com dados desbalanceados da documentação do TensorFlow, que carrega os dados de um espelho hospedado na infraestrutura de armazenamento do Google — mas a fonte e a citação originais continuam sendo ULB/Kaggle. Vale deixar registrado: a proporção de 0,17% de fraudes citada abaixo não é algo que eu calculei — é a taxa conhecida desse dataset específico, e uso ela também pra calibrar o modelo (mais sobre isso na seção de limitações, porque isso tem uma pegadinha metodológica que prefiro admitir do que esconder).

## Estrutura do projeto

```text
DETECÇÃO-DE-ANOMALIA-EM-TRANSAÇÕES/
│
├── utils/
│   ├── __init__.py           # exporta as funções principais do pacote
│   ├── ausentes.py           # checagem de valores nulos
│   ├── fraudes.py            # proporção de classes / desbalanceamento
│   ├── processamentos.py     # StandardScaler + split estratificado
│   ├── modelo.py             # Isolation Forest
│   ├── avaliacao.py          # matriz de confusão e métricas
│   └── agente_genai.py       # monta os prompts de triagem pra um LLM
│
├── .gitattributes
├── main.py
└── README.md
```

## Como o pipeline funciona

A primeira etapa é só olhar os dados: confirmar que não tem valor nulo escondido e medir o desbalanceamento de classes. Com 0,17% de casos positivos, qualquer modelo que simplesmente "chutar tudo normal" já acerta mais de 99% — o que faz da acurácia uma métrica praticamente inútil aqui. Isso guiou a escolha de olhar recall e precision em vez de acurácia desde o início.

Depois vem o pré-processamento: `Amount` e `Time` são padronizados com `StandardScaler` (as variáveis `V1`-`V28` já chegam normalizadas do PCA original) e o split treino/teste é estratificado, pra manter a proporção de fraudes em ambos os conjuntos.

O modelo é um `IsolationForest`. A escolha faz sentido pro problema: em vez de aprender "o que é fraude" (que exigiria muitos exemplos rotulados de uma classe rara), o algoritmo aprende a isolar pontos que se afastam do padrão geral dos dados — que é exatamente a natureza do problema de fraude em cartão.

Na avaliação, o foco é matriz de confusão, recall e precision, evitando a armadilha de reportar acurácia isolada.

Por último, a camada de GenAI: o `agente_genai.py` pega as transações marcadas como anômalas e monta um prompt estruturado com o contexto da transação, pronto pra ser enviado a um LLM (Claude, GPT, etc.) que devolve uma explicação em linguagem natural pro analista humano decidir se escala o caso ou não.

## Uma limitação que prefiro deixar clara

O `contamination` do Isolation Forest foi calibrado usando a taxa real de fraude do dataset — ou seja, uma informação derivada do rótulo verdadeiro (`Class`). Isso é uma inconsistência sutil: o modelo é chamado de não-supervisionado, mas recebe de forma indireta uma estatística que só existe porque os dados já vêm rotulados. Em um cenário de produção real, você não tem esse luxo — teria que estimar a taxa de contaminação via conhecimento de domínio, testar uma faixa de valores, ou usar validação cruzada com alguma amostra rotulada menor. Deixo isso registrado aqui porque é o tipo de pergunta que costuma aparecer em entrevista técnica.

Também vale ser preciso com o termo "GenAI/XAI": o que esse projeto faz não é interpretabilidade de modelo no sentido clássico (SHAP, LIME, feature importance). É geração de linguagem natural a partir do output do modelo — útil pra triagem, mas uma coisa diferente de abrir a caixa-preta do Isolation Forest.

## Como rodar

Pré-requisitos: Python 3.10+.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Execução:

```bash
python main.py
```

## Tecnologias

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn (`IsolationForest`, `StandardScaler`). A camada de explicação é agnóstica de provedor de LLM — o `agente_genai.py` só gera o prompt, quem plugar a API decide qual modelo usar.

## Fonte dos dados

Dataset "Credit Card Fraud Detection", ULB / Worldline, disponível publicamente no Kaggle (também usado como exemplo no tutorial oficial de dados desbalanceados do TensorFlow).
