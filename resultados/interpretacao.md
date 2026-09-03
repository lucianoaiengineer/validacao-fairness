# Interpretação dos resultados

## Qualidade dos dados

As validações de schema, tipos, colunas críticas e faixas foram aprovadas para
os conjuntos de treino e teste. As ausências observadas ficaram restritas às
colunas permitidas e foram tratadas no pré-processamento por imputação
categórica com valor mais frequente.

- Treino: classe_trabalho=1836, ocupacao=1843, pais_origem=583
- Teste: classe_trabalho=963, ocupacao=966, pais_origem=274

Foram encontrados 24 registros totalmente duplicados no treino
e 5 no teste. Como o dataset não possui chave natural
confiável, o tratamento adotado foi remover apenas duplicados totais antes da
modelagem, sem criar regra artificial de unicidade.

## Desempenho

O modelo principal, uma regressão logística com pré-processamento em pipeline,
obteve acurácia de 0.8512 no conjunto de teste oficial
após validação e remoção de duplicados totais. O modelo de referência
`DummyClassifier` com estratégia `most_frequent` obteve acurácia de
0.7637.

O intervalo de confiança bootstrap de 95% para a acurácia do modelo principal
foi [0.8454, 0.8568], com
2000 reamostragens. Esse intervalo expressa a incerteza
amostral da métrica no conjunto de teste, não uma garantia sobre populações
futuras.

Na comparação bootstrap pareada, a diferença pontual de acurácia
(regressão logística menos referência) foi 0.0875,
com intervalo de 95% [0.0810,
0.0941]. Como o intervalo ficou totalmente acima de zero, há evidência de desempenho superior do modelo principal.

## Fairness

A paridade demográfica foi avaliada sobre as predições do conjunto de teste
oficial, considerando 16276 registros utilizados. As taxas de
predições positivas por grupo foram:

- Feminino: 0.0768 (416/5420)
- Masculino: 0.2448 (2658/10856)

A taxa positiva do grupo Masculino foi superior à do grupo Feminino em
aproximadamente 16.81 pontos percentuais.

O gap absoluto de paridade demográfica foi 0.1681. Perante o
limite operacional de 0.10, adotado nesta análise e não
como regra legal ou universal, a paridade demográfica foi classificada como
**não atendida**.

## Limitações

O dataset Adult é histórico, derivado de dados censitários dos Estados Unidos,
e reflete condições sociais, econômicas e de coleta do período em que foi
produzido. A exclusão direta de `sexo` e `raca` dos preditores não garante
ausência de disparidade, pois outras variáveis podem funcionar como proxies.

Diferenças observadas nas taxas positivas indicam disparidade estatística sob
o critério operacional adotado, mas não provam causalidade, discriminação
individual ou intenção discriminatória.
