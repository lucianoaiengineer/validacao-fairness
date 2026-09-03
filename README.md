# Validação de Dados e Fairness

Este projeto implementa uma suíte que integra validação de dados, testes
estatísticos e avaliação de fairness aplicada ao dataset público Adult,
também conhecido como Census Income.

## Objetivo

O objetivo deste projeto é validar a qualidade e a integridade do dataset
Adult, treinar e avaliar um modelo de classificação com evidências estatísticas
e verificar se suas predições apresentam disparidade entre os grupos Feminino e
Masculino. A solução reúne validações automatizadas, comparação com um modelo
de referência, intervalos de confiança e análise de paridade demográfica,
produzindo resultados reproduzíveis e interpretáveis.

O projeto baixa os arquivos oficiais `adult.data` e `adult.test`, prepara os
dados, treina uma regressão logística e gera suas próprias predições para o
conjunto de teste oficial. A avaliação de fairness usa o atributo sensível
`sexo`, com os grupos Feminino e Masculino, somente na etapa de análise. As
variáveis `sexo` e `raca` são excluídas dos preditores do modelo.

## Fonte dos dados

- Dataset: UCI Adult / Census Income
- Página oficial: <https://archive.ics.uci.edu/dataset/2/adult>
- DOI: <https://doi.org/10.24432/C5XW20>
- Licença indicada pelo UCI Machine Learning Repository: CC BY 4.0

Os dados brutos ficam em `data/brutos/` e não são versionados. A separação
oficial entre treinamento (`adult.data`) e teste (`adult.test`) é preservada.

## Metodologia

Validação de dados:

- presença das colunas obrigatórias;
- tipos esperados;
- dataset não vazio;
- ausência de valores faltantes inesperados fora das colunas permitidas;
- domínio binário do rótulo;
- domínio esperado de `sexo`;
- faixas válidas para idade, nível de escolaridade, horas semanais e capitais;
- contabilização e remoção documentada de registros totalmente duplicados.

Testes estatísticos:

- acurácia do modelo principal;
- intervalo de confiança bootstrap de 95% para a acurácia;
- comparação bootstrap pareada entre regressão logística e `DummyClassifier`;
- 2.000 reamostragens com semente determinística 42.

Fairness:

- paridade demográfica sobre as predições do conjunto de teste;
- taxas positivas para Feminino e Masculino;
- gap absoluto entre as taxas;
- limite operacional de 0,10, adotado nesta análise e não como regra legal
  ou universal.

A exclusão direta de `sexo` e `raca` não garante ausência de disparidade, pois
outras variáveis podem funcionar como proxies.

## Estrutura

```text
data/                    Dados brutos locais e documentação da origem
resultados/              Métricas e interpretação geradas pela análise
scripts/                 Comandos de download e execução
src/validacao_fairness/  Código da suíte
tests/                   Testes automatizados independentes da rede
```

## Ambiente

Use Python 3.14 ou superior. Após clonar o repositório, prepare o ambiente com:

```bash
git clone https://github.com/lucianoaiengineer/validacao-fairness.git
cd validacao-fairness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Download

```bash
python scripts/baixar_dados.py
```

O download usa apenas HTTPS a partir de `archive.ics.uci.edu` e reutiliza os
arquivos existentes.

## Execução

```bash
python scripts/executar_analise.py
```

A execução gera:

- `resultados/metricas.json`;
- `resultados/equidade_por_grupo.csv`;
- `resultados/interpretacao.md`.

## Qualidade

```bash
ruff check .
mypy src
pytest --cov=validacao_fairness --cov-report=term-missing --cov-fail-under=85
```

## Resultados reais

Os resultados apresentados nesta seção correspondem à última execução real da
análise e também estão disponíveis nos arquivos gerados em `resultados/`.

Última execução local:

- treino bruto: 32.561 registros;
- teste bruto: 16.281 registros;
- treino utilizado após remoção de duplicados totais: 32.537 registros;
- teste utilizado após remoção de duplicados totais: 16.276 registros;
- acurácia da regressão logística: 0,8512;
- IC bootstrap 95% da acurácia: [0,8454, 0,8568];
- acurácia do `DummyClassifier`: 0,7637;
- diferença pareada de acurácia: 0,0875;
- IC bootstrap 95% da diferença: [0,0810, 0,0941];
- taxa positiva Feminino: 0,0768;
- taxa positiva Masculino: 0,2448;
- gap de paridade demográfica: 0,1681;
- status perante o limite operacional de 0,10: não atendida.

## Interpretação e limitações

As validações confirmaram que os conjuntos de treino e teste atendem ao schema,
aos tipos e às faixas definidos. Os valores ausentes ficaram restritos às
colunas previstas e foram tratados durante o pré-processamento. Também foram
removidos 24 registros totalmente duplicados no treino e 5 no teste.

A regressão logística alcançou acurácia de 0,8512, acima dos 0,7637 obtidos pelo
modelo de referência. O intervalo bootstrap pareado de 95% para a diferença
entre os modelos, [0,0810, 0,0941], permaneceu acima de zero, indicando
desempenho estatisticamente superior do modelo principal no conjunto analisado.

Na avaliação de fairness, a taxa de predições positivas foi de 0,0768 para o
grupo Feminino e de 0,2448 para o grupo Masculino. O gap de 0,1681 ultrapassou
o limite operacional de 0,10; por isso, a paridade demográfica foi classificada
como não atendida.

O dataset Adult é histórico e reflete condições sociais e econômicas do período
em que foi coletado. A disparidade observada não demonstra, isoladamente,
causalidade, discriminação individual ou intenção discriminatória. Além disso,
a exclusão de `sexo` e `raca` dos preditores não impede que outras variáveis
funcionem como proxies.
