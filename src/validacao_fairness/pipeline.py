"""
Arquivo: pipeline.py

Orquestra validação, modelagem, estatística, fairness e relatórios.

Fluxo:
1. Carrega dados oficiais locais do Adult.
2. Valida schema, completude, domínios, faixas e duplicados.
3. Remove registros totalmente duplicados antes da modelagem.
4. Treina regressão logística e `DummyClassifier`.
5. Calcula acurácia, IC bootstrap e comparação bootstrap pareada.
6. Calcula paridade demográfica e registra violação sem abortar relatório.
7. Salva JSON, CSV e interpretação Markdown em `resultados`.

Autor: Luciano Magalhães
Setembro 2026
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from validacao_fairness.configuracao import (
    DIRETORIO_RESULTADOS,
    DOI_DATASET,
    LICENCA_DATASET,
    LIMITE_PARIDADE_DEMOGRAFICA,
    NIVEL_CONFIANCA,
    REAMOSTRAGENS_BOOTSTRAP,
    SEMENTE,
    URL_DATASET_UCI,
)
from validacao_fairness.dados import carregar_datasets_adult
from validacao_fairness.equidade import (
    ResultadoEquidade,
    ViolacaoParidadeDemograficaError,
    verificar_paridade_demografica,
)
from validacao_fairness.estatistica import (
    ComparacaoBootstrap,
    IntervaloBootstrap,
    calcular_acuracia,
    comparar_acuracias_bootstrap_pareado,
    intervalo_confianca_bootstrap_acuracia,
)
from validacao_fairness.modelagem import ResultadoModelagem, treinar_e_predizer
from validacao_fairness.relatorio import (
    escrever_interpretacao,
    salvar_equidade_csv,
    salvar_metricas_json,
)
from validacao_fairness.validacao import (
    ResultadoValidacao,
    remover_duplicados_totais,
    validar_dataset,
)


@dataclass(frozen=True)
class ResumoDados:
    """
    Resume volumes e validações dos dados brutos e utilizados.

    Atributos:
        treino_bruto: Quantidade de registros lidos de `adult.data`.
        teste_bruto: Quantidade de registros lidos de `adult.test`.
        treino_utilizado: Quantidade de registros usados após deduplicação.
        teste_utilizado: Quantidade de registros avaliados após deduplicação.
        duplicados_treino_removidos: Duplicados totais removidos do treino.
        duplicados_teste_removidos: Duplicados totais removidos do teste.
        validacao_treino: ResultadoValidacao do conjunto de treino bruto.
        validacao_teste: ResultadoValidacao do conjunto de teste bruto.
    """

    treino_bruto: int
    teste_bruto: int
    treino_utilizado: int
    teste_utilizado: int
    duplicados_treino_removidos: int
    duplicados_teste_removidos: int
    validacao_treino: ResultadoValidacao
    validacao_teste: ResultadoValidacao


@dataclass(frozen=True)
class ResultadoAnalise:
    """
    Consolida todos os objetos produzidos pela análise completa.

    Atributos:
        resumo_dados: Volumes, ausências e duplicados dos dados.
        modelagem: Modelos treinados, rótulos reais e predições.
        acuracia_principal: Acurácia pontual da regressão logística.
        acuracia_referencia: Acurácia pontual do `DummyClassifier`.
        intervalo_principal: IC bootstrap da acurácia principal.
        comparacao: Comparação bootstrap pareada entre os modelos.
        equidade: Resultado de paridade demográfica.
        metricas: Dicionário serializável gravado em `metricas.json`.
    """

    resumo_dados: ResumoDados
    modelagem: ResultadoModelagem
    acuracia_principal: float
    acuracia_referencia: float
    intervalo_principal: IntervaloBootstrap
    comparacao: ComparacaoBootstrap
    equidade: ResultadoEquidade
    metricas: dict[str, Any]


# Orquestração única da análise, deixando etapas específicas nos módulos próprios.
def executar_pipeline(
    diretorio_resultados: Path = DIRETORIO_RESULTADOS,
    reamostragens: int = REAMOSTRAGENS_BOOTSTRAP,
    semente: int = SEMENTE,
) -> ResultadoAnalise:
    """
    Executa a análise completa usando os arquivos Adult locais.

    Parâmetros:
        diretorio_resultados: Diretório onde os artefatos finais serão salvos.
        reamostragens: Quantidade de reamostragens bootstrap.
        semente: Semente determinística aplicada aos modelos e ao bootstrap.

    Retorno:
        ResultadoAnalise com dados resumidos, modelagem, estatística,
        fairness e métricas serializadas.

    Exceções:
        DadosAdultNaoEncontradosError: Propagado quando os arquivos brutos não
        existem.
        ValidacaoDadosError: Propagado quando os dados violam regra crítica.
        ValueError: Propagado quando métricas recebem vetores incompatíveis.
    """
    # Leitura e validação ocorrem antes de qualquer modelagem.
    datasets = carregar_datasets_adult()
    validacao_treino = validar_dataset(datasets.treino, "treino")
    validacao_teste = validar_dataset(datasets.teste, "teste")

    # O Adult não tem chave natural; removemos apenas duplicados totais.
    treino, duplicados_treino = remover_duplicados_totais(datasets.treino)
    teste, duplicados_teste = remover_duplicados_totais(datasets.teste)

    # Treino e predição ficam isolados para manter o pipeline testável.
    modelagem = treinar_e_predizer(treino=treino, teste=teste, semente=semente)

    # Métricas pontuais e incerteza são calculadas sobre o mesmo teste oficial.
    acuracia_principal = calcular_acuracia(
        modelagem.y_teste,
        modelagem.predicoes_principal,
    )
    acuracia_referencia = calcular_acuracia(
        modelagem.y_teste,
        modelagem.predicoes_referencia,
    )
    intervalo = intervalo_confianca_bootstrap_acuracia(
        modelagem.y_teste,
        modelagem.predicoes_principal,
        reamostragens=reamostragens,
        nivel_confianca=NIVEL_CONFIANCA,
        semente=semente,
    )
    comparacao = comparar_acuracias_bootstrap_pareado(
        modelagem.y_teste,
        modelagem.predicoes_principal,
        modelagem.predicoes_referencia,
        reamostragens=reamostragens,
        nivel_confianca=NIVEL_CONFIANCA,
        semente=semente,
    )

    # A função de fairness pode falhar; a análise registra a violação e continua.
    try:
        equidade = verificar_paridade_demografica(
            grupos_sensiveis=list(teste["sexo"].astype(str)),
            predicoes=modelagem.predicoes_principal,
            limite=LIMITE_PARIDADE_DEMOGRAFICA,
            falhar_ao_violar=True,
        )
    except ViolacaoParidadeDemograficaError as erro:
        equidade = erro.resultado

    resumo_dados = ResumoDados(
        treino_bruto=validacao_treino.registros,
        teste_bruto=validacao_teste.registros,
        treino_utilizado=int(len(treino)),
        teste_utilizado=int(len(teste)),
        duplicados_treino_removidos=duplicados_treino,
        duplicados_teste_removidos=duplicados_teste,
        validacao_treino=validacao_treino,
        validacao_teste=validacao_teste,
    )
    metricas = _montar_metricas(
        resumo_dados=resumo_dados,
        acuracia_principal=acuracia_principal,
        acuracia_referencia=acuracia_referencia,
        intervalo=intervalo,
        comparacao=comparacao,
        equidade=equidade,
    )

    salvar_metricas_json(metricas, diretorio_resultados / "metricas.json")
    salvar_equidade_csv(equidade, diretorio_resultados / "equidade_por_grupo.csv")
    escrever_interpretacao(
        caminho=diretorio_resultados / "interpretacao.md",
        acuracia_principal=acuracia_principal,
        intervalo=intervalo,
        acuracia_referencia=acuracia_referencia,
        comparacao=comparacao,
        equidade=equidade,
        registros_teste=resumo_dados.teste_utilizado,
        ausentes_permitidos_treino=validacao_treino.ausentes_por_coluna,
        ausentes_permitidos_teste=validacao_teste.ausentes_por_coluna,
        duplicados_treino=duplicados_treino,
        duplicados_teste=duplicados_teste,
    )

    return ResultadoAnalise(
        resumo_dados=resumo_dados,
        modelagem=modelagem,
        acuracia_principal=acuracia_principal,
        acuracia_referencia=acuracia_referencia,
        intervalo_principal=intervalo,
        comparacao=comparacao,
        equidade=equidade,
        metricas=metricas,
    )


# Métricas são serializadas em estrutura explícita para consumo e auditoria.
def _montar_metricas(
    resumo_dados: ResumoDados,
    acuracia_principal: float,
    acuracia_referencia: float,
    intervalo: IntervaloBootstrap,
    comparacao: ComparacaoBootstrap,
    equidade: ResultadoEquidade,
) -> dict[str, Any]:
    """
    Monta estrutura serializável com todas as métricas calculadas.

    Parâmetros:
        resumo_dados: Resumo dos dados brutos e utilizados.
        acuracia_principal: Acurácia pontual da regressão logística.
        acuracia_referencia: Acurácia pontual do modelo de referência.
        intervalo: Intervalo bootstrap da acurácia principal.
        comparacao: Resultado da comparação bootstrap pareada.
        equidade: Resultado de paridade demográfica.

    Retorno:
        Dicionário pronto para serialização em JSON, contendo origem dos dados,
        qualidade, modelagem, estatística e fairness.
    """
    return {
        "fonte": {
            "dataset": "Adult / Census Income",
            "pagina_oficial": URL_DATASET_UCI,
            "doi": DOI_DATASET,
            "licenca": LICENCA_DATASET,
        },
        "registros": {
            "treino_bruto": resumo_dados.treino_bruto,
            "teste_bruto": resumo_dados.teste_bruto,
            "treino_utilizado": resumo_dados.treino_utilizado,
            "teste_utilizado": resumo_dados.teste_utilizado,
            "duplicados_treino_removidos": resumo_dados.duplicados_treino_removidos,
            "duplicados_teste_removidos": resumo_dados.duplicados_teste_removidos,
        },
        "qualidade_dados": {
            "ausentes_treino": resumo_dados.validacao_treino.ausentes_por_coluna,
            "ausentes_teste": resumo_dados.validacao_teste.ausentes_por_coluna,
            "observacao_unicidade": (
                "O dataset nao possui chave natural confiavel; por isso nao foi "
                "criada regra artificial de unicidade. Registros totalmente "
                "duplicados foram contabilizados e removidos antes da modelagem."
            ),
        },
        "modelo_principal": {
            "tipo": "RegressaoLogistica",
            "preprocessamento": (
                "ColumnTransformer com imputacao numerica/categorica, "
                "normalizacao numerica e OneHotEncoder com categorias "
                "desconhecidas ignoradas."
            ),
            "acuracia": acuracia_principal,
            "ic_bootstrap_95": {
                "limite_inferior": intervalo.limite_inferior,
                "limite_superior": intervalo.limite_superior,
                "reamostragens": intervalo.reamostragens,
                "semente": SEMENTE,
            },
        },
        "modelo_referencia": {
            "tipo": "DummyClassifier",
            "estrategia": "most_frequent",
            "acuracia": acuracia_referencia,
        },
        "comparacao_bootstrap_pareada": {
            "diferenca_pontual": comparacao.diferenca_pontual,
            "limite_inferior": comparacao.limite_inferior,
            "limite_superior": comparacao.limite_superior,
            "reamostragens": comparacao.reamostragens,
            "significativa": comparacao.significativa,
            "modelo_principal_superior": comparacao.modelo_principal_superior,
            "usa_p_valor": False,
        },
        "fairness": {
            "metrica": "paridade_demografica",
            "atributo_sensivel": "sexo",
            "limite_operacional": equidade.limite,
            "gap": equidade.gap,
            "status": equidade.status,
            "taxas_por_grupo": [
                {
                    "grupo": taxa.grupo,
                    "n_registros": taxa.n_registros,
                    "predicoes_positivas": taxa.predicoes_positivas,
                    "taxa_positiva": taxa.taxa_positiva,
                }
                for taxa in equidade.taxas_por_grupo
            ],
            "justificativa": (
                "Paridade demografica foi adotada por avaliar diferencas nas "
                "taxas de predicao positiva entre grupos sensiveis. O limite "
                "0,10 e um criterio operacional adotado na analise, nao uma regra "
                "legal ou universal."
            ),
        },
    }
