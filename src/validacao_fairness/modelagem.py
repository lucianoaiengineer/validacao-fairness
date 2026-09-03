"""
Arquivo: modelagem.py

Cria pipelines de pré-processamento, modelo principal e modelo de referência.

Fluxo:
1. Define colunas preditoras excluindo `sexo`, `raca` e o rótulo.
2. Cria `ColumnTransformer` com imputação, normalização e codificação.
3. Treina regressão logística como modelo principal.
4. Treina `DummyClassifier` como referência estatística documentada.
5. Gera predições para o conjunto de teste oficial.

Autor: Luciano Magalhães
Setembro 2026
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validacao_fairness.configuracao import (
    COLUNAS_CATEGORICAS_MODELO,
    COLUNAS_NUMERICAS_MODELO,
    ROTULO,
    SEMENTE,
)


@dataclass(frozen=True)
class ResultadoModelagem:
    """
    Armazena modelos treinados, rótulos reais e predições.

    Atributos:
        modelo_principal: Pipeline treinado da regressão logística.
        modelo_referencia: Pipeline treinado do `DummyClassifier`.
        y_teste: Rótulos reais do conjunto de teste.
        predicoes_principal: Predições binárias do modelo principal.
        predicoes_referencia: Predições binárias do modelo de referência.
    """

    modelo_principal: Pipeline
    modelo_referencia: Pipeline
    y_teste: npt.NDArray[np.int_]
    predicoes_principal: npt.NDArray[np.int_]
    predicoes_referencia: npt.NDArray[np.int_]


# Expor as colunas facilita testar que sexo e raça não entram no treinamento.
def colunas_preditoras_modelo() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Retorna colunas numéricas e categóricas usadas como preditoras.

    Parâmetros:
        Não recebe parâmetros. As colunas são definidas centralmente em
        `validacao_fairness.configuracao`.

    Retorno:
        Tupla com duas tuplas internas: colunas numéricas do modelo e colunas
        categóricas do modelo.
    """
    return COLUNAS_NUMERICAS_MODELO, COLUNAS_CATEGORICAS_MODELO


def criar_preprocessador() -> ColumnTransformer:
    """
    Cria pré-processamento com imputação, escala e one-hot encoding.

    Parâmetros:
        Não recebe parâmetros. A seleção de colunas vem das constantes do
        projeto.

    Retorno:
        ColumnTransformer preparado para imputar dados numéricos e categóricos,
        normalizar variáveis numéricas e codificar categorias desconhecidas sem
        erro no conjunto de teste.
    """
    # Numéricas recebem mediana e escala, adequadas para regressão logística.
    transformador_numerico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("normalizador", StandardScaler()),
        ],
    )
    # Categóricas recebem moda e one-hot com tolerância a categorias novas.
    transformador_categorico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore")),
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", transformador_numerico, list(COLUNAS_NUMERICAS_MODELO)),
            ("categoricas", transformador_categorico, list(COLUNAS_CATEGORICAS_MODELO)),
        ],
        remainder="drop",
    )


def criar_modelo_principal(semente: int = SEMENTE) -> Pipeline:
    """
    Cria pipeline da regressão logística principal.

    Parâmetros:
        semente: Valor usado pelo classificador para reprodutibilidade quando
        aplicável.

    Retorno:
        Pipeline do scikit-learn com pré-processamento e regressão logística.
    """
    return Pipeline(
        steps=[
            ("preprocessamento", criar_preprocessador()),
            (
                "classificador",
                LogisticRegression(max_iter=1_000, random_state=semente),
            ),
        ],
    )


# Baseline simples e documentado para comparação estatística.
def criar_modelo_referencia(semente: int = SEMENTE) -> Pipeline:
    """
    Cria baseline `DummyClassifier` com estratégia `most_frequent`.

    Parâmetros:
        semente: Valor usado pelo classificador de referência para manter
        reprodutibilidade quando aplicável.

    Retorno:
        Pipeline do scikit-learn com o mesmo pré-processamento e baseline
        supervisionado simples.
    """
    return Pipeline(
        steps=[
            ("preprocessamento", criar_preprocessador()),
            (
                "classificador",
                DummyClassifier(strategy="most_frequent", random_state=semente),
            ),
        ],
    )


# Treinamento respeita a separação oficial: fit em treino, predict em teste.
def treinar_e_predizer(
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    semente: int = SEMENTE,
) -> ResultadoModelagem:
    """
    Treina modelo principal e referência, retornando predições de teste.

    Parâmetros:
        treino: DataFrame oficial de treino já validado e deduplicado.
        teste: DataFrame oficial de teste já validado e deduplicado.
        semente: Semente determinística usada na criação dos modelos.

    Retorno:
        ResultadoModelagem com pipelines treinados, rótulos reais e predições
        de ambos os modelos.

    Exceções:
        Propaga erros do scikit-learn quando os dados não são compatíveis com
        ajuste, transformação ou predição.
    """
    x_treino = treino.drop(columns=[ROTULO])
    y_treino = np.asarray(treino[ROTULO].to_numpy(), dtype=np.int_)
    x_teste = teste.drop(columns=[ROTULO])
    y_teste = np.asarray(teste[ROTULO].to_numpy(), dtype=np.int_)

    modelo_principal = criar_modelo_principal(semente)
    modelo_referencia = criar_modelo_referencia(semente)

    modelo_principal.fit(x_treino, y_treino)
    modelo_referencia.fit(x_treino, y_treino)

    predicoes_principal = np.asarray(modelo_principal.predict(x_teste), dtype=np.int_)
    predicoes_referencia = np.asarray(modelo_referencia.predict(x_teste), dtype=np.int_)

    return ResultadoModelagem(
        modelo_principal=modelo_principal,
        modelo_referencia=modelo_referencia,
        y_teste=y_teste,
        predicoes_principal=predicoes_principal,
        predicoes_referencia=predicoes_referencia,
    )
