"""
Arquivo: estatistica.py

Implementa métricas e intervalos bootstrap para avaliação do modelo.

Fluxo:
1. Valida vetores pareados de rótulos e predições.
2. Calcula acurácia pontual do modelo principal.
3. Estima intervalo de confiança bootstrap de 95% para a acurácia.
4. Compara modelo principal e referência por bootstrap pareado.
5. Classifica significância pela posição do intervalo em relação a zero.

Autor: Luciano Magalhães
Setembro 2026
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from validacao_fairness.configuracao import (
    NIVEL_CONFIANCA,
    REAMOSTRAGENS_BOOTSTRAP,
    SEMENTE,
)

VetorInteiro = Sequence[int] | npt.NDArray[np.integer[Any]]


@dataclass(frozen=True)
class IntervaloBootstrap:
    """
    Representa intervalo bootstrap de uma métrica pontual.

    Atributos:
        metrica_pontual: Valor da métrica calculado no conjunto observado.
        limite_inferior: Limite inferior do intervalo de confiança.
        limite_superior: Limite superior do intervalo de confiança.
        nivel_confianca: Nível de confiança adotado no cálculo.
        reamostragens: Quantidade de amostras bootstrap utilizadas.
    """

    metrica_pontual: float
    limite_inferior: float
    limite_superior: float
    nivel_confianca: float
    reamostragens: int


@dataclass(frozen=True)
class ComparacaoBootstrap:
    """
    Representa comparação bootstrap pareada entre dois modelos.

    Atributos:
        diferenca_pontual: Diferença entre acurácia principal e referência.
        limite_inferior: Limite inferior do intervalo da diferença.
        limite_superior: Limite superior do intervalo da diferença.
        nivel_confianca: Nível de confiança adotado no cálculo.
        reamostragens: Quantidade de amostras bootstrap utilizadas.
        significativa: Indica se o intervalo não contém zero.
        modelo_principal_superior: Indica se o intervalo ficou acima de zero.
    """

    diferenca_pontual: float
    limite_inferior: float
    limite_superior: float
    nivel_confianca: float
    reamostragens: int
    significativa: bool
    modelo_principal_superior: bool


def calcular_acuracia(
    y_verdadeiro: VetorInteiro,
    y_predito: VetorInteiro,
) -> float:
    """
    Calcula a acurácia entre rótulos reais e preditos.

    Parâmetros:
        y_verdadeiro: Vetor unidimensional com rótulos reais.
        y_predito: Vetor unidimensional com predições do modelo.

    Retorno:
        Proporção de acertos no intervalo fechado de 0 a 1.

    Exceções:
        ValueError: Lançado quando os vetores são vazios, têm dimensões
        incompatíveis ou tamanhos diferentes.
    """
    verdadeiro, predito = _validar_vetores_pareados(y_verdadeiro, y_predito)
    return float(np.mean(verdadeiro == predito))


# Bootstrap estima incerteza amostral da métrica sem supor normalidade.
def intervalo_confianca_bootstrap_acuracia(
    y_verdadeiro: VetorInteiro,
    y_predito: VetorInteiro,
    reamostragens: int = REAMOSTRAGENS_BOOTSTRAP,
    nivel_confianca: float = NIVEL_CONFIANCA,
    semente: int = SEMENTE,
) -> IntervaloBootstrap:
    """
    Calcula intervalo de confiança bootstrap para a acurácia.

    Parâmetros:
        y_verdadeiro: Vetor unidimensional com rótulos reais.
        y_predito: Vetor unidimensional com predições do modelo principal.
        reamostragens: Quantidade de reamostragens bootstrap com reposição.
        nivel_confianca: Nível de confiança usado para definir os quantis.
        semente: Semente determinística usada pelo gerador NumPy.

    Retorno:
        IntervaloBootstrap com métrica pontual, limites, nível de confiança e
        quantidade de reamostragens.

    Exceções:
        ValueError: Lançado quando os vetores de entrada não são válidos.
    """
    verdadeiro, predito = _validar_vetores_pareados(y_verdadeiro, y_predito)
    rng = np.random.default_rng(semente)
    metricas = np.empty(reamostragens, dtype=float)
    tamanho = verdadeiro.size

    # Cada reamostragem usa índices com reposição do mesmo conjunto de teste.
    for indice in range(reamostragens):
        amostra = rng.integers(0, tamanho, size=tamanho)
        metricas[indice] = float(np.mean(verdadeiro[amostra] == predito[amostra]))

    alfa = (1.0 - nivel_confianca) / 2.0
    quantis = np.quantile(metricas, [alfa, 1.0 - alfa])
    return IntervaloBootstrap(
        metrica_pontual=calcular_acuracia(verdadeiro, predito),
        limite_inferior=float(quantis[0]),
        limite_superior=float(quantis[1]),
        nivel_confianca=nivel_confianca,
        reamostragens=reamostragens,
    )


# Comparação pareada reamostra os mesmos índices para ambos os modelos.
def comparar_acuracias_bootstrap_pareado(
    y_verdadeiro: VetorInteiro,
    predicoes_principal: VetorInteiro,
    predicoes_referencia: VetorInteiro,
    reamostragens: int = REAMOSTRAGENS_BOOTSTRAP,
    nivel_confianca: float = NIVEL_CONFIANCA,
    semente: int = SEMENTE,
) -> ComparacaoBootstrap:
    """
    Compara acurácias por bootstrap pareado no mesmo conjunto de teste.

    Parâmetros:
        y_verdadeiro: Vetor unidimensional com rótulos reais.
        predicoes_principal: Predições do modelo principal.
        predicoes_referencia: Predições do modelo de referência.
        reamostragens: Quantidade de reamostragens bootstrap com reposição.
        nivel_confianca: Nível de confiança usado para definir os quantis.
        semente: Semente determinística usada pelo gerador NumPy.

    Retorno:
        ComparacaoBootstrap com diferença pontual, intervalo da diferença e
        indicadores de significância estatística.

    Exceções:
        ValueError: Lançado quando os vetores têm tamanho incompatível ou
        quando os rótulos reais pareados não são idênticos.
    """
    verdadeiro, principal = _validar_vetores_pareados(
        y_verdadeiro,
        predicoes_principal,
    )
    verdadeiro_referencia, referencia = _validar_vetores_pareados(
        y_verdadeiro,
        predicoes_referencia,
    )
    if not np.array_equal(verdadeiro, verdadeiro_referencia):
        raise ValueError("Os vetores reais usados na comparacao devem ser identicos.")

    rng = np.random.default_rng(semente)
    diferencas = np.empty(reamostragens, dtype=float)
    tamanho = verdadeiro.size

    # A diferença por amostra evita independência falsa entre modelos.
    for indice in range(reamostragens):
        amostra = rng.integers(0, tamanho, size=tamanho)
        acuracia_principal = float(np.mean(verdadeiro[amostra] == principal[amostra]))
        acuracia_referencia = float(np.mean(verdadeiro[amostra] == referencia[amostra]))
        diferencas[indice] = acuracia_principal - acuracia_referencia

    alfa = (1.0 - nivel_confianca) / 2.0
    quantis = np.quantile(diferencas, [alfa, 1.0 - alfa])
    limite_inferior = float(quantis[0])
    limite_superior = float(quantis[1])
    significativa = not (limite_inferior <= 0.0 <= limite_superior)

    return ComparacaoBootstrap(
        diferenca_pontual=calcular_acuracia(verdadeiro, principal)
        - calcular_acuracia(verdadeiro, referencia),
        limite_inferior=limite_inferior,
        limite_superior=limite_superior,
        nivel_confianca=nivel_confianca,
        reamostragens=reamostragens,
        significativa=significativa,
        modelo_principal_superior=limite_inferior > 0.0,
    )


def _validar_vetores_pareados(
    y_verdadeiro: VetorInteiro,
    y_predito: VetorInteiro,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Normaliza vetores pareados e garante formato compatível.

    Parâmetros:
        y_verdadeiro: Vetor de rótulos reais a ser convertido para NumPy.
        y_predito: Vetor de predições a ser convertido para NumPy.

    Retorno:
        Tupla com dois arrays NumPy inteiros, unidimensionais e de mesmo
        tamanho.

    Exceções:
        ValueError: Lançado quando qualquer vetor é multidimensional, vazio ou
        possui tamanho diferente do outro.
    """
    verdadeiro = np.asarray(y_verdadeiro, dtype=np.int_)
    predito = np.asarray(y_predito, dtype=np.int_)

    if verdadeiro.ndim != 1 or predito.ndim != 1:
        raise ValueError("Os vetores devem ser unidimensionais.")
    if verdadeiro.size == 0:
        raise ValueError("Os vetores nao podem ser vazios.")
    if verdadeiro.size != predito.size:
        raise ValueError("Os vetores devem possuir o mesmo tamanho.")

    return verdadeiro, predito
