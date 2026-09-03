"""
Arquivo: test_estatistica.py

Testa acurácia, intervalo bootstrap e comparação bootstrap pareada.

Fluxo:
1. Valida cálculo pontual de acurácia.
2. Confirma determinismo do bootstrap com semente fixa.
3. Testa cenário com superioridade estatística evidente.
4. Testa cenário em que a diferença entre modelos contém zero.

Autor: Luciano Magalhães
Setembro 2026
"""

import numpy as np

from validacao_fairness.estatistica import (
    calcular_acuracia,
    comparar_acuracias_bootstrap_pareado,
    intervalo_confianca_bootstrap_acuracia,
)


# Acurácia simples serve como base para os testes bootstrap.
def test_calcular_acuracia() -> None:
    """
    Calcula corretamente a proporção de acertos.

    Parâmetros:
        Não recebe parâmetros. Os vetores são definidos dentro do próprio
        teste para manter o caso mínimo e auditável.

    Retorno:
        None. A asserção valida a acurácia esperada de 0,75.
    """
    assert calcular_acuracia([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75


def test_intervalo_bootstrap_e_deterministico() -> None:
    """
    Garante determinismo do intervalo bootstrap com semente 42.

    Parâmetros:
        Não recebe parâmetros. O teste usa vetores sintéticos pequenos e
        reamostragens reduzidas para execução rápida.

    Retorno:
        None. Asserções confirmam igualdade entre execuções e coerência dos
        limites em relação à métrica pontual.
    """
    y = np.array([1, 0, 1, 0, 1, 0])
    pred = np.array([1, 0, 1, 0, 0, 0])

    intervalo_a = intervalo_confianca_bootstrap_acuracia(
        y,
        pred,
        reamostragens=200,
        semente=42,
    )
    intervalo_b = intervalo_confianca_bootstrap_acuracia(
        y,
        pred,
        reamostragens=200,
        semente=42,
    )

    assert intervalo_a == intervalo_b
    assert intervalo_a.limite_inferior <= intervalo_a.metrica_pontual
    assert intervalo_a.limite_superior >= intervalo_a.metrica_pontual


def test_comparacao_bootstrap_detecta_superioridade() -> None:
    """
    Detecta modelo principal estatisticamente superior ao baseline.

    Parâmetros:
        Não recebe parâmetros. O cenário extremo é criado localmente para
        produzir intervalo totalmente acima de zero.

    Retorno:
        None. Asserções confirmam significância e superioridade do modelo
        principal.
    """
    y = np.array([1, 0] * 40)
    principal = y.copy()
    referencia = 1 - y

    comparacao = comparar_acuracias_bootstrap_pareado(
        y,
        principal,
        referencia,
        reamostragens=300,
        semente=42,
    )

    assert comparacao.significativa is True
    assert comparacao.modelo_principal_superior is True
    assert comparacao.limite_inferior > 0


def test_comparacao_bootstrap_intervalo_contem_zero() -> None:
    """
    Confirma que empate perfeito não gera diferença significativa.

    Parâmetros:
        Não recebe parâmetros. O mesmo vetor é usado para ambos os modelos.

    Retorno:
        None. Asserções confirmam que o intervalo da diferença é exatamente
        zero e não significativo.
    """
    y = np.array([1, 0, 1, 0, 1, 0])

    comparacao = comparar_acuracias_bootstrap_pareado(
        y,
        y,
        y,
        reamostragens=100,
        semente=42,
    )

    assert comparacao.significativa is False
    assert comparacao.limite_inferior == 0
    assert comparacao.limite_superior == 0
