"""
Arquivo: test_validacao.py

Testa regras de schema, completude, faixas e duplicados.

Fluxo:
1. Usa fixture sintética independente de rede.
2. Confirma aceite de ausentes em colunas não críticas.
3. Confirma reprovação de ausentes críticos e faixas inválidas.
4. Valida contagem e remoção de registros totalmente duplicados.

Autor: Luciano Magalhães
Setembro 2026
"""

import pandas as pd
import pytest

from validacao_fairness.validacao import (
    ValidacaoDadosError,
    remover_duplicados_totais,
    validar_dataset,
)


def test_validar_dataset_aceita_amostra_com_ausentes_permitidos(
    dados_adult_validos: pd.DataFrame,
) -> None:
    """
    Valida uma amostra correta com ausentes em colunas não críticas.

    Parâmetros:
        dados_adult_validos: Fixture com DataFrame sintético no schema Adult.

    Retorno:
        None. Asserções verificam que a validação aceita o dataset e registra
        ausências permitidas.
    """
    resultado = validar_dataset(dados_adult_validos, "amostra")

    assert resultado.registros == 8
    assert resultado.ausentes_por_coluna == {
        "classe_trabalho": 1,
        "ocupacao": 1,
        "pais_origem": 1,
    }


def test_validar_dataset_reprova_ausente_critico(
    dados_adult_validos: pd.DataFrame,
) -> None:
    """
    Confirma reprovação quando `sexo` possui valor ausente.

    Parâmetros:
        dados_adult_validos: Fixture usada como base antes da alteração
        controlada no atributo sensível.

    Retorno:
        None. O teste passa quando `ValidacaoDadosError` é lançado com
        mensagem de coluna crítica.
    """
    dados = dados_adult_validos.copy()
    dados.loc[0, "sexo"] = None

    with pytest.raises(ValidacaoDadosError, match="colunas criticas"):
        validar_dataset(dados, "amostra")


def test_validar_dataset_reprova_ausente_inesperado_numerico(
    dados_adult_validos: pd.DataFrame,
) -> None:
    """
    Confirma reprovação de ausência inesperada em coluna numérica.

    Parâmetros:
        dados_adult_validos: Fixture usada como base antes da inserção de
        ausência em `peso_amostral`.

    Retorno:
        None. O teste passa quando a validação reprova a ausência inesperada.
    """
    dados = dados_adult_validos.copy()
    dados.loc[0, "peso_amostral"] = None

    with pytest.raises(ValidacaoDadosError, match="ausentes inesperados"):
        validar_dataset(dados, "amostra")


def test_validar_dataset_reprova_ausente_inesperado_categorico(
    dados_adult_validos: pd.DataFrame,
) -> None:
    """
    Confirma reprovação de ausência inesperada em coluna categórica.

    Parâmetros:
        dados_adult_validos: Fixture usada como base antes da inserção de
        ausência em `escolaridade`.

    Retorno:
        None. O teste passa quando a validação reprova a ausência inesperada.
    """
    dados = dados_adult_validos.copy()
    dados.loc[0, "escolaridade"] = None

    with pytest.raises(ValidacaoDadosError, match="ausentes inesperados"):
        validar_dataset(dados, "amostra")


@pytest.mark.parametrize("rotulo_invalido", [0.5, 2])
def test_validar_dataset_reprova_rotulo_fora_do_dominio_exato(
    dados_adult_validos: pd.DataFrame,
    rotulo_invalido: object,
) -> None:
    """
    Confirma reprovação de rótulos diferentes dos inteiros exatos 0 e 1.

    Parâmetros:
        dados_adult_validos: Fixture usada como base antes da alteração do
        rótulo.
        rotulo_invalido: Valor inválido injetado no rótulo binário.

    Retorno:
        None. O teste passa quando a validação reprova 0.5 e 2.
    """
    dados = dados_adult_validos.copy()
    if isinstance(rotulo_invalido, float):
        dados["renda_acima_50k"] = dados["renda_acima_50k"].astype(float)
    dados.loc[0, "renda_acima_50k"] = rotulo_invalido

    with pytest.raises(ValidacaoDadosError, match="dominio invalido"):
        validar_dataset(dados, "amostra")


# Faixas inválidas precisam ser detectadas antes de modelagem.
def test_validar_dataset_reprova_faixa_invalida(
    dados_adult_validos: pd.DataFrame,
) -> None:
    """
    Confirma reprovação de horas semanais fora do domínio esperado.

    Parâmetros:
        dados_adult_validos: Fixture usada como base antes da alteração
        controlada em `horas_semanais`.

    Retorno:
        None. O teste passa quando a validação reprova a faixa inválida.
    """
    dados = dados_adult_validos.copy()
    dados.loc[0, "horas_semanais"] = 0

    with pytest.raises(ValidacaoDadosError, match="horas_semanais"):
        validar_dataset(dados, "amostra")


# Duplicados totais são tratados sem criar falsa chave de unicidade.
def test_remover_duplicados_totais_documenta_tratamento(
    dados_adult_validos: pd.DataFrame,
) -> None:
    """
    Verifica contagem e remoção de registros totalmente duplicados.

    Parâmetros:
        dados_adult_validos: Fixture usada para criar duplicação total
        controlada.

    Retorno:
        None. Asserções verificam quantidade removida e tamanho final.
    """
    dados = pd.concat(
        [dados_adult_validos, dados_adult_validos.iloc[[0]]],
        ignore_index=True,
    )

    dados_sem_duplicados, quantidade = remover_duplicados_totais(dados)

    assert quantidade == 1
    assert len(dados_sem_duplicados) == len(dados_adult_validos)
