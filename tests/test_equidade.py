"""
Arquivo: test_equidade.py

Testa a verificação de paridade demográfica em cenários controlados.

Fluxo:
1. Confirma status atendido quando taxas positivas são iguais.
2. Confirma erro controlado quando o gap ultrapassa o limite.
3. Confirma retorno sem interrupção quando a falha é desativada.

Autor: Luciano Magalhães
Setembro 2026
"""

import numpy as np
import pytest

from validacao_fairness.equidade import (
    ViolacaoParidadeDemograficaError,
    verificar_paridade_demografica,
)


def test_paridade_demografica_atendida() -> None:
    """
    Calcula paridade quando os dois grupos têm a mesma taxa positiva.

    Parâmetros:
        Não recebe parâmetros. Grupos e predições são definidos localmente para
        criar taxa positiva idêntica.

    Retorno:
        None. Asserções confirmam status atendido, gap zero e taxas esperadas.
    """
    resultado = verificar_paridade_demografica(
        grupos_sensiveis=["Feminino", "Feminino", "Masculino", "Masculino"],
        predicoes=[1, 0, 1, 0],
        limite=0.10,
    )

    assert resultado.status == "atendida"
    assert resultado.gap == 0
    assert [taxa.taxa_positiva for taxa in resultado.taxas_por_grupo] == [0.5, 0.5]


def test_paridade_demografica_viola_limite_com_erro_controlado() -> None:
    """
    Gera erro controlado quando o gap excede o limite operacional.

    Parâmetros:
        Não recebe parâmetros. O cenário sintético força taxa 0 para um grupo
        e taxa 1 para o outro.

    Retorno:
        None. O teste passa quando a exceção preserva o resultado calculado.
    """
    with pytest.raises(ViolacaoParidadeDemograficaError) as erro:
        verificar_paridade_demografica(
            grupos_sensiveis=["Feminino", "Feminino", "Masculino", "Masculino"],
            predicoes=[0, 0, 1, 1],
            limite=0.10,
        )

    resultado = erro.value.resultado
    assert resultado.status == "não atendida"
    assert resultado.gap == 1.0


# Pipeline usa este modo para registrar violação sem interromper relatório.
def test_paridade_demografica_pode_retornar_violacao_sem_interromper() -> None:
    """
    Retorna status não atendida quando a falha é desativada.

    Parâmetros:
        Não recebe parâmetros. O cenário é igual ao de violação, mas com
        `falhar_ao_violar=False`.

    Retorno:
        None. Asserções confirmam que a função retorna o status sem lançar
        exceção.
    """
    resultado = verificar_paridade_demografica(
        grupos_sensiveis=["Feminino", "Feminino", "Masculino", "Masculino"],
        predicoes=[0, 0, 1, 1],
        limite=0.10,
        falhar_ao_violar=False,
    )

    assert resultado.status == "não atendida"
    assert resultado.gap == 1.0


# Predições devem ser estritamente binárias para a métrica fazer sentido.
def test_paridade_demografica_reprova_predicao_nao_binaria() -> None:
    """
    Reprova predição fora do domínio binário 0/1.

    Parâmetros:
        Não recebe parâmetros. O cenário injeta valor 2 nas predições.

    Retorno:
        None. O teste passa quando a função lança ValueError.
    """
    with pytest.raises(ValueError, match="binarios 0 e 1"):
        verificar_paridade_demografica(
            grupos_sensiveis=["Feminino", "Feminino", "Masculino", "Masculino"],
            predicoes=[0, 2, 1, 1],
        )


# Vetores multidimensionais não são aceitos porque quebram o pareamento linha a linha.
def test_paridade_demografica_reprova_vetor_multidimensional() -> None:
    """
    Reprova vetor de predições multidimensional.

    Parâmetros:
        Não recebe parâmetros. O teste usa array NumPy bidimensional.

    Retorno:
        None. O teste passa quando a função lança ValueError.
    """
    with pytest.raises(ValueError, match="unidimensional"):
        verificar_paridade_demografica(
            grupos_sensiveis=["Feminino", "Feminino", "Masculino", "Masculino"],
            predicoes=np.array([[0, 1], [1, 0]]),
        )


# O limite operacional precisa estar em escala de proporção.
def test_paridade_demografica_reprova_limite_invalido() -> None:
    """
    Reprova limite fora do intervalo fechado entre 0 e 1.

    Parâmetros:
        Não recebe parâmetros. O teste usa limite maior que 1.

    Retorno:
        None. O teste passa quando a função lança ValueError.
    """
    with pytest.raises(ValueError, match="entre 0 e 1"):
        verificar_paridade_demografica(
            grupos_sensiveis=["Feminino", "Feminino", "Masculino", "Masculino"],
            predicoes=[1, 0, 1, 0],
            limite=1.1,
        )
