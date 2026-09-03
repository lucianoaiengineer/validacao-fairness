"""
Arquivo: equidade.py

Calcula paridade demográfica sobre predições do conjunto de teste.

Fluxo:
1. Recebe grupos sensíveis e predições binárias já geradas pelo modelo.
2. Calcula taxa de predições positivas para cada grupo obrigatório.
3. Calcula o gap absoluto entre a maior e a menor taxa positiva.
4. Compara o gap com o limite operacional adotado na análise.
5. Pode lançar erro controlado quando a paridade não é atendida.

Autor: Luciano Magalhães
Setembro 2026
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from validacao_fairness.configuracao import (
    GRUPOS_SEXO,
    LIMITE_PARIDADE_DEMOGRAFICA,
)


@dataclass(frozen=True)
class TaxaGrupo:
    """
    Representa a taxa de predições positivas de um grupo sensível.

    Atributos:
        grupo: Nome do grupo avaliado.
        n_registros: Quantidade de registros do grupo no conjunto de teste.
        predicoes_positivas: Quantidade de predições positivas no grupo.
        taxa_positiva: Proporção de predições positivas do grupo.
    """

    grupo: str
    n_registros: int
    predicoes_positivas: int
    taxa_positiva: float


@dataclass(frozen=True)
class ResultadoEquidade:
    """
    Consolida o resultado da verificação de paridade demográfica.

    Atributos:
        taxas_por_grupo: Taxas positivas calculadas para cada grupo avaliado.
        gap: Diferença absoluta entre a maior e a menor taxa positiva.
        limite: Limite operacional adotado para o gap.
        status: Texto `atendida` ou `não atendida` conforme o limite.
    """

    taxas_por_grupo: tuple[TaxaGrupo, ...]
    gap: float
    limite: float
    status: str


class ViolacaoParidadeDemograficaError(ValueError):
    """
    Indica violação do limite operacional de paridade demográfica.

    Atributos:
        resultado: ResultadoEquidade completo, preservado para relatórios.

    Uso:
        A exceção permite que testes validem a falha enquanto o pipeline
        principal pode capturar o resultado e continuar a geração de artefatos.
    """

    def __init__(self, resultado: ResultadoEquidade) -> None:
        """
        Inicializa a exceção com o resultado de fairness calculado.

        Parâmetros:
            resultado: ResultadoEquidade cujo gap ultrapassou o limite
            operacional.

        Retorno:
            None. O método apenas inicializa a instância de exceção.
        """
        self.resultado = resultado
        super().__init__(
            "Gap de paridade demografica "
            f"{resultado.gap:.4f} acima do limite operacional {resultado.limite:.4f}."
        )


def verificar_paridade_demografica(
    grupos_sensiveis: Sequence[str],
    predicoes: Sequence[int] | npt.NDArray[np.integer],
    limite: float = LIMITE_PARIDADE_DEMOGRAFICA,
    grupos_avaliados: tuple[str, ...] = GRUPOS_SEXO,
    falhar_ao_violar: bool = True,
) -> ResultadoEquidade:
    """
    Verifica paridade demográfica e opcionalmente falha ao violar o limite.

    Parâmetros:
        grupos_sensiveis: Sequência com o grupo sensível de cada registro.
        predicoes: Sequência ou array NumPy com predições binárias 0/1.
        limite: Gap máximo aceito pelo critério operacional adotado na análise.
        grupos_avaliados: Grupos que devem obrigatoriamente ser avaliados.
        falhar_ao_violar: Quando verdadeiro, lança exceção se o gap ultrapassa
        o limite.

    Retorno:
        ResultadoEquidade com taxas positivas, gap, limite e status.

    Exceções:
        ViolacaoParidadeDemograficaError: Lançada quando o gap ultrapassa o
        limite e `falhar_ao_violar` é verdadeiro.
        ValueError: Lançado quando entradas têm tamanho incompatível, estão
        vazias ou não contêm algum grupo obrigatório.
    """
    resultado = calcular_paridade_demografica(
        grupos_sensiveis=grupos_sensiveis,
        predicoes=predicoes,
        limite=limite,
        grupos_avaliados=grupos_avaliados,
    )
    if falhar_ao_violar and resultado.gap > limite:
        raise ViolacaoParidadeDemograficaError(resultado)
    return resultado


def calcular_paridade_demografica(
    grupos_sensiveis: Sequence[str],
    predicoes: Sequence[int] | npt.NDArray[np.integer],
    limite: float = LIMITE_PARIDADE_DEMOGRAFICA,
    grupos_avaliados: tuple[str, ...] = GRUPOS_SEXO,
) -> ResultadoEquidade:
    """
    Calcula taxas positivas por grupo e gap absoluto de paridade.

    Parâmetros:
        grupos_sensiveis: Sequência com o grupo sensível de cada registro.
        predicoes: Sequência ou array NumPy com predições binárias 0/1.
        limite: Gap máximo aceito para classificar o status operacional.
        grupos_avaliados: Grupos obrigatórios incluídos no cálculo.

    Retorno:
        ResultadoEquidade com os detalhes da paridade demográfica.

    Exceções:
        ValueError: Lançado quando o limite é inválido, o vetor de predições
        não é unidimensional/binário, os tamanhos diferem, não há registros ou
        algum grupo obrigatório está ausente.
    """
    if not 0.0 <= limite <= 1.0:
        raise ValueError("O limite de paridade demografica deve estar entre 0 e 1.")

    grupos = list(grupos_sensiveis)
    predicoes_array = np.asarray(predicoes)
    if predicoes_array.ndim != 1:
        raise ValueError("O vetor de predicoes deve ser unidimensional.")
    if not np.issubdtype(predicoes_array.dtype, np.integer):
        raise ValueError("Predicoes devem conter somente valores binarios 0 e 1.")
    if not np.isin(predicoes_array, [0, 1]).all():
        raise ValueError("Predicoes devem conter somente valores binarios 0 e 1.")

    predicoes_binarias = predicoes_array.astype(np.int_)
    if len(grupos) != int(predicoes_binarias.size):
        raise ValueError("Grupos sensiveis e predicoes devem ter o mesmo tamanho.")
    if predicoes_binarias.size == 0:
        raise ValueError("Nao e possivel calcular fairness sem registros.")

    taxas: list[TaxaGrupo] = []
    for grupo in grupos_avaliados:
        # Cada grupo obrigatório precisa aparecer para a comparação fazer sentido.
        mascara = np.asarray([valor == grupo for valor in grupos], dtype=bool)
        n_registros = int(mascara.sum())
        if n_registros == 0:
            raise ValueError(f"Grupo obrigatorio ausente na avaliacao: {grupo}.")
        positivos = int(predicoes_binarias[mascara].sum())
        taxas.append(
            TaxaGrupo(
                grupo=grupo,
                n_registros=n_registros,
                predicoes_positivas=positivos,
                taxa_positiva=positivos / n_registros,
            )
        )

    valores = [taxa.taxa_positiva for taxa in taxas]
    gap = max(valores) - min(valores)
    status = "atendida" if gap <= limite else "não atendida"
    return ResultadoEquidade(
        taxas_por_grupo=tuple(taxas),
        gap=float(gap),
        limite=limite,
        status=status,
    )
