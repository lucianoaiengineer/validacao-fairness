"""
Arquivo: validacao.py

Valida schema, completude, domínios e faixas dos dados Adult normalizados.

Fluxo:
1. Confirma que o DataFrame não está vazio.
2. Verifica presença das colunas obrigatórias.
3. Valida tipos numéricos, categóricos e binários.
4. Reprova ausências fora das colunas explicitamente permitidas.
5. Confere domínios esperados de rótulo e atributo sensível.
6. Aplica regras de faixa e contabiliza duplicados totais.

Autor: Luciano Magalhães
Setembro 2026
"""

from dataclasses import dataclass
from numbers import Integral

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

from validacao_fairness.configuracao import (
    COLUNAS_AUSENTES_PERMITIDOS,
    COLUNAS_CATEGORICAS,
    COLUNAS_CRITICAS_SEM_AUSENTES,
    COLUNAS_NUMERICAS,
    COLUNAS_OBRIGATORIAS,
    DOMINIO_ROTULO,
    DOMINIO_SEXO,
    ROTULO,
)


# Erro de domínio do projeto, usado para mensagens controladas nos testes.
class ValidacaoDadosError(ValueError):
    """
    Representa uma falha controlada de qualidade de dados.

    Uso:
        A exceção é levantada quando o dataset viola uma regra explícita de
        schema, domínio, completude ou faixa.

    Parâmetros:
        Herda os parâmetros de `ValueError`, normalmente uma mensagem objetiva
        com o nome do dataset e a regra violada.
    """


@dataclass(frozen=True)
class ResultadoValidacao:
    """
    Resume evidências de qualidade coletadas na validação.

    Atributos:
        nome_dataset: Identificador lógico do conjunto validado.
        registros: Quantidade de linhas presentes no DataFrame validado.
        colunas: Quantidade de colunas presentes no DataFrame validado.
        ausentes_por_coluna: Contagem de valores ausentes por coluna, apenas
            para colunas com pelo menos um valor ausente.
        duplicados_totais: Quantidade de linhas totalmente duplicadas.
    """

    nome_dataset: str
    registros: int
    colunas: int
    ausentes_por_coluna: dict[str, int]
    duplicados_totais: int


def validar_dataset(dados: pd.DataFrame, nome_dataset: str) -> ResultadoValidacao:
    """
    Executa todas as validações obrigatórias sobre um dataset Adult.

    Parâmetros:
        dados: DataFrame já carregado e normalizado com nomes em português.
        nome_dataset: Nome usado em mensagens de erro e no resumo final.

    Retorno:
        ResultadoValidacao com dimensões, ausências contabilizadas e número de
        duplicados totais.

    Exceções:
        ValidacaoDadosError: Lançada quando qualquer regra obrigatória é
        violada.
    """
    validar_nao_vazio(dados, nome_dataset)
    validar_colunas_obrigatorias(dados, nome_dataset)
    validar_tipos(dados, nome_dataset)
    validar_ausentes_criticos(dados, nome_dataset)
    validar_dominio_rotulo(dados, nome_dataset)
    validar_dominio_sexo(dados, nome_dataset)
    validar_faixas(dados, nome_dataset)

    # Pandas tipa nomes de colunas como Hashable; o schema do projeto usa str.
    ausentes = {
        str(coluna): int(quantidade)
        for coluna, quantidade in dados.isna().sum().items()
        if int(quantidade) > 0
    }
    return ResultadoValidacao(
        nome_dataset=nome_dataset,
        registros=int(len(dados)),
        colunas=int(dados.shape[1]),
        ausentes_por_coluna=ausentes,
        duplicados_totais=contar_duplicados_totais(dados),
    )


# Dataset vazio invalida qualquer conclusão estatística ou de fairness.
def validar_nao_vazio(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Garante que o dataset possui pelo menos um registro.

    Parâmetros:
        dados: DataFrame a ser validado.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica que a regra foi atendida.

    Exceções:
        ValidacaoDadosError: Lançada quando o DataFrame está vazio.
    """
    if dados.empty:
        raise ValidacaoDadosError(f"{nome_dataset}: dataset vazio.")


# A presença de colunas é validada antes dos demais acessos ao DataFrame.
def validar_colunas_obrigatorias(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Garante presença de todas as colunas obrigatórias do schema.

    Parâmetros:
        dados: DataFrame a ser inspecionado.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica que todas as colunas existem.

    Exceções:
        ValidacaoDadosError: Lançada quando uma ou mais colunas esperadas estão
        ausentes.
    """
    ausentes = [
        coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in dados.columns
    ]
    if ausentes:
        raise ValidacaoDadosError(
            f"{nome_dataset}: colunas obrigatorias ausentes: {', '.join(ausentes)}."
        )


# Tipos são verificados depois da leitura normalizada dos arquivos oficiais.
def validar_tipos(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Valida tipos numéricos, categóricos e binário do rótulo.

    Parâmetros:
        dados: DataFrame com colunas obrigatórias já presentes.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica compatibilidade dos tipos.

    Exceções:
        ValidacaoDadosError: Lançada quando uma coluna possui tipo incompatível
        com seu papel no dataset Adult.
    """
    for coluna in COLUNAS_NUMERICAS:
        if not is_numeric_dtype(dados[coluna]):
            raise ValidacaoDadosError(
                f"{nome_dataset}: coluna {coluna} deveria ser numerica."
            )

    for coluna in COLUNAS_CATEGORICAS:
        if not (is_string_dtype(dados[coluna]) or is_object_dtype(dados[coluna])):
            raise ValidacaoDadosError(
                f"{nome_dataset}: coluna {coluna} deveria ser categorica."
            )

    if not is_numeric_dtype(dados[ROTULO]):
        raise ValidacaoDadosError(
            f"{nome_dataset}: rotulo {ROTULO} deveria ser binario."
        )


# Ausências críticas são reprovação porque afetam rótulo e análise por grupo.
def validar_ausentes_criticos(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Reprova valores ausentes em colunas críticas para a análise.

    Parâmetros:
        dados: DataFrame com colunas críticas disponíveis.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica que idade, sexo e rótulo estão
        completos.

    Exceções:
        ValidacaoDadosError: Lançada quando qualquer coluna crítica possui
        valores ausentes.
    """
    ausentes_criticos = [
        coluna
        for coluna in COLUNAS_CRITICAS_SEM_AUSENTES
        if dados[coluna].isna().any()
    ]
    if ausentes_criticos:
        raise ValidacaoDadosError(
            f"{nome_dataset}: valores ausentes em colunas criticas: "
            f"{', '.join(ausentes_criticos)}."
        )

    # A regra de completude permite ausências somente nas colunas imputáveis.
    ausentes_inesperados = [
        coluna
        for coluna in COLUNAS_OBRIGATORIAS
        if coluna not in COLUNAS_AUSENTES_PERMITIDOS and dados[coluna].isna().any()
    ]
    if ausentes_inesperados:
        raise ValidacaoDadosError(
            f"{nome_dataset}: valores ausentes inesperados em: "
            f"{', '.join(ausentes_inesperados)}."
        )


# O rótulo precisa estar em 0/1 para métricas estatísticas e modelos.
def validar_dominio_rotulo(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Garante que o rótulo possui domínio binário 0/1.

    Parâmetros:
        dados: DataFrame com a coluna `renda_acima_50k`.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica que o domínio do rótulo é válido.

    Exceções:
        ValidacaoDadosError: Lançada quando valores diferentes de 0 e 1 são
        encontrados no rótulo.
    """
    valores = set(dados[ROTULO].dropna().unique())
    invalidos = [valor for valor in valores if not _valor_rotulo_valido(valor)]
    if invalidos:
        raise ValidacaoDadosError(
            f"{nome_dataset}: dominio invalido para {ROTULO}: "
            f"{sorted(repr(valor) for valor in invalidos)}."
        )


def validar_dominio_sexo(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Garante que o atributo sensível possui grupos esperados.

    Parâmetros:
        dados: DataFrame com a coluna `sexo` já traduzida para português.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica que os grupos pertencem ao domínio
        esperado.

    Exceções:
        ValidacaoDadosError: Lançada quando `sexo` contém grupo fora do domínio
        Feminino/Masculino.
    """
    valores = set(str(valor) for valor in dados["sexo"].dropna().unique())
    if not valores.issubset(DOMINIO_SEXO):
        raise ValidacaoDadosError(
            f"{nome_dataset}: dominio invalido para sexo: {sorted(valores)}."
        )


def validar_faixas(dados: pd.DataFrame, nome_dataset: str) -> None:
    """
    Valida faixas e restrições numéricas compatíveis com o Adult.

    Parâmetros:
        dados: DataFrame com colunas numéricas convertidas.
        nome_dataset: Nome contextual usado na mensagem de falha.

    Retorno:
        None. A ausência de exceção indica que as faixas obrigatórias foram
        respeitadas.

    Exceções:
        ValidacaoDadosError: Lançada quando idade, escolaridade, horas semanais
        ou valores de capital violam as faixas esperadas.
    """
    regras: tuple[tuple[str, float, float], ...] = (
        ("idade", 17, 90),
        ("nivel_escolaridade", 1, 16),
        ("horas_semanais", 1, 99),
        ("ganho_capital", 0, float("inf")),
        ("perda_capital", 0, float("inf")),
    )
    for coluna, minimo, maximo in regras:
        invalidos = dados[coluna].dropna()
        invalidos = invalidos[(invalidos < minimo) | (invalidos > maximo)]
        if not invalidos.empty:
            raise ValidacaoDadosError(
                f"{nome_dataset}: coluna {coluna} fora da faixa esperada "
                f"[{minimo}, {maximo}]."
            )


# Duplicados totais são contabilizados, não tratados como violação de chave.
def contar_duplicados_totais(dados: pd.DataFrame) -> int:
    """
    Conta registros totalmente duplicados sem impor chave natural.

    Parâmetros:
        dados: DataFrame no qual duplicados totais serão identificados.

    Retorno:
        Número de registros repetidos considerando todas as colunas.
    """
    return int(dados.duplicated(keep="first").sum())


# Remoção é feita só após validação para documentar a diferença bruto/utilizado.
def remover_duplicados_totais(dados: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove registros totalmente duplicados e informa a quantidade removida.

    Parâmetros:
        dados: DataFrame que pode conter registros repetidos em todas as
        colunas.

    Retorno:
        Tupla composta pelo DataFrame sem duplicados totais e pela quantidade
        de registros removidos.

    Observação:
        A remoção não cria uma regra artificial de unicidade, pois o Adult não
        possui chave natural confiável.
    """
    quantidade = contar_duplicados_totais(dados)
    if quantidade == 0:
        return dados.reset_index(drop=True), quantidade
    return dados.drop_duplicates(keep="first").reset_index(drop=True), quantidade


def _valor_rotulo_valido(valor: object) -> bool:
    """
    Verifica se um valor do rótulo é exatamente inteiro 0 ou 1.

    Parâmetros:
        valor: Valor único encontrado na coluna `renda_acima_50k`.

    Retorno:
        True quando o valor é inteiro e pertence ao domínio binário; False
        para floats, strings, booleanos e inteiros fora do domínio.
    """
    return (
        isinstance(valor, Integral)
        and not isinstance(valor, bool)
        and valor in DOMINIO_ROTULO
    )
