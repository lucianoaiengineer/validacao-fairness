"""
Arquivo: dados.py

Leitura, download e preparação inicial dos arquivos oficiais do Adult.

Fluxo:
1. Baixa `adult.data` e `adult.test` do domínio oficial quando necessário.
2. Reutiliza cópias locais para evitar download repetido.
3. Lê os arquivos preservando a separação oficial de treino e teste.
4. Trata `?` como valor ausente e remove espaços residuais dos campos.
5. Normaliza rótulos, grupos de sexo e tipos numéricos para a validação.

Autor: Luciano Magalhães
Setembro 2026
"""

from dataclasses import dataclass
from pathlib import Path
from urllib import request

import pandas as pd

from validacao_fairness.configuracao import (
    ARQUIVOS_ADULT,
    COLUNAS_CATEGORICAS,
    COLUNAS_NUMERICAS,
    COLUNAS_OBRIGATORIAS,
    DIRETORIO_DADOS_BRUTOS,
    ROTULO,
)


class DadosAdultNaoEncontradosError(FileNotFoundError):
    """
    Indica ausência dos arquivos brutos oficiais esperados no projeto.

    Uso:
        A exceção é lançada quando `adult.data` ou `adult.test` não existem no
        diretório configurado em `data/brutos`.

    Parâmetros:
        Herda os parâmetros de `FileNotFoundError`, normalmente uma mensagem
        textual com orientação de execução do script de download.
    """


@dataclass(frozen=True)
class DatasetAdult:
    """
    Representa os conjuntos oficiais de treino e teste do Adult.

    Atributos:
        treino: DataFrame carregado de `adult.data`, usado para ajuste dos
            modelos supervisionados.
        teste: DataFrame carregado de `adult.test`, usado para avaliação,
            estatística bootstrap e fairness.

    Observação:
        A estrutura é imutável para evitar troca acidental entre treino e
        teste durante a execução do pipeline.
    """

    treino: pd.DataFrame
    teste: pd.DataFrame


# O download é idempotente: arquivos existentes são reutilizados por padrão.
def baixar_arquivos_adult(
    diretorio_destino: Path = DIRETORIO_DADOS_BRUTOS,
    forcar: bool = False,
) -> list[Path]:
    """
    Baixa os arquivos oficiais do Adult ou reutiliza cópias locais.

    Parâmetros:
        diretorio_destino: Diretório onde `adult.data` e `adult.test` devem
            ser armazenados.
        forcar: Quando verdadeiro, baixa novamente os arquivos mesmo que já
            existam localmente.

    Retorno:
        Lista de caminhos locais correspondentes aos arquivos prontos para
        leitura.

    Exceções:
        Propaga exceções de rede, tempo limite e escrita em disco levantadas
        por `urllib` ou `pathlib`.
    """
    diretorio_destino.mkdir(parents=True, exist_ok=True)
    caminhos: list[Path] = []

    for nome_arquivo, url in ARQUIVOS_ADULT.items():
        destino = diretorio_destino / nome_arquivo
        if destino.exists() and destino.stat().st_size > 0 and not forcar:
            caminhos.append(destino)
            continue

        # `urllib` da biblioteca padrão evita dependência extra só para download.
        with request.urlopen(url, timeout=60) as resposta:
            conteudo = resposta.read()
        destino.write_bytes(conteudo)
        caminhos.append(destino)

    return caminhos


# O carregamento exige os dois arquivos oficiais para preservar o desenho do UCI.
def carregar_datasets_adult(
    diretorio_origem: Path = DIRETORIO_DADOS_BRUTOS,
) -> DatasetAdult:
    """
    Carrega os arquivos oficiais `adult.data` e `adult.test` locais.

    Parâmetros:
        diretorio_origem: Diretório onde os arquivos brutos oficiais devem
            estar disponíveis.

    Retorno:
        Instância de `DatasetAdult` contendo DataFrames de treino e teste
        normalizados.

    Exceções:
        DadosAdultNaoEncontradosError: Lançada quando um dos arquivos oficiais
            esperados não está presente no diretório de origem.
    """
    caminho_treino = diretorio_origem / "adult.data"
    caminho_teste = diretorio_origem / "adult.test"

    faltantes = [
        caminho
        for caminho in (caminho_treino, caminho_teste)
        if not caminho.exists()
    ]
    if faltantes:
        nomes = ", ".join(caminho.name for caminho in faltantes)
        raise DadosAdultNaoEncontradosError(
            "Arquivos brutos ausentes em data/brutos. "
            f"Execute `python scripts/baixar_dados.py`. Faltantes: {nomes}."
        )

    return DatasetAdult(
        treino=carregar_arquivo_adult(caminho_treino),
        teste=carregar_arquivo_adult(caminho_teste),
    )


def carregar_arquivo_adult(caminho: Path) -> pd.DataFrame:
    """
    Lê um arquivo Adult oficial e normaliza nomes, ausentes e rótulo.

    Parâmetros:
        caminho: Caminho do arquivo `adult.data` ou `adult.test` a ser lido.

    Retorno:
        DataFrame com colunas em português, rótulo binário, valores `?`
        convertidos para ausentes e grupos de sexo traduzidos para português.

    Exceções:
        Propaga erros de leitura do pandas e erros de sistema associados ao
        acesso ao arquivo informado.
    """
    # `comment="|"` remove a linha de cabeçalho/comentário presente no teste.
    dados = pd.read_csv(
        caminho,
        names=list(COLUNAS_OBRIGATORIAS),
        header=None,
        na_values=["?"],
        skipinitialspace=True,
        comment="|",
    )
    dados = dados.dropna(how="all").reset_index(drop=True)

    # Campos categóricos chegam com espaços e, no teste, rótulos terminados em ".".
    for coluna in (*COLUNAS_CATEGORICAS, ROTULO):
        dados[coluna] = dados[coluna].map(_limpar_texto_categorico)

    # Conversão explícita permite que validações capturem valores inválidos.
    for coluna in COLUNAS_NUMERICAS:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    # Tradução dos grupos sensíveis facilita leitura dos relatórios em português.
    dados["sexo"] = dados["sexo"].map(
        {
            "Female": "Feminino",
            "Male": "Masculino",
        },
    )
    dados[ROTULO] = dados[ROTULO].map(
        {
            "<=50K": 0,
            ">50K": 1,
        },
    )

    return dados


def _limpar_texto_categorico(valor: object) -> object:
    """
    Remove espaços e ponto final dos campos textuais oficiais.

    Parâmetros:
        valor: Valor bruto lido de uma coluna categórica ou do rótulo.

    Retorno:
        String limpa quando o valor é textual; caso contrário, retorna o valor
        original para preservar ausentes e outros marcadores não textuais.
    """
    if isinstance(valor, str):
        return valor.strip().removesuffix(".")
    return valor
