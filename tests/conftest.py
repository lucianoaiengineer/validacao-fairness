"""
Arquivo: conftest.py

Define fixtures controladas para os testes automatizados.

Fluxo:
1. Cria uma amostra sintética com o schema em português do Adult.
2. Inclui ausentes apenas em colunas permitidas pelo pré-processamento.
3. Mantém rótulo e atributo sensível completos para testes de validação.
4. Evita qualquer dependência de internet ou dos arquivos brutos oficiais.

Autor: Luciano Magalhães
Setembro 2026
"""

import pandas as pd
import pytest


@pytest.fixture()
def dados_adult_validos() -> pd.DataFrame:
    """
    Cria amostra controlada no schema português do Adult.

    Parâmetros:
        Não recebe parâmetros diretamente. A fixture é injetada pelo pytest nos
        testes que declaram `dados_adult_validos`.

    Retorno:
        DataFrame sintético com colunas, tipos, domínios e faixas compatíveis
        com as validações do projeto.
    """
    return pd.DataFrame(
        {
            "idade": [25, 38, 44, 52, 31, 47, 29, 60],
            "classe_trabalho": [
                "Private",
                "Self-emp",
                "Private",
                None,
                "Private",
                "State-gov",
                "Private",
                "Federal-gov",
            ],
            "peso_amostral": [
                100_000,
                120_000,
                95_000,
                80_000,
                110_000,
                130_000,
                90_000,
                70_000,
            ],
            "escolaridade": [
                "HS-grad",
                "Bachelors",
                "Masters",
                "HS-grad",
                "Some-college",
                "Doctorate",
                "Bachelors",
                "Masters",
            ],
            "nivel_escolaridade": [9, 13, 14, 9, 10, 16, 13, 14],
            "estado_civil": [
                "Never-married",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Married-civ-spouse",
                "Never-married",
                "Married-civ-spouse",
            ],
            "ocupacao": [
                "Adm-clerical",
                "Exec-managerial",
                "Prof-specialty",
                None,
                "Sales",
                "Prof-specialty",
                "Tech-support",
                "Exec-managerial",
            ],
            "relacionamento": [
                "Own-child",
                "Husband",
                "Wife",
                "Not-in-family",
                "Unmarried",
                "Husband",
                "Not-in-family",
                "Wife",
            ],
            "raca": [
                "White",
                "White",
                "Black",
                "White",
                "Asian-Pac-Islander",
                "White",
                "Black",
                "White",
            ],
            "sexo": [
                "Feminino",
                "Masculino",
                "Feminino",
                "Masculino",
                "Feminino",
                "Masculino",
                "Feminino",
                "Masculino",
            ],
            "ganho_capital": [
                0,
                0,
                5_000,
                0,
                0,
                10_000,
                0,
                0,
            ],
            "perda_capital": [0, 0, 0, 0, 0, 0, 0, 1_900],
            "horas_semanais": [40, 45, 50, 30, 38, 55, 40, 45],
            "pais_origem": [
                "United-States",
                "United-States",
                "United-States",
                None,
                "United-States",
                "Canada",
                "United-States",
                "United-States",
            ],
            "renda_acima_50k": [0, 1, 1, 0, 0, 1, 0, 1],
        }
    )
