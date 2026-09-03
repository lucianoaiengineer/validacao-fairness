"""
Arquivo: test_pipeline.py

Testa a integração do pipeline sem acessar internet ou dados brutos reais.

Fluxo:
1. Monta treino e teste sintéticos a partir da fixture Adult válida.
2. Substitui o carregador real por função controlada via monkeypatch.
3. Executa validação, modelagem, estatística, fairness e relatórios.
4. Confirma geração dos arquivos esperados.
5. Confirma que `sexo` e `raca` não entram como preditores.

Autor: Luciano Magalhães
Setembro 2026
"""

from pathlib import Path

import pandas as pd

from validacao_fairness.dados import DatasetAdult
from validacao_fairness.modelagem import colunas_preditoras_modelo
from validacao_fairness.pipeline import executar_pipeline


def test_pipeline_gera_resultados_com_dados_controlados(
    dados_adult_validos: pd.DataFrame,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    Executa o pipeline completo sem internet nem arquivos brutos.

    Parâmetros:
        dados_adult_validos: Fixture com DataFrame sintético válido.
        tmp_path: Diretório temporário do pytest para artefatos de resultado.
        monkeypatch: Fixture do pytest usada para substituir o carregador real.

    Retorno:
        None. Asserções verificam volumes resumidos e criação dos artefatos
        finais esperados.
    """
    treino = pd.concat([dados_adult_validos] * 4, ignore_index=True)
    teste = dados_adult_validos.copy()

    # Função local preserva o contrato do carregador oficial para monkeypatch.
    def carregar_controlado() -> DatasetAdult:
        """
        Retorna datasets sintéticos no formato esperado pelo pipeline.

        Parâmetros:
            Não recebe parâmetros. Usa os DataFrames fechados no escopo do
            teste.

        Retorno:
            DatasetAdult com treino ampliado e teste controlado.
        """
        return DatasetAdult(treino=treino, teste=teste)

    monkeypatch.setattr(
        "validacao_fairness.pipeline.carregar_datasets_adult",
        carregar_controlado,
    )

    resultado = executar_pipeline(
        diretorio_resultados=tmp_path,
        reamostragens=100,
        semente=42,
    )

    assert resultado.resumo_dados.treino_bruto == 32
    assert (tmp_path / "metricas.json").exists()
    assert (tmp_path / "equidade_por_grupo.csv").exists()
    assert (tmp_path / "interpretacao.md").exists()


# Regressão contra inclusão acidental de atributos sensíveis no modelo.
def test_modelo_nao_usa_sexo_nem_raca() -> None:
    """
    Confirma exclusão direta de sexo e raça dos preditores.

    Parâmetros:
        Não recebe parâmetros. A função consulta a configuração pública das
        colunas usadas pelo modelo.

    Retorno:
        None. Asserções impedem regressão que inclua atributos sensíveis no
        treinamento.
    """
    numericas, categoricas = colunas_preditoras_modelo()

    assert "sexo" not in numericas
    assert "sexo" not in categoricas
    assert "raca" not in numericas
    assert "raca" not in categoricas
