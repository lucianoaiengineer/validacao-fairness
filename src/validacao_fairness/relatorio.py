"""
Arquivo: relatorio.py

Gera artefatos finais de métricas, fairness e interpretação técnica.

Fluxo:
1. Recebe estruturas de resultado já calculadas pelo pipeline.
2. Serializa métricas consolidadas em JSON.
3. Salva taxas de fairness por grupo em CSV.
4. Escreve interpretação em Markdown sem inventar valores.
5. Explicita limitações e evita afirmações causais indevidas.

Autor: Luciano Magalhães
Setembro 2026
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from validacao_fairness.equidade import ResultadoEquidade
from validacao_fairness.estatistica import ComparacaoBootstrap, IntervaloBootstrap


def salvar_metricas_json(metricas: dict[str, Any], caminho: Path) -> None:
    """
    Salva métricas calculadas em arquivo JSON UTF-8.

    Parâmetros:
        metricas: Dicionário serializável com resultados reais do pipeline.
        caminho: Caminho de destino do arquivo JSON.

    Retorno:
        None. A função cria o diretório de destino quando necessário e grava o
        arquivo.

    Exceções:
        TypeError: Lançado quando `metricas` contém objeto não serializável.
        OSError: Lançado quando há falha de criação ou escrita no destino.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# CSV separado facilita inspeção direta das taxas de fairness por grupo.
def salvar_equidade_csv(resultado: ResultadoEquidade, caminho: Path) -> None:
    """
    Salva taxas positivas por grupo em arquivo CSV.

    Parâmetros:
        resultado: ResultadoEquidade calculado sobre as predições do teste.
        caminho: Caminho de destino do arquivo CSV.

    Retorno:
        None. A função cria o diretório de destino quando necessário e grava o
        arquivo.

    Exceções:
        OSError: Lançado quando há falha de criação ou escrita no destino.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=[
                "grupo",
                "n_registros",
                "predicoes_positivas",
                "taxa_positiva",
            ],
        )
        escritor.writeheader()
        for taxa in resultado.taxas_por_grupo:
            escritor.writerow(
                {
                    "grupo": taxa.grupo,
                    "n_registros": taxa.n_registros,
                    "predicoes_positivas": taxa.predicoes_positivas,
                    "taxa_positiva": f"{taxa.taxa_positiva:.10f}",
                }
            )


def escrever_interpretacao(
    caminho: Path,
    acuracia_principal: float,
    intervalo: IntervaloBootstrap,
    acuracia_referencia: float,
    comparacao: ComparacaoBootstrap,
    equidade: ResultadoEquidade,
    registros_teste: int,
    ausentes_permitidos_treino: dict[str, int],
    ausentes_permitidos_teste: dict[str, int],
    duplicados_treino: int,
    duplicados_teste: int,
) -> None:
    """
    Escreve interpretação objetiva dos resultados calculados.

    Parâmetros:
        caminho: Caminho de destino do relatório Markdown.
        acuracia_principal: Acurácia pontual da regressão logística.
        intervalo: Intervalo bootstrap da acurácia do modelo principal.
        acuracia_referencia: Acurácia pontual do `DummyClassifier`.
        comparacao: Resultado da comparação bootstrap pareada.
        equidade: Resultado da paridade demográfica.
        registros_teste: Quantidade de registros usados na avaliação final.
        ausentes_permitidos_treino: Ausências permitidas observadas no treino.
        ausentes_permitidos_teste: Ausências permitidas observadas no teste.
        duplicados_treino: Quantidade de duplicados totais removidos do treino.
        duplicados_teste: Quantidade de duplicados totais removidos do teste.

    Retorno:
        None. A função grava o relatório interpretativo em Markdown.

    Exceções:
        OSError: Lançado quando há falha de criação ou escrita no destino.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    interpretacao_comparacao = _interpretar_comparacao(comparacao)
    ausentes_treino = _formatar_ausentes(ausentes_permitidos_treino)
    ausentes_teste = _formatar_ausentes(ausentes_permitidos_teste)
    taxas_por_grupo = {
        taxa.grupo: taxa.taxa_positiva for taxa in equidade.taxas_por_grupo
    }
    diferenca_sexo_pp = (
        taxas_por_grupo["Masculino"] - taxas_por_grupo["Feminino"]
    ) * 100.0
    linhas_taxas = "\n".join(
        (
            f"- {taxa.grupo}: {taxa.taxa_positiva:.4f} "
            f"({taxa.predicoes_positivas}/{taxa.n_registros})"
        )
        for taxa in equidade.taxas_por_grupo
    )

    texto = f"""# Interpretação dos resultados

## Qualidade dos dados

As validações de schema, tipos, colunas críticas e faixas foram aprovadas para
os conjuntos de treino e teste. As ausências observadas ficaram restritas às
colunas permitidas e foram tratadas no pré-processamento por imputação
categórica com valor mais frequente.

- Treino: {ausentes_treino}
- Teste: {ausentes_teste}

Foram encontrados {duplicados_treino} registros totalmente duplicados no treino
e {duplicados_teste} no teste. Como o dataset não possui chave natural
confiável, o tratamento adotado foi remover apenas duplicados totais antes da
modelagem, sem criar regra artificial de unicidade.

## Desempenho

O modelo principal, uma regressão logística com pré-processamento em pipeline,
obteve acurácia de {acuracia_principal:.4f} no conjunto de teste oficial
após validação e remoção de duplicados totais. O modelo de referência
`DummyClassifier` com estratégia `most_frequent` obteve acurácia de
{acuracia_referencia:.4f}.

O intervalo de confiança bootstrap de 95% para a acurácia do modelo principal
foi [{intervalo.limite_inferior:.4f}, {intervalo.limite_superior:.4f}], com
{intervalo.reamostragens} reamostragens. Esse intervalo expressa a incerteza
amostral da métrica no conjunto de teste, não uma garantia sobre populações
futuras.

Na comparação bootstrap pareada, a diferença pontual de acurácia
(regressão logística menos referência) foi {comparacao.diferenca_pontual:.4f},
com intervalo de 95% [{comparacao.limite_inferior:.4f},
{comparacao.limite_superior:.4f}]. {interpretacao_comparacao}

## Fairness

A paridade demográfica foi avaliada sobre as predições do conjunto de teste
oficial, considerando {registros_teste} registros utilizados. As taxas de
predições positivas por grupo foram:

{linhas_taxas}

A taxa positiva do grupo Masculino foi superior à do grupo Feminino em
aproximadamente {diferenca_sexo_pp:.2f} pontos percentuais.

O gap absoluto de paridade demográfica foi {equidade.gap:.4f}. Perante o
limite operacional de {equidade.limite:.2f}, adotado nesta análise e não
como regra legal ou universal, a paridade demográfica foi classificada como
**{equidade.status}**.

## Limitações

O dataset Adult é histórico, derivado de dados censitários dos Estados Unidos,
e reflete condições sociais, econômicas e de coleta do período em que foi
produzido. A exclusão direta de `sexo` e `raca` dos preditores não garante
ausência de disparidade, pois outras variáveis podem funcionar como proxies.

Diferenças observadas nas taxas positivas indicam disparidade estatística sob
o critério operacional adotado, mas não provam causalidade, discriminação
individual ou intenção discriminatória.
"""
    caminho.write_text(texto, encoding="utf-8")


def _formatar_ausentes(ausentes: dict[str, int]) -> str:
    """
    Formata ausências permitidas para o relatório textual.

    Parâmetros:
        ausentes: Dicionário com colunas permitidas e respectivas contagens.

    Retorno:
        Texto curto com contagens por coluna ou indicação de ausência total.
    """
    if not ausentes:
        return "sem ausências permitidas observadas"
    return ", ".join(
        f"{coluna}={quantidade}" for coluna, quantidade in ausentes.items()
    )


# Interpretação segue apenas o IC da diferença, sem mencionar p-valor inexistente.
def _interpretar_comparacao(comparacao: ComparacaoBootstrap) -> str:
    """
    Interpreta o intervalo bootstrap da diferença de acurácias.

    Parâmetros:
        comparacao: Estrutura com diferença pontual e limites do intervalo.

    Retorno:
        Texto curto explicando se o intervalo indica superioridade, empate
        estatisticamente não comprovado ou inferioridade do modelo principal.
    """
    if comparacao.limite_inferior > 0.0:
        return (
            "Como o intervalo ficou totalmente acima de zero, há evidência de "
            "desempenho superior do modelo principal."
        )
    if comparacao.limite_inferior <= 0.0 <= comparacao.limite_superior:
        return (
            "Como o intervalo contém zero, a diferença não foi comprovada "
            "estatisticamente."
        )
    return (
        "Como o intervalo ficou totalmente abaixo de zero, há evidência de "
        "desempenho inferior do modelo principal."
    )
