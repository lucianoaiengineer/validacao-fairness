"""
Arquivo: executar_analise.py

Ponto de entrada para executar a suíte completa de validação e fairness.

Fluxo:
1. Carrega os arquivos Adult já disponíveis em `data/brutos`.
2. Valida qualidade e schema dos dados.
3. Treina o modelo principal e o modelo de referência.
4. Calcula estatística bootstrap e paridade demográfica.
5. Grava métricas, CSV de fairness e interpretação textual em `resultados`.

Execução:
python scripts/executar_analise.py

Autor: Luciano Magalhães
Setembro 2026
"""

from validacao_fairness.pipeline import executar_pipeline


def main() -> None:
    """
    Executa o pipeline completo e apresenta um resumo operacional.

    Parâmetros:
        Não recebe parâmetros. O pipeline utiliza os caminhos e limites
        configurados no pacote `validacao_fairness`.

    Retorno:
        None. A função grava os artefatos calculados e imprime acurácia, gap
        de fairness e status de paridade demográfica.

    Exceções:
        Propaga erros de validação, leitura ou modelagem que impeçam a análise
        completa de produzir resultados confiáveis.
    """
    resultado = executar_pipeline()
    print(
        "Analise concluida: "
        f"acuracia={resultado.acuracia_principal:.4f}, "
        f"gap_fairness={resultado.equidade.gap:.4f}, "
        f"status={resultado.equidade.status}"
    )


if __name__ == "__main__":
    main()
