"""
Arquivo: baixar_dados.py

Ponto de entrada para obtenção dos arquivos oficiais do dataset Adult.

Fluxo:
1. Aciona a rotina centralizada de download em `validacao_fairness.dados`.
2. Reutiliza arquivos existentes quando eles já foram baixados.
3. Exibe apenas uma mensagem curta com os nomes dos arquivos disponíveis.

Execução:
python scripts/baixar_dados.py

Autor: Luciano Magalhães
Setembro 2026
"""

from validacao_fairness.dados import baixar_arquivos_adult


# Ponto de entrada simples para manter o download separado da análise.
def main() -> None:
    """
    Executa o download idempotente dos dados brutos oficiais.

    Parâmetros:
        Não recebe parâmetros. A configuração de destino é definida no módulo
        `validacao_fairness.configuracao`.

    Retorno:
        None. A função apenas garante que os arquivos estejam locais e imprime
        um resumo curto para o terminal.

    Exceções:
        Propaga erros de rede, permissão ou escrita levantados pela rotina de
        download, preservando a causa original para diagnóstico.
    """
    arquivos = baixar_arquivos_adult()
    nomes = ", ".join(caminho.name for caminho in arquivos)
    print(f"Arquivos prontos em data/brutos: {nomes}")


if __name__ == "__main__":
    main()
