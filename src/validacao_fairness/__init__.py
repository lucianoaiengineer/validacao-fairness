"""
Arquivo: __init__.py

Inicialização do pacote `validacao_fairness`.

Responsabilidade:
1. Expor metadados mínimos do pacote.
2. Manter uma interface pública explícita para identificação da versão.
3. Evitar execução de lógica pesada no momento de importação.

Uso:
import validacao_fairness

Autor: Luciano Magalhães
Setembro 2026
"""

# A versão fica centralizada para permitir identificação simples do pacote.
__all__ = [
    "__version__",
]

__version__ = "0.1.0"
