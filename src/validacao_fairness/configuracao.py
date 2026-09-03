"""
Arquivo: configuracao.py

Configura parâmetros, caminhos, nomes de colunas e domínios do projeto.

Responsabilidade:
1. Centralizar constantes para evitar duplicação entre módulos.
2. Registrar metadados oficiais do dataset Adult.
3. Definir colunas usadas em validação, modelagem e fairness.
4. Fixar sementes e limites operacionais para reprodutibilidade.

Observação:
Este módulo não executa leitura, escrita, download ou modelagem. Ele apenas
declara valores usados por outras partes da suíte.

Autor: Luciano Magalhães
Setembro 2026
"""

from pathlib import Path

# Parâmetros determinísticos exigidos para reprodutibilidade estatística.
SEMENTE: int = 42
REAMOSTRAGENS_BOOTSTRAP: int = 2_000
NIVEL_CONFIANCA: float = 0.95
LIMITE_PARIDADE_DEMOGRAFICA: float = 0.10

# Caminhos derivados da localização do pacote dentro da raiz do projeto.
RAIZ_PROJETO: Path = Path(__file__).resolve().parents[2]
DIRETORIO_DADOS: Path = RAIZ_PROJETO / "data"
DIRETORIO_DADOS_BRUTOS: Path = DIRETORIO_DADOS / "brutos"
DIRETORIO_RESULTADOS: Path = RAIZ_PROJETO / "resultados"

# Metadados oficiais usados em documentação e artefatos de resultado.
URL_DATASET_UCI: str = "https://archive.ics.uci.edu/dataset/2/adult"
DOI_DATASET: str = "https://doi.org/10.24432/C5XW20"
LICENCA_DATASET: str = "CC BY 4.0"

# Endereços oficiais dos dois arquivos preservando a separação treino/teste.
ARQUIVOS_ADULT: dict[str, str] = {
    "adult.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    ),
    "adult.test": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    ),
}

# Nomes originais do UCI Adult, mantidos aqui para rastreabilidade técnica.
COLUNAS_ORIGINAIS: tuple[str, ...] = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)

# Mapeamento para nomes em português usados internamente no projeto.
MAPA_COLUNAS_PORTUGUES: dict[str, str] = {
    "age": "idade",
    "workclass": "classe_trabalho",
    "fnlwgt": "peso_amostral",
    "education": "escolaridade",
    "education-num": "nivel_escolaridade",
    "marital-status": "estado_civil",
    "occupation": "ocupacao",
    "relationship": "relacionamento",
    "race": "raca",
    "sex": "sexo",
    "capital-gain": "ganho_capital",
    "capital-loss": "perda_capital",
    "hours-per-week": "horas_semanais",
    "native-country": "pais_origem",
    "income": "renda_acima_50k",
}

# Colunas e papéis principais da análise.
COLUNAS_OBRIGATORIAS: tuple[str, ...] = tuple(MAPA_COLUNAS_PORTUGUES.values())
ROTULO: str = "renda_acima_50k"
ATRIBUTO_SENSIVEL: str = "sexo"
PREDICAO: str = "predicao_renda_acima_50k"

# Tipos de variáveis esperados após a leitura e normalização do Adult.
COLUNAS_NUMERICAS: tuple[str, ...] = (
    "idade",
    "peso_amostral",
    "nivel_escolaridade",
    "ganho_capital",
    "perda_capital",
    "horas_semanais",
)

COLUNAS_CATEGORICAS: tuple[str, ...] = (
    "classe_trabalho",
    "escolaridade",
    "estado_civil",
    "ocupacao",
    "relacionamento",
    "raca",
    "sexo",
    "pais_origem",
)

# Colunas críticas não podem ter ausentes porque sustentam validação e métricas.
COLUNAS_CRITICAS_SEM_AUSENTES: tuple[str, ...] = (
    "idade",
    "sexo",
    "renda_acima_50k",
)

# Somente estas colunas podem ter ausências e serão imputadas no pipeline.
COLUNAS_AUSENTES_PERMITIDOS: frozenset[str] = frozenset(
    {
        "classe_trabalho",
        "ocupacao",
        "pais_origem",
    }
)

# Domínios controlados usados para validação explícita de rótulo e sexo.
GRUPOS_SEXO: tuple[str, str] = ("Feminino", "Masculino")
DOMINIO_ROTULO: set[int] = {0, 1}
DOMINIO_SEXO: set[str] = set(GRUPOS_SEXO)

# Atributos sensíveis ficam fora do treinamento e entram somente em fairness.
COLUNAS_EXCLUIDAS_DO_MODELO: tuple[str, ...] = (
    "sexo",
    "raca",
    "renda_acima_50k",
)

COLUNAS_NUMERICAS_MODELO: tuple[str, ...] = tuple(
    coluna for coluna in COLUNAS_NUMERICAS if coluna not in COLUNAS_EXCLUIDAS_DO_MODELO
)

COLUNAS_CATEGORICAS_MODELO: tuple[str, ...] = tuple(
    coluna
    for coluna in COLUNAS_CATEGORICAS
    if coluna not in COLUNAS_EXCLUIDAS_DO_MODELO
)
