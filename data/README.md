# Dados brutos

Este diretório armazena cópias locais dos arquivos oficiais `adult.data` e
`adult.test` do dataset público Adult, também conhecido como Census Income.

Fonte oficial: <https://archive.ics.uci.edu/dataset/2/adult>

DOI: <https://doi.org/10.24432/C5XW20>

Licença indicada pelo UCI Machine Learning Repository: CC BY 4.0.

Os arquivos brutos não são versionados no Git. Para obtê-los, execute a partir
da raiz do projeto:

```bash
python scripts/baixar_dados.py
```

O script baixa os arquivos por HTTPS do domínio oficial
`archive.ics.uci.edu` e reutiliza as cópias existentes em execuções
posteriores, evitando novo download desnecessário.

