# Análise Preditiva e Machine Learning - ENEM 

Projeto desenvolvido para análise, tratamento de dados e modelagem de Machine Learning utilizando os microdados oficiais do ENEM  disponibilizados pelo INEP.

## 📌 Estrutura do Repositório
- `DADOS/`: Arquivos originais de microdados (ignorados pelo `.gitignore` devido ao tamanho).
- `DICIONARIO/`: Documentação oficial dos dados.
- `src/`: Scripts modulares em Python contendo o pipeline de pré-processamento, limpeza, tratamentos customizados e codificação de variáveis.

## 🚀 Pipeline de Dados (`src/`)
O projeto conta com um pipeline modularizado que realiza:
1. **Carregamento e Limpeza Inicial**: Leitura otimizada do arquivo CSV e remoção de colunas irrelevantes para o escopo do modelo.
2. **Transformação de Variáveis Socioeconômicas**: Agrupamento e padronização de faixas de escolaridade (`Q001`, `Q002`) e renda familiar (`Q006`).
3. **Label Encoding**: Conversão de variáveis nominais (como sexo e estados) para formato numérico.
4. **Ordinal Encoder**: Codificação sequencial de variáveis categóricas ordenadas (questionário socioeconômico).

## 📥 Como Reproduzir
1. Clone este repositório:
   ```bash
   git clone [https://github.com/SEU-USUARIO/enem-machine-learning.git](https://github.com/SEU-USUARIO/enem-machine-learning.git)
