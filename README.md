# Análise Preditiva - Trabalho A3

Projeto acadêmico de Análise Preditiva de Séries Temporais desenvolvido como trabalho A3, seguindo o Guia Prático de Análise de Séries Temporais (roteiro passo a passo). O foco é aplicar todo o fluxo analítico, da organização dos dados até a avaliação de modelos, em três conjuntos de dados reais de vendas.

## Integrantes

- Laiane P. Da Silva - 32317230
- Gabriela de Oliveira Tavares - 324116412
- Thiago Thadeu Leal Santos - 324231402
- Lucas de Souza Antunes - 324214906
- Larissa Felipe Reis - 32419802

## Estrutura do Projeto

### Datasets

- **Updated_sales.csv**: Dataset de vendas diárias (linhas de pedido) para a Análise 1
- **Sales_Data/**: Arquivos Sales_January_2019.csv a Sales_December_2019.csv, com vendas mensais de 2019, usados na Análise 2
- **Retail and wherehouse Sale.csv**: Dataset consolidado por YEAR e MONTH, com colunas de vendas de varejo e armazém, usado na Análise 3

### Scripts Principais

- **gerar_relatorios_series.py**: Análises mensais (Updated_sales e Retail & Warehouse)
- **relatorio_semanal.py**: Análise semanal do Updated_sales.csv
- **analise_2_mensal.py**: Análise mensal do ano fechado Sales_2019 (12 meses)
- **index_geral.py**: Gera o índice geral em HTML (menu principal)

### Outros Arquivos

- **relatorios/**: Pasta de saída para todos os relatórios HTML, imagens e índices
- **requirements.txt**: Lista de dependências Python

## Objetivos das Análises

As três análises seguem a mesma filosofia metodológica, com granularidades e horizontes adaptados a cada dataset.

### Análise 1: Updated_sales (dataset completo e diário)

**Objetivo**: Analisar e prever a receita de vendas.

- **Semanal**: Horizonte de 12 semanas à frente
- **Mensal**: Horizonte de 3 meses à frente (previsão ilustrativa, devido à série curta e irregular)

**Frequências utilizadas**:
- Diário: Utilizado apenas na preparação
- Semanal (W-MON): Principal para previsão
- Mensal (MS): Utilizada para visão agregada

### Análise 2: Sales_2019 (12 meses)

**Objetivo**: Analisar o comportamento da receita mensal de 2019 e prever os três meses subsequentes.

- **Dataset**: Concatenação dos arquivos Sales_*.csv de 2019
- **Frequência**: Mensal

### Análise 3: Retail & Warehouse (9 meses)

**Objetivo**: Prever a receita mensal consolidada (soma de Retail Sales, Retail Transfers e Warehouse Sales) para três meses à frente.

- **Dataset**: Retail and wherehouse Sale.csv
- **Frequência**: Mensal (série curta, nove meses)

## Metodologia

As análises seguem o roteiro didático estabelecido, compreendendo as seguintes etapas:

### 1. Definição do Objetivo

Especificação da variável alvo, horizonte de previsão e frequência (diária, semanal, mensal).

### 2. Carregamento e Organização dos Dados

- Conversão de datas
- Ordenação temporal
- Remoção de duplicatas
- Ajuste de frequência

### 3. Tratamento de Valores Faltantes

- Contagem antes e depois da interpolação
- Interpolação linear para receita
- Forward/backward fill para contagens

### 4. Identificação de Outliers

- Detecção por escore Z e intervalo interquartílico (IQR)
- Winsorização via clip para reduzir distorções sem remover eventos reais

### 5. Estatísticas Descritivas

- Cálculo de média, desvio padrão, mínimos, máximos e quantis
- Resumos formatados em reais para as séries de receita

### 6. Visualização da Série

Gráficos de linha com histórico, treino e teste.

### 7. Decomposição da Série

Decomposição aditiva semanal (quando há dados suficientes) em tendência, sazonalidade e ruído.

### 8. Testes de Estacionariedade

- Testes ADF (Dickey-Fuller)
- KPSS
- Interpretação textual dos resultados

### 9. Seleção de Modelos

Decisão entre modelos com ou sem sazonalidade, modelos ingênuos e modelos clássicos (Holt-Winters, SARIMAX).

### 10. Ajuste de Modelos

- **Holt-Winters**: Tendência aditiva, com ou sem sazonalidade
- **SARIMAX(1,1,1)**: Com sazonalidade quando há dados suficientes

### 11. Divisão Temporal

Split 80% treino e 20% teste (ou 9/3 meses para Sales_2019). A ordem temporal é preservada.

### 12. Avaliação de Desempenho

- Métricas MAE, RMSE e MAPE no conjunto de teste
- Comparação entre Holt-Winters e SARIMAX

### 13. Análise de Resíduos

- Geração de gráficos
- Testes Ljung-Box, Jarque-Bera e Breusch-Pagan

### 14. Detecção de Anomalias

Tratamento de outliers conforme metodologia estabelecida.

### 15. Testes Estatísticos Finais

- **Ljung-Box**: Autocorrelação
- **Jarque-Bera**: Normalidade
- **Breusch-Pagan**: Heterocedasticidade

## Scripts e Relatórios Gerados

### gerar_relatorios_series.py - Análises Mensais

Gera relatórios mensais em HTML na pasta relatorios/.

**Para Updated_sales.csv**:
- Agregação mensal da receita
- Modelo ingênuo (repete o último valor observado)
- Cálculo de métricas MAE, RMSE e MAPE
- Gráficos de série histórica, treino versus teste e previsão de três meses
- Testes Ljung-Box, Jarque-Bera e Breusch-Pagan sobre os resíduos

**Para Retail and wherehouse Sale.csv**:
- Cálculo de total_sales = RETAIL SALES + RETAIL TRANSFERS + WAREHOUSE SALES
- Modelos Holt-Winters ou SARIMAX (sem sazonalidade, devido à curta série)
- Avaliação com métricas e previsão de três meses

**Saídas**:
- relatorios/relatorio_vendas_mensal_receita.html
- relatorios/relatorio_retail_mensal_total.html
- relatorios/index.html
- CSVs previsao_3meses_*.csv
- Imagens relatorios/figuras_*/*.png

### relatorio_semanal.py - Análise Semanal (Updated_sales)

Executa a análise semanal para o dataset mais detalhado.

**Processamento**:
- Agregação semanal da receita, número de pedidos e unidades
- Tratamento de valores faltantes e outliers
- Ajuste de Holt-Winters e SARIMAX com ou sem sazonalidade (52 semanas)
- Comparação de modelos e escolha do melhor
- Previsão de 12 semanas à frente
- Testes Ljung-Box, Jarque-Bera e Breusch-Pagan
- Testes ADF e KPSS para estacionariedade

**Saídas**:
- relatorios/relatorio_semanal_receita.html
- relatorios/index_semanal.html
- Imagens relatorios/figuras_semanal/*.png

### analise_2_mensal.py - Análise 2 (Sales_2019 - 12 meses)

Análise específica do ano de 2019.

**Processamento**:
- Concatenação dos dados mensais
- Divisão em nove meses de treino e três de teste
- Ajuste de Holt-Winters e SARIMAX sem sazonalidade
- Avaliação com MAE, RMSE e MAPE
- Geração de previsões e gráficos de resíduos

**Saídas**:
- relatorios/analise_2_mensal.html
- relatorios/plot_analise2.png
- relatorios/residuos_analise2.png

### index_geral.py - Índice Geral

Cria um painel HTML com os links para todos os relatórios.

**Saída**: relatorios/index_geral.html, contendo:
- Análise 1: Updated_sales (relatórios semanal e mensal)
- Análise 2: Sales_2019
- Análise 3: Retail & Warehouse
- Relatório de demografia (opcional)

## Como Executar o Projeto

### Pré-requisitos

- Python 3.10 ou superior (testado em 3.12)

### Instalação de Dependências
```bash
pip install -r requirements.txt
```

### Execução dos Scripts

**Gerar relatórios mensais (Updated_sales e Retail & Warehouse)**:
```bash
python gerar_relatorios_series.py
```

**Gerar relatório semanal (Updated_sales)**:
```bash
python relatorio_semanal.py
```

**Gerar análise mensal de Sales_2019**:
```bash
python analise_2_mensal.py
```

**Gerar índice geral (menu)**:
```bash
python index_geral.py
```

**Saídas**: Geradas automaticamente na pasta relatorios/

## Versionamento

### Configuração Inicial do Repositório

Após copiar todos os arquivos para o repositório local Analise-Preditiva, execute:
```bash
git add .
git commit -m "Trabalho A3 - Analise Preditiva de Series Temporais"
git remote add origin https://github.com/codewithsouza/Analise-Preditiva.git
git branch -M main
git push -u origin main
```

## Resultados e Conclusões

O projeto apresenta um pipeline completo de análise e previsão de séries temporais, abrangendo desde a limpeza e preparação dos dados até a modelagem e validação. 

Os modelos Holt-Winters e SARIMAX mostraram-se adequados para séries curtas e médias, respectivamente, com erros médios percentuais entre 10% e 15%. 

A metodologia implementada permite replicação, extensão e integração com ferramentas externas de visualização e atualização periódica de dados.

