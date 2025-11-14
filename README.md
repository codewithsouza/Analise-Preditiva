Analise Preditiva – Trabalho A3
================================

Projeto acadêmico de **Análise Preditiva de Séries Temporais** desenvolvido como trabalho A3, seguindo o _“Guia Prático de Análise de Séries Temporais”_ (roteiro passo a passo).  
O foco é aplicar todo o fluxo analítico – da organização dos dados até a avaliação de modelos – em três conjuntos de dados reais de vendas.

<small>
Integrantes:<br>
Laiane P. Da Silva - 32317230<br>
Gabriela de Oliveira Tavares - 324116412<br>
Thiago Thadeu Leal Santos - 324231402<br>
Marcos Vinicios Oliveira Santos - 325144209<br>
Lucas de Souza Antunes - 324214906<br>
Larissa Felipe Reis - 32419802<br>
</small>

---

## 1. Estrutura Geral do Projeto

Pasta de trabalho (raiz do repositório):

- `Updated_sales.csv`  
  Dataset de vendas diárias (linhas de pedido) para a **Análise 1**.

- `Sales_Data/`  
  Arquivos `Sales_January_2019.csv` … `Sales_December_2019.csv`, com vendas mensais de 2019, usados na **Análise 2**.

- `Retail and wherehouse Sale.csv`  
  Dataset consolidado por `YEAR` e `MONTH`, com colunas de vendas de varejo e armazém, usado na **Análise 3**.

- Scripts principais:
  - `gerar_relatorios_series.py` – análises mensais (Updated_sales e Retail & Warehouse).
  - `relatorio_semanal.py` – análise semanal do `Updated_sales.csv`.
  - `analise_2_mensal.py` – análise mensal do ano-fechado `Sales_2019` (12 meses).
  - `index_geral.py` – gera o índice geral em HTML (menu principal).

- Outros arquivos relevantes:
  - `relatorio_semanal.py` e `gerar_relatorios_series.py` utilizam a pasta `relatorios/` para salvar todos os HTMLs, imagens e índices.
  - `requirements.txt` – dependências Python.

---

## 2. Objetivos das Análises

As três análises seguem a mesma filosofia, com granularidades e horizontes adaptados a cada dataset.

### 2.1 Análise 1 – Updated_sales (dataset completo e diário)

- **Objetivo:** analisar e prever a receita de vendas:
  - **Semanal:** horizonte de 12 semanas à frente.
  - **Mensal:** horizonte de 3 meses à frente (previsão ilustra­ tiva, devido à série curta e irregular).
- **Frequências utilizadas:**
  - Diário → usado apenas na preparação.
  - Semanal (`W-MON`) – principal para previsão.
  - Mensal (`MS`) – para visão agregada e ilustrações de curto prazo.

### 2.2 Análise 2 – Sales_2019 (12 meses)

- **Objetivo:** analisar o comportamento da receita mensal de 2019 e prever os **3 meses subsequentes**.
- Dataset: concatenação dos arquivos `Sales_*.csv` de 2019.
- Frequência: **mensal**.

### 2.3 Análise 3 – Retail & Warehouse (9 meses)

- **Objetivo:** prever a receita mensal consolidada (soma de Retail Sales, Retail Transfers e Warehouse Sales) para **3 meses à frente**.
- Dataset: `Retail and wherehouse Sale.csv`.
- Frequência: **mensal** (série curta, 9 meses).

---

## 3. Roteiro Metodológico (Guia de Séries Temporais)

As análises seguem o roteiro didático do professor, contemplando os seguintes passos:

1. **Definir o objetivo**  
   - Variável‐alvo, horizonte de previsão e frequência (diária, semanal, mensal).

2. **Carregar e organizar os dados**  
   - Conversão de datas, ordenação temporal, remoção de duplicatas, ajuste de frequência (`asfreq`).

3. **Verificar dados faltantes**  
   - Contagem de missings antes/depois.  
   - Interpolação (receita) e _forward/backward fill_ para contagens.

4. **Identificar outliers**  
   - Detecção por **z‑score** e **IQR** (quartis).  
   - Winsorização via `clip` para reduzir distorções sem remover eventos reais.

5. **Estatísticas descritivas**  
   - `count`, média, desvio‐padrão, mínimo, máximo, quantis.  
   - Resumos formatados em reais (`R$`) para as séries de receita.

6. **Visualizar a série**  
   - Gráficos de linha (histórico semanal/mensal), com indicação de períodos de treino e teste.

7. **Decomposição da série**  
   - Decomposição aditiva semanal (quando há dados suficientes) em tendência, sazonalidade e ruído.

8. **Testar estacionariedade**  
   - Testes **ADF (Dickey‑Fuller)** e **KPSS** para a série semanal, com interpretação textual.

9. **Identificar padrões e características**  
   - Decisão entre modelos com/sem sazonalidade, modelos ingênuos (naïve) e modelos clássicos (Holt‑Winters, SARIMAX).

10. **Selecionar e ajustar modelos clássicos**  
    - **Holt‑Winters** (tendência aditiva, com ou sem sazonalidade).  
    - **SARIMAX(1,1,1)** com sazonalidade quando há dados suficientes.  
    - Em séries curtas, uso de modelos **sem componente sazonal**.

11. **Dividir em treino e teste**  
    - Split temporal **80% treino / 20% teste** (ou 9/3 meses para o Sales_2019).  
    - Nunca há embaralhamento; a ordem temporal é preservada.

12. **Avaliar desempenho dos modelos**  
    - Métricas **MAE**, **RMSE** e **MAPE** em conjunto de teste.  
    - Comparação Holt‑Winters × SARIMAX, escolha do modelo vencedor.

13. **Analisar resíduos** – (implementado)
    - **Gráfico de resíduos** do modelo vencedor em todas as análises de série temporal:
      - Semanal (Updated_sales).
      - Mensal (Updated_sales, Sales_2019, Retail & Warehouse).
    - Verificação visual de:
      - ausência de tendência nos resíduos,
      - variância aproximadamente constante,
      - presença de ruído branco.

14. **Detectar anomalias**  
    - Tratamento de outliers já contemplado no passo 4 (z‑score/IQR).  
    - Para o escopo do trabalho A3, técnicas mais avançadas (Isolation Forest, DBSCAN) são consideradas como extensões futuras.

15. **Testes estatísticos finais (resíduos)** – (implementado)
    - Para o modelo vencedor de cada análise:
      - **Ljung‑Box**: autocorrelação dos resíduos.  
      - **Jarque‑Bera**: normalidade dos resíduos.  
      - **Breusch‑Pagan**: heterocedasticidade.  
    - Os p‑valores são apresentados em texto nos relatórios HTML, acompanhados do gráfico de resíduos.

---

## 4. Scripts e Relatórios Gerados

### 4.1 `gerar_relatorios_series.py` – Análises Mensais

Responsável por gerar relatórios mensais em HTML dentro da pasta `relatorios/`.

- **Para `Updated_sales.csv`:**
  - Agregação mensal da receita.
  - Modelo **ingênuo**: previsão mensal repete o último valor de receita observado (devido à série curta e irregular).  
  - Métricas MAE/RMSE/MAPE no teste.  
  - Gráficos:
    - Série mensal histórica.
    - Treino vs teste com previsão.
    - Previsão dos próximos 3 meses.  
  - **Análise de resíduos:** gráfico e testes Ljung‑Box, Jarque‑Bera e Breusch‑Pagan (resíduos calculados em relação ao modelo ingênuo).

- **Para `Retail and wherehouse Sale.csv`:**
  - Construção de `total_sales = RETAIL SALES + RETAIL TRANSFERS + WAREHOUSE SALES`.  
  - Série mensal de `total_sales` (MS).  
  - Modelo Holt‑Winters ou SARIMAX (sem sazonalidade, pela curta extensão).  
  - Métricas e previsões 3 meses à frente.  
  - **Resíduos:** gráfico e testes Ljung‑Box, Jarque‑Bera e Breusch‑Pagan.

Arquivos gerados principais:

- `relatorios/relatorio_vendas_mensal_receita.html`  
- `relatorios/relatorio_retail_mensal_total.html`  
- `relatorios/index.html` (índice simples mensal)  
- CSVs com previsões: `previsao_3meses_*.csv`  
- Imagens em `relatorios/figuras_*/*.png`


### 4.2 `relatorio_semanal.py` – Análise Semanal (Updated_sales)

Fluxo de análise semanal para o dataset mais granular.

- Agregação semanal (`W-MON`) da receita (`revenue`), número de pedidos (`orders`) e unidades (`units`).  
- Tratamento de faltantes e outliers em `revenue`.  
- Ajuste de **Holt‑Winters** e **SARIMAX**, com ou sem sazonalidade semanal (52), dependendo do número de semanas disponíveis.  
- Comparação de modelos com MAE/RMSE/MAPE e escolha do melhor.
- **Previsão de 12 semanas** à frente com o modelo vencedor.
- **Análise de resíduos** do modelo vencedor:
  - Gráfico de resíduos.  
  - Testes Ljung‑Box, Jarque‑Bera, Breusch‑Pagan.
- Testes ADF e KPSS com comentário textual sobre estacionariedade.

Arquivos gerados:

- `relatorios/relatorio_semanal_receita.html`  
- `relatorios/index_semanal.html`  
- Imagens em `relatorios/figuras_semanal/*.png`


### 4.3 `analise_2_mensal.py` – Análise 2 (Sales_2019 – 12 meses)

Análise específica do ano fechado de 2019, usando os arquivos `Sales_*.csv`.

- Concatenação dos dados mensais e cálculo da receita mensal.  
- Split fixo: **9 meses de treino / 3 de teste**.  
- Ajuste de:
  - Holt‑Winters (trend aditivo, sem sazonalidade).  
  - SARIMAX(1,1,1) sem componente sazonal forte (seasonal neutro).  
- Comparação de MAE/RMSE/MAPE e escolha do melhor modelo.
- Previsão dos **3 meses seguintes**.  
- **Análise de resíduos** do modelo vencedor (Holt‑Winters ou SARIMAX):
  - Gráfico de resíduos.  
  - Testes Ljung‑Box, Jarque‑Bera, Breusch‑Pagan.

Arquivo gerado:

- `relatorios/analise_2_mensal.html`  
- Imagem: `relatorios/plot_analise2.png`  
- Resíduos: `relatorios/residuos_analise2.png`


### 4.4 `index_geral.py` – Menu Geral

Gera um **dashboard estático** em HTML com links organizados para todos os relatórios:

- `relatorios/index_geral.html`
  - Análise 1 – Vendas (Updated_sales):
    - Relatório Mensal de Receita
    - Relatório Semanal de Receita
  - Análise 2 – Previsão (Sales_2019)
  - Análise 3 – Retail & Warehouse
  - Relatórios de Demografia (quando existir `customer_details.csv`)

---

## 5. Como Executar o Projeto

Pré‐requisitos:

- Python 3.10+ (testado em 3.12).
- Instalar dependências:

```bash
pip install -r requirements.txt
```

### 5.1 Gerar relatórios mensais (Updated_sales e Retail & Warehouse)

```bash
python gerar_relatorios_series.py
```

Saídas principais em `relatorios/`:

- `relatorio_vendas_mensal_receita.html`
- `relatorio_retail_mensal_total.html`
- `index.html`

### 5.2 Gerar relatório semanal (Updated_sales)

```bash
python relatorio_semanal.py
```

Saídas:

- `relatorio_semanal_receita.html`
- `index_semanal.html`

### 5.3 Gerar análise mensal de Sales_2019 (12 meses)

```bash
python analise_2_mensal.py
```

Saída:

- `relatorios/analise_2_mensal.html`

### 5.4 Gerar índice geral (menu)

```bash
python index_geral.py
```

Saída:

- `relatorios/index_geral.html`

---

## 6. Como Fazer o Primeiro Commit neste Repositório

Depois de copiar todos os arquivos deste projeto para a pasta do repositório local `Analise-Preditiva` (clonado do GitHub), execute os comandos abaixo no terminal dentro dessa pasta:

```bash
# 1) Adicionar todos os arquivos
git add .

# 2) Criar o primeiro commit
git commit -m "Trabalho A3 - Analise Preditiva de Series Temporais"

# 3) Definir o repositório remoto (se ainda não estiver configurado)
git remote add origin https://github.com/codewithsouza/Analise-Preditiva.git

# 4) Enviar para o GitHub (primeiro push)
git branch -M main
git push -u origin main
```

Após o push, o repositório GitHub `codewithsouza/Analise-Preditiva` passará a conter todos os scripts, datasets (se desejado) e relatórios do trabalho A3. Você poderá navegar pelo código, gerar novos relatórios e anexar o link do repositório na entrega da disciplina.


