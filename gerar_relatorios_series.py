import os
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser
from datetime import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera




BASE_DIR = Path(__file__).parent.resolve()
PASTA_SAIDA = BASE_DIR / "relatorios"
PASTA_SAIDA.mkdir(exist_ok=True, parents=True)



def _ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def carregar_vendas_mensal(arquivo: Path) -> pd.DataFrame:
    """
    Carrega Updated_sales.csv e produz série mensal (MS) de receita.
    """
    df = pd.read_csv(arquivo)
    df.columns = [c.strip() for c in df.columns]

    # Conversões robustas
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", infer_datetime_format=True)
    df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"], errors="coerce")
    df["Price Each"] = pd.to_numeric(df["Price Each"], errors="coerce")

    df = df.dropna(subset=["Order Date", "Quantity Ordered", "Price Each"]).copy()
    df["Revenue"] = df["Quantity Ordered"] * df["Price Each"]

    # Série mensal
    df = df.sort_values("Order Date")
    df = df.set_index("Order Date")
    mensal = (
        df["Revenue"]
        .resample("MS")
        .sum()
        .to_frame(name="valor")
        .asfreq("MS")
    )
    # Meses com receita zero geralmente indicam ausência de dados nesse dataset;
    # tratamos como missing para permitir interpolação posterior no pipeline.
    mensal["valor"] = mensal["valor"].replace(0, np.nan)
    return mensal


def carregar_retail_mensal(arquivo: Path) -> pd.DataFrame:
    """
    Carrega 'Retail and wherehouse Sale.csv' e produz série mensal (MS) de total_sales.

    total_sales = RETAIL SALES + RETAIL TRANSFERS + WAREHOUSE SALES

    YEAR, MONTH -> primeira data do mês.
    """
    df = pd.read_csv(arquivo)
    df.columns = [c.strip() for c in df.columns]

    col_year = "YEAR"
    col_month = "MONTH"
    col_rs = "RETAIL SALES"
    col_rt = "RETAIL TRANSFERS"
    col_ws = "WAREHOUSE SALES"

    # Conversões numéricas
    df[col_year] = pd.to_numeric(df[col_year], errors="coerce")
    df[col_month] = pd.to_numeric(df[col_month], errors="coerce")
    for c in (col_rs, col_rt, col_ws):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df = df.dropna(subset=[col_year, col_month]).copy()

    # total_sales com fillna(0) já aplicado
    df["total_sales"] = df[col_rs] + df[col_rt] + df[col_ws]

    # Construir data (primeiro dia do mês)
    df["date"] = pd.to_datetime(
        df[col_year].astype(int).astype(str)
        + "-"
        + df[col_month].astype(int).astype(str).str.zfill(2)
        + "-01"
    )

    mensal = (
        df.groupby("date", as_index=True)["total_sales"]
          .sum()
          .resample("MS")
          .sum()
          .to_frame(name="valor")
          .asfreq("MS")
    )
    return mensal


def carregar_baskets_mensal(arquivo: Path) -> pd.DataFrame:
    """
    LEGADO (não usado em main): carrega basket_details.csv e produz série mensal (MS) de basket_count (soma).
    Mantido apenas para compatibilidade com análises antigas.
    """
    df = pd.read_csv(arquivo, parse_dates=["basket_date"])
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["basket_date", "basket_count"]).copy()
    df["basket_count"] = pd.to_numeric(df["basket_count"], errors="coerce")
    df = df.dropna(subset=["basket_count"]).copy()

    df = df.sort_values("basket_date").set_index("basket_date")
    mensal = (
        df["basket_count"]
        .resample("MS")
        .sum()
        .to_frame(name="valor")
        .asfreq("MS")
    )
    return mensal


def tratar_missing_interpolar(df: pd.DataFrame, coluna: str) -> Tuple[pd.DataFrame, int, int]:
    """
    Interpola valores faltantes de forma simples (linear).
    Retorna df tratado e contagem de missing antes/depois.
    """
    missing_ini = int(df[coluna].isna().sum())
    df[coluna] = df[coluna].interpolate(method="linear", limit_direction="both")
    missing_fim = int(df[coluna].isna().sum())
    return df, missing_ini, missing_fim


def detectar_outliers_zscore(df: pd.DataFrame, coluna: str, limiar: float = 3.0) -> pd.DataFrame:
    serie = df[coluna]
    media = float(serie.mean())
    std = float(serie.std())
    if std == 0 or np.isnan(std):
        return df.iloc[0:0]
    z = (serie - media) / std
    return df[np.abs(z) > limiar]


def testar_adf(serie: pd.Series) -> Tuple[float, float]:
    serie = serie.dropna()
    if len(serie) < 10:
        return (np.nan, np.nan)
    stat, pval, *_ = adfuller(serie)
    return stat, pval


def ajustar_sarimax(
    serie: pd.Series,
    ordem: Tuple[int, int, int] = (1, 1, 1),
    ordem_sazonal: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> SARIMAX:
    modelo = SARIMAX(
        serie,
        order=ordem,
        seasonal_order=ordem_sazonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return modelo.fit(disp=False)


def avaliar(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    denom = y_true.replace(0, np.nan).abs()
    mape = float((np.abs((y_true - y_pred) / denom)).dropna().mean() * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def fmt_num(x: float, casas: int = 2) -> str:
    try:
        return f"{x:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(x)


def fmt_moeda(x: float) -> str:
    try:
        return f"R$ {fmt_num(x, 2)}"
    except Exception:
        return str(x)


def format_describe(serie: pd.Series, usar_moeda: bool) -> str:
    """
    Formata o resultado de describe() sem notação científica,
    usando formato monetário quando apropriado.
    """
    desc = serie.describe()
    linhas = []
    for idx, val in desc.items():
        if pd.isna(val):
            continue
        label = str(idx)
        if label == "count":
            val_str = fmt_num(float(val), 0)
        elif usar_moeda:
            val_str = fmt_moeda(float(val))
        else:
            val_str = fmt_num(float(val), 2)
        linhas.append(f"{label:>8}: {val_str}")
    return "\n".join(linhas)


def diagnosticos_residuos_mensal(
    resid: pd.Series,
    fitted: Optional[pd.Series],
    nome_slug: str,
) -> Tuple[Dict[str, float], str]:
    """
    Calcula testes de resíduos (Ljung-Box, Jarque-Bera, Breusch-Pagan)
    e gera gráfico de resíduos para séries mensais.
    """
    resid = resid.dropna()
    if resid.empty:
        return {"lb_p": float("nan"), "jb_p": float("nan"), "bp_p": float("nan")}, ""

    # Ljung-Box
    lb_p = float("nan")
    try:
        lag = max(1, min(10, len(resid) - 1))
        lb_res = acorr_ljungbox(resid, lags=[lag], return_df=True)
        lb_p = float(lb_res["lb_pvalue"].iloc[0])
    except Exception:
        pass

    # Jarque-Bera
    jb_p = float("nan")
    try:
        _, jb_p, _, _ = jarque_bera(resid)
    except Exception:
        pass

    # Breusch-Pagan
    bp_p = float("nan")
    try:
        if fitted is not None and len(fitted) == len(resid):
            exog = sm.add_constant(fitted.loc[resid.index])
        else:
            exog = sm.add_constant(np.arange(len(resid)))
        _, bp_p, _, _ = het_breuschpagan(resid, exog)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 4))
    resid.plot(ax=ax, label="Resíduos")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Resíduos do Modelo Vencedor")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Resíduo")
    ax.legend()
    fig.tight_layout()
    caminho = PASTA_SAIDA / f"residuos_{nome_slug}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=144)
    plt.close(fig)

    tests = {"lb_p": lb_p, "jb_p": jb_p, "bp_p": bp_p}
    return tests, os.path.relpath(caminho, PASTA_SAIDA)


def salvar_grafico_serie(df: pd.DataFrame, coluna: str, caminho: Path, titulo: str) -> Path:
    plt.figure(figsize=(12, 4))
    df[coluna].plot()
    plt.title(titulo)
    plt.xlabel("Data")
    plt.ylabel(coluna)
    plt.tight_layout()
    plt.savefig(caminho, bbox_inches="tight", dpi=144)
    plt.close()
    return caminho


def salvar_grafico_treino_teste(
    treino: pd.Series,
    teste: pd.Series,
    previsao_teste: pd.Series,
    caminho: Path,
    titulo: str,
) -> Path:
    plt.figure(figsize=(12, 4))
    treino.plot(label="Treino")
    teste.plot(label="Teste", linestyle="--")
    previsao_teste.plot(label="Previsão no teste")
    plt.legend()
    plt.title(titulo)
    plt.xlabel("Data")
    plt.ylabel("valor")
    plt.tight_layout()
    plt.savefig(caminho, bbox_inches="tight", dpi=144)
    plt.close()
    return caminho


def salvar_grafico_previsao_futuro(
    serie: pd.Series,
    previsao_futuro: pd.Series,
    caminho: Path,
    titulo: str,
) -> Path:
    plt.figure(figsize=(12, 4))
    serie.plot(label="Histórico")
    previsao_futuro.plot(label="Previsão 3 meses", linestyle="--")
    plt.legend()
    plt.title(titulo)
    plt.xlabel("Data")
    plt.ylabel("valor")
    plt.tight_layout()
    plt.savefig(caminho, bbox_inches="tight", dpi=144)
    plt.close()
    return caminho


def gerar_html_relatorio_serie(
    nome: str,  # slug usado nos nomes de arquivos
    resumo: str,
    caminhos_imagens: Dict[str, str],
    metricas: Dict[str, float],
    descricao_modelo: str,
    arquivo_html: Path,
    titulo_legivel: str,
    dataset_en: str,
    variavel_rotulo: str,
    frequencia_rotulo: str,
    horizonte_meses: int,
    tabela_previsao_html: str,
    format_moeda: bool,
    res_tests: Optional[Dict[str, float]] = None,
) -> None:
    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>{titulo_legivel}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        img {{ max-width: 960px; display:block; margin-bottom: 20px; }}
        pre {{ background:#f5f5f5; padding:10px; border-radius:4px; white-space: pre-wrap; }}
        table, th, td {{ border:1px solid #aaa; border-collapse: collapse; padding:6px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        .muted {{ color:#666; font-size: 0.95em; }}
    </style>
</head>
<body>
    <h1>{titulo_legivel}</h1>

    <p class="muted">
        Dataset: <b>{dataset_en}</b> • Variável: <b>{variavel_rotulo}</b> • Frequência: <b>{frequencia_rotulo}</b> • Horizonte: <b>{horizonte_meses} meses</b>
    </p>

    <h2>Resumo dos Dados</h2>
    <pre>{resumo}</pre>

    <h2>Métricas no Conjunto de Teste</h2>
    <table>
        <tr><th>Métrica</th><th>Valor</th></tr>
        <tr><td>MAE</td><td>{(fmt_moeda(metricas.get("MAE", float("nan"))) if format_moeda else fmt_num(metricas.get("MAE", float("nan")), 2))}</td></tr>
        <tr><td>RMSE</td><td>{(fmt_moeda(metricas.get("RMSE", float("nan"))) if format_moeda else fmt_num(metricas.get("RMSE", float("nan")), 2))}</td></tr>
        <tr><td>MAPE (%)</td><td>{fmt_num(metricas.get("MAPE", float("nan")), 2)}%</td></tr>
    </table>

    <h2>Descrição do Modelo</h2>
    <p>{descricao_modelo}</p>

    {"<h2>Análise de Resíduos</h2>" if res_tests is not None else ""}
    {("<pre>Ljung-Box p=" + fmt_num(res_tests.get("lb_p", float("nan")), 4) +
      "; Jarque-Bera p=" + fmt_num(res_tests.get("jb_p", float("nan")), 4) +
      "; Breusch-Pagan p=" + fmt_num(res_tests.get("bp_p", float("nan")), 4) + "</pre>")
     if res_tests is not None else ""}
    {("<img src='" + caminhos_imagens.get("residuos","") + "' alt='Resíduos do modelo'>") if res_tests is not None and caminhos_imagens.get("residuos") else ""}

    <div class="grid">
      <div>
        <h2>Série Original</h2>
        <img src="{caminhos_imagens.get("serie", "")}" alt="Série temporal">
      </div>
      <div>
        <h2>Previsão para os Próximos {horizonte_meses} Meses — dataset: {dataset_en}</h2>
        <img src="{caminhos_imagens.get("previsao", "")}" alt="Previsão {horizonte_meses} meses">
        <h3>Valores Previstos</h3>
        {tabela_previsao_html}
      </div>
    </div>

    <h2>Treino vs Teste (Previsão no Teste)</h2>
    <img src="{caminhos_imagens.get("treino_teste", "")}" alt="Treino vs Teste">
</body>
</html>
"""
    arquivo_html.write_text(html, encoding="utf-8")


def gerar_html_demografia(
    resumo: str,
    caminhos_imagens: Dict[str, str],
    arquivo_html: Path,
) -> None:
    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Relatório - Demografia de Clientes</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        img {{ max-width: 960px; display:block; margin-bottom: 20px; }}
        pre {{ background:#f5f5f5; padding:10px; border-radius:4px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>Relatório - Demografia de Clientes</h1>
    <h2>Resumo</h2>
    <pre>{resumo}</pre>
    <h2>Distribuições</h2>
    <img src="{caminhos_imagens.get("idade", "")}" alt="Distribuição de Idade">
    <img src="{caminhos_imagens.get("tenure", "")}" alt="Distribuição de Tenure">
</body>
</html>
"""
    arquivo_html.write_text(html, encoding="utf-8")


# =========================
# PIPELINES
# =========================

def processar_serie(
    nome: str,
    df: pd.DataFrame,
    coluna_valor: str = "valor",
    usar_sazonalidade_anual: bool = True,
    dataset_en: str = "",
    variavel_rotulo: str = "valor",
    usar_moeda: bool = False,
) -> Tuple[Path, Dict[str, float]]:
    """
    Executa: limpeza leve, missing, outliers, split 80/20, treino SARIMAX, avaliação,
    re-treino full e previsão 3 meses, gera gráficos e HTML.
    Retorna caminho do HTML e métricas.
    """
    # Garantir frequência mensal
    df = df.asfreq("MS")
    df, missing_ini, missing_fim = tratar_missing_interpolar(df, coluna_valor)

    # Outliers
    outliers = detectar_outliers_zscore(df, coluna_valor)

    # Tamanho da série (apenas valores não nulos)
    n = int(df[coluna_valor].dropna().shape[0])

    # Resumo descritivo (formatado)
    resumo = format_describe(df[coluna_valor], usar_moeda=usar_moeda)
    resumo += f"\n\nn pontos mensais: {n}"
    if len(df.index) > 0:
        inicio = df.index.min()
        fim = df.index.max()
        resumo += f"\nPeríodo coberto: {inicio:%Y-%m} a {fim:%Y-%m}"
    stat, pval = testar_adf(df[coluna_valor])
    if not np.isnan(pval):
        resumo += f"\n\nADF stat: {stat:.4f}, p-value: {pval:.4f}"
    if missing_ini or missing_fim:
        resumo += f"\n\nMissing antes: {missing_ini} | depois: {missing_fim}"
    if len(outliers) > 0:
        resumo += f"\nOutliers (z>3): {len(outliers)} (mantidos; avaliar contexto)."

    # Split temporal 80/20
    serie = df[coluna_valor].astype(float)
    n = len(serie.dropna())
    n_treino = max(int(n * 0.8), 1)
    treino = serie.iloc[:n_treino]
    teste = serie.iloc[n_treino:]
    res_tests: Dict[str, float] = {}
    caminho_residuos = ""

    # Caso especial: Vendas mensais (Updated_sales) — usamos modelo ingênuo explícito
    if nome == "vendas_mensal_receita":
        # Previsão = repetir o último valor não nulo observado
        serie_na = serie.dropna()
        last_val = float(serie_na.iloc[-1]) if len(serie_na) > 0 else 0.0

        if len(teste) > 0:
            previsao_teste = pd.Series([last_val] * len(teste), index=teste.index)
            metricas = avaliar(teste, previsao_teste)
        else:
            previsao_teste = pd.Series(dtype=float)
            metricas = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

        # Próximos 3 meses: repetir último valor
        futuro_idx = pd.period_range(serie.index.max(), periods=3, freq="M").to_timestamp(how="start")
        previsao_futuro = pd.Series([last_val] * len(futuro_idx), index=futuro_idx)

        descricao_modelo = (
            "Modelo ingênuo: a previsão mensal repete o último valor de receita observado. "
            "Esta previsão é meramente ilustrativa, pois a série mensal de `Updated_sales.csv` "
            "é curta e irregular; para decisões de negócio, use preferencialmente a análise semanal."
        )

        # Resíduos: treino - último valor
        if len(treino.dropna()) > 0:
            fitted_train = pd.Series([last_val] * len(treino), index=treino.index)
            resid_train = treino - fitted_train
            res_tests, caminho_residuos = diagnosticos_residuos_mensal(resid_train, fitted_train, nome)
    else:
        # Caso geral
        # Caso extremo: treino com menos de 2 pontos -> usar média simples
        if len(treino.dropna()) < 2:
            if len(teste) > 0:
                previsao_teste = pd.Series([float(treino.mean())] * len(teste), index=teste.index)
                metricas = avaliar(teste, previsao_teste)
            else:
                previsao_teste = pd.Series(dtype=float)
                metricas = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
            descricao_modelo = "Naïve (média histórica) — dados insuficientes para modelagem"
            futuro_idx = pd.period_range(serie.index.max(), periods=3, freq="M").to_timestamp(how="start")
            previsao_futuro = pd.Series([float(serie.mean())] * 3, index=futuro_idx)
        else:
            # Modelo com fallback (poucos pontos -> Holt-Winters sem sazonalidade)
            ordem = (1, 1, 1)
            use_sazonal = usar_sazonalidade_anual and (n >= 24)
            ordem_sazonal = (1, 1, 1, 12) if use_sazonal else (0, 0, 0, 0)

            # Séries curtas (<24 meses) não usam sazonalidade anual confiável.
            # Para muito poucos pontos (<14), usamos Holt-Winters sem sazonalidade;
            # acima disso, SARIMAX com (ou sem) sazonalidade dependendo de use_sazonal.
            use_hw = n < 14  # muito poucos pontos mensais
            if use_hw:
                # Holt-Winters (tendência aditiva, sem sazonalidade) para robustez
                modelo_hw = ExponentialSmoothing(treino, trend="add", seasonal=None, initialization_method="estimated")
                fit_hw = modelo_hw.fit(optimized=True)
                if len(teste) > 0:
                    previsao_teste = fit_hw.forecast(steps=len(teste))
                    metricas = avaliar(teste, previsao_teste)
                else:
                    previsao_teste = pd.Series(dtype=float)
                    metricas = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
                descricao_modelo = "Holt-Winters (tendência aditiva, sem sazonalidade)"

                modelo_final = ExponentialSmoothing(serie, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
                previsao_futuro = modelo_final.forecast(steps=3)

                # Resíduos no treino
                fitted_train = fit_hw.fittedvalues
                resid_train = treino - fitted_train
                res_tests, caminho_residuos = diagnosticos_residuos_mensal(resid_train, fitted_train, nome)
            else:
                modelo = ajustar_sarimax(treino, ordem=ordem, ordem_sazonal=ordem_sazonal)
                if len(teste) > 0:
                    previsao_teste = modelo.predict(start=teste.index[0], end=teste.index[-1])
                    metricas = avaliar(teste, previsao_teste)
                else:
                    previsao_teste = pd.Series(dtype=float)
                    metricas = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
                descricao_modelo = f"SARIMAX{ordem} sazonal {ordem_sazonal}." + (f" ADF p-value={pval:.4f}" if not np.isnan(pval) else "")

                modelo_final = ajustar_sarimax(serie, ordem=ordem, ordem_sazonal=ordem_sazonal)
                previsao_futuro = modelo_final.forecast(steps=3)

                # Resíduos no treino
                resid_train = modelo.resid
                fitted_train = modelo.fittedvalues
                resid_train = resid_train.loc[treino.index]
                res_tests, caminho_residuos = diagnosticos_residuos_mensal(resid_train, fitted_train, nome)

            if n < 24:
                # Apenas aviso textual – mantemos o modelo já ajustado (Holt-Winters ou SARIMAX simples)
                descricao_modelo += " Atenção: série curta (menos de 24 meses); previsão de curto prazo e sem sazonalidade anual confiável."

    # Heurística de robustez: se a previsão futura for ~0 enquanto a série tem nível alto,
    # substituímos por uma previsão ingênua baseada na média dos últimos meses.
    serie_abs_mean = float(serie.dropna().abs().mean()) if len(serie.dropna()) else 0.0
    if serie_abs_mean > 0:
        very_small = np.all(np.abs(previsao_futuro.values) < 0.01 * serie_abs_mean)
        if very_small:
            tail = serie.dropna().tail(3)
            if len(tail) > 0:
                fallback_val = float(tail.mean())
                previsao_futuro = pd.Series(
                    [fallback_val] * len(previsao_futuro.index),
                    index=previsao_futuro.index,
                )
                descricao_modelo += " Observação: previsão ajustada para usar a média dos últimos meses, pois o modelo original produziu valores muito próximos de zero."

    # Garantir não-negatividade
    previsao_futuro = pd.Series(np.maximum(previsao_futuro.values, 0.0), index=previsao_futuro.index)

    # Tabela de previsão (garantir index mensal legível)
    prev_df = previsao_futuro.to_frame(name="previsao").copy()
    prev_df.index.name = "mês"
    prev_df.reset_index(inplace=True)
    prev_df["mês"] = prev_df["mês"].dt.strftime("%Y-%m")
    if usar_moeda:
        prev_df["previsao"] = prev_df["previsao"].apply(fmt_moeda)
    else:
        prev_df["previsao"] = prev_df["previsao"].apply(lambda v: fmt_num(float(v), 2))
    tabela_previsao_html = (
        "<table><tr><th>Mês</th><th>Previsão</th></tr>"
        + "".join(f"<tr><td>{r['mês']}</td><td>{r['previsao']}</td></tr>" for _, r in prev_df.iterrows())
        + "</table>"
    )

    # Pastas de saída
    pasta_figs = PASTA_SAIDA / f"figuras_{nome}"
    _ensure_dir(pasta_figs)

    # Gráficos
    caminho_serie = pasta_figs / "serie.png"
    salvar_grafico_serie(df, coluna_valor, caminho_serie, f"Série temporal - {nome}")

    caminho_treino_teste = pasta_figs / "treino_teste.png"
    if len(teste) > 0 and len(previsao_teste) > 0:
        salvar_grafico_treino_teste(treino, teste, previsao_teste, caminho_treino_teste, f"Treino x Teste - {nome}")
    else:
        salvar_grafico_treino_teste(treino, teste, pd.Series(dtype=float), caminho_treino_teste, f"Treino x Teste - {nome}")

    caminho_prev = pasta_figs / "previsao_3meses.png"
    salvar_grafico_previsao_futuro(serie, previsao_futuro, caminho_prev, f"Previsão 3 meses - {nome}")

    # CSV de previsão
    csv_prev = PASTA_SAIDA / f"previsao_3meses_{nome}.csv"
    previsao_futuro.to_frame(name="previsao").to_csv(csv_prev, encoding="utf-8")
    try:
        # Verificação simples de consistência
        _chk = pd.read_csv(csv_prev, index_col=0).squeeze("columns")
        if len(_chk) != len(previsao_futuro) or not np.allclose(_chk.astype(float).values, previsao_futuro.values, equal_nan=True):
            print(f"Aviso: divergência encontrada ao salvar {csv_prev.name}.")
    except Exception:
        pass

    # HTML
    arquivo_html = PASTA_SAIDA / f"relatorio_{nome}.html"
    caminhos_imagens = {
        "serie": os.path.relpath(caminho_serie, PASTA_SAIDA),
        "treino_teste": os.path.relpath(caminho_treino_teste, PASTA_SAIDA),
        "previsao": os.path.relpath(caminho_prev, PASTA_SAIDA),
        "residuos": caminho_residuos,
    }
    gerar_html_relatorio_serie(
        nome=nome,
        resumo=resumo,
        caminhos_imagens=caminhos_imagens,
        metricas=metricas,
        descricao_modelo=descricao_modelo,
        arquivo_html=arquivo_html,
        titulo_legivel=f"Relatório de Série Temporal - {nome.replace('_', ' ').title()}",
        dataset_en=dataset_en,
        variavel_rotulo=variavel_rotulo,
        frequencia_rotulo="Mensal",
        horizonte_meses=3,
        tabela_previsao_html=tabela_previsao_html,
        format_moeda=usar_moeda,
        res_tests=res_tests,
    )

    return arquivo_html, metricas


def processar_demografia(arquivo: Path) -> Path:
    """
    Gera um relatório HTML simples de demografia (sem previsão).
    """
    df = pd.read_csv(arquivo)
    df.columns = [c.strip() for c in df.columns]

    # Limpezas básicas
    if "customer_age" in df.columns:
        df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    # Tratar idades claramente inválidas (<0 ou >120) como NaN
    invalid_age_mask = df["customer_age"].notna() & ((df["customer_age"] < 0) | (df["customer_age"] > 120))
    n_invalid_age = int(invalid_age_mask.sum())
    df.loc[invalid_age_mask, "customer_age"] = np.nan

    # Opcional: tratar tenure negativo como NaN
    invalid_tenure_mask = df["tenure"].notna() & (df["tenure"] < 0)
    n_invalid_tenure = int(invalid_tenure_mask.sum())
    df.loc[invalid_tenure_mask, "tenure"] = np.nan

    df = df.dropna(subset=["customer_age", "tenure"]).copy()

    resumo = df[["customer_age", "tenure"]].describe().to_string()
    if n_invalid_age > 0 or n_invalid_tenure > 0:
        resumo += f"\n\nObservação: foram removidos {n_invalid_age} registros com idade fora de [0,120] e {n_invalid_tenure} com tenure negativo."
    if "sex" in df.columns:
        share = df["sex"].value_counts(normalize=True).mul(100).round(1).to_string()
        resumo += f"\n\nDistribuição por sexo (%):\n{share}"

    pasta_figs = PASTA_SAIDA / "figuras_demografia"
    _ensure_dir(pasta_figs)

    # Histogramas
    caminho_idade = pasta_figs / "idade.png"
    plt.figure(figsize=(10, 4))
    df["customer_age"].plot(kind="hist", bins=30, alpha=0.8)
    plt.title("Distribuição de Idade")
    plt.xlabel("Idade")
    plt.tight_layout()
    plt.savefig(caminho_idade, bbox_inches="tight", dpi=144)
    plt.close()

    caminho_tenure = pasta_figs / "tenure.png"
    plt.figure(figsize=(10, 4))
    df["tenure"].plot(kind="hist", bins=30, alpha=0.8)
    plt.title("Distribuição de Tempo de Relacionamento (meses)")
    plt.xlabel("Meses")
    plt.tight_layout()
    plt.savefig(caminho_tenure, bbox_inches="tight", dpi=144)
    plt.close()

    arquivo_html = PASTA_SAIDA / "relatorio_demografia.html"
    caminhos_imagens = {
        "idade": os.path.relpath(caminho_idade, PASTA_SAIDA),
        "tenure": os.path.relpath(caminho_tenure, PASTA_SAIDA),
    }
    gerar_html_demografia(resumo, caminhos_imagens, arquivo_html)
    return arquivo_html


def main() -> None:
    print("Iniciando geração de relatórios...")

    # Séries temporais
    vendas_path = BASE_DIR / "Updated_sales.csv"
    retail_path = BASE_DIR / "Retail and wherehouse Sale.csv"
    clientes_path = BASE_DIR / "customer_details.csv"

    relatorios: List[Tuple[str, Path, Dict[str, float]]] = []

    # 1) Vendas mensais (receita) – Análise 1 (Updated_sales)
    # Aqui geramos o relatório MENSAL; a análise semanal é feita em `relatorio_semanal.py`.
    if vendas_path.exists():
        print("Processando: Vendas mensais (receita)")
        vendas_m = carregar_vendas_mensal(vendas_path)
        html_vendas, mets_vendas = processar_serie(
            nome="vendas_mensal_receita",
            df=vendas_m,
            coluna_valor="valor",
            usar_sazonalidade_anual=True,
            dataset_en="Updated_sales.csv",
            variavel_rotulo="Receita (R$)",
            usar_moeda=True,
        )
        relatorios.append(("vendas_mensal_receita", html_vendas, mets_vendas))
    else:
        print("Arquivo não encontrado:", vendas_path)

    # 2) Retail & Warehouse mensais (total_sales)
    if retail_path.exists():
        print("Processando: Retail & Warehouse mensais (total_sales)")
        retail_m = carregar_retail_mensal(retail_path)
        html_retail, mets_retail = processar_serie(
            nome="retail_mensal_total",
            df=retail_m,
            coluna_valor="valor",
            usar_sazonalidade_anual=True,
            dataset_en="Retail and wherehouse Sale.csv",
            variavel_rotulo="Total Sales (retail + transfers + warehouse)",
            usar_moeda=False,
        )
        relatorios.append(("retail_mensal_total", html_retail, mets_retail))
    else:
        print("Arquivo não encontrado:", retail_path)

    # 3) Relatório de demografia (não é série temporal)
    html_demo = None
    if clientes_path.exists():
        print("Processando: Demografia de clientes")
        html_demo = processar_demografia(clientes_path)
    else:
        print("Arquivo não encontrado:", clientes_path)

    # Índice geral
    indice_path = PASTA_SAIDA / "index.html"
    html_index = [
        "<html><head><meta charset='utf-8'><title>Relatórios de Séries</title></head><body>",
        "<h1>Relatórios</h1><ul>",
    ]
    for nome, arq, mets in relatorios:
        rel_path = os.path.relpath(arq, PASTA_SAIDA)
        mape_txt = f"{fmt_num(mets.get('MAPE', float('nan')), 2)}%"
        if "vendas_mensal_receita" in nome:
            legivel = "Vendas - Receita Mensal (Updated_sales)"
        elif "retail_mensal_total" in nome:
            legivel = "Vendas - Retail & Warehouse Mensal"
        elif "baskets" in nome:
            legivel = "Cestas (legacy) - Basket Count Mensal"
        else:
            legivel = nome
        html_index.append(f"<li><a href='{rel_path}'>{legivel}</a> — MAPE teste: {mape_txt}</li>")
    html_index.append("</ul>")

    if html_demo is not None:
        rel_path = os.path.relpath(html_demo, PASTA_SAIDA)
        html_index.append(f"<p><a href='{rel_path}'>Relatório de Demografia</a></p>")

    html_index.append("</body></html>")
    indice_path.write_text("\n".join(html_index), encoding="utf-8")

    print("Concluído. Abra o arquivo:", indice_path)
    # Abrir automaticamente o índice e relatórios gerados
    try:
        webbrowser.open(indice_path.resolve().as_uri())
        for _, arq, _ in relatorios:
            webbrowser.open(Path(arq).resolve().as_uri())
        if html_demo is not None:
            webbrowser.open(Path(html_demo).resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()


