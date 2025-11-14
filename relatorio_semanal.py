import os
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


BASE_DIR = Path(__file__).parent.resolve()
PASTA_SAIDA = BASE_DIR / "relatorios"
PASTA_FIGS = PASTA_SAIDA / "figuras_semanal"
PASTA_SAIDA.mkdir(exist_ok=True, parents=True)
PASTA_FIGS.mkdir(exist_ok=True, parents=True)

plt.style.use("seaborn-v0_8")
sns.set_context("talk")


# ========= Funções auxiliares =========

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


def avaliar(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    denom = y_true.replace(0, np.nan).abs()
    mape = float((np.abs((y_true - y_pred) / denom)).dropna().mean() * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def diagnosticos_residuos(
    resid: pd.Series,
    fitted: Optional[pd.Series],
    prefixo_fig: str,
) -> Tuple[Dict[str, float], str]:
    """
    Calcula testes de resíduos (Ljung-Box, Jarque-Bera, Breusch-Pagan)
    e gera gráfico de resíduos.
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

    # Jarque-Bera (normalidade)
    jb_p = float("nan")
    try:
        _, jb_p, _, _ = jarque_bera(resid)
    except Exception:
        pass

    # Breusch-Pagan (heterocedasticidade)
    bp_p = float("nan")
    try:
        if fitted is not None and len(fitted) == len(resid):
            exog = sm.add_constant(fitted.loc[resid.index])
        else:
            exog = sm.add_constant(np.arange(len(resid)))
        _, bp_p, _, _ = het_breuschpagan(resid, exog)
    except Exception:
        pass

    # Gráfico de resíduos
    fig, ax = plt.subplots(figsize=(14, 4))
    resid.plot(ax=ax, label="Resíduos")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Resíduos do Modelo Vencedor")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Resíduo")
    ax.legend()
    fig.tight_layout()
    caminho = PASTA_FIGS / f"{prefixo_fig}.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=144)
    plt.close(fig)

    tests = {"lb_p": lb_p, "jb_p": jb_p, "bp_p": bp_p}
    return tests, os.path.relpath(caminho, PASTA_SAIDA)


# ========= Carregamento & preparação =========

def load_sales_weekly(path: Path, freq: str = "W-MON") -> pd.DataFrame:
    """
    Carrega Updated_sales.csv e agrega para série semanal:
    - revenue: soma de Revenue
    - orders: nº único de Order ID
    - units: soma de Quantity Ordered
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Quantity Ordered"] = pd.to_numeric(df["Quantity Ordered"], errors="coerce")
    df["Price Each"] = pd.to_numeric(df["Price Each"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Quantity Ordered", "Price Each"]).copy()

    df["Revenue"] = df["Quantity Ordered"] * df["Price Each"]
    df["date"] = df["Order Date"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    weekly = (
        df.set_index("date")
          .resample(freq)
          .agg(
              revenue=("Revenue", "sum"),
              orders=("Order ID", pd.Series.nunique),
              units=("Quantity Ordered", "sum"),
          )
    )
    weekly = weekly.asfreq(freq)
    return weekly


def load_basket_weekly(path: Path, freq: str = "W-MON") -> pd.Series:
    """
    Carrega basket_details.csv e agrega basket_count semanalmente.
    """
    df = pd.read_csv(path, parse_dates=["basket_date"])
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["basket_date", "basket_count"]).copy()
    df["basket_count"] = pd.to_numeric(df["basket_count"], errors="coerce")
    df = df.dropna(subset=["basket_count"]).copy()

    weekly = (
        df.set_index("basket_date")
          .resample(freq)
          .agg(basket_count=("basket_count", "sum"))
          .asfreq(freq)
    )
    weekly["basket_count"] = weekly["basket_count"].interpolate(limit_direction="both")
    return weekly["basket_count"]


def clean_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata faltantes e outliers em uma série semanal com colunas:
    revenue, orders, units.
    - Faltantes: interpolação para revenue, ffill/bfill para counts.
    - Outliers em revenue: clip via IQR.
    """
    weekly = df.copy()

    # Preenchimento de faltantes
    weekly["revenue"] = weekly["revenue"].interpolate(limit_direction="both")
    for col in ["orders", "units"]:
        weekly[col] = weekly[col].ffill().bfill()

    # Outliers por IQR em revenue
    q1, q3 = weekly["revenue"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    weekly["revenue"] = weekly["revenue"].clip(lower=lower, upper=upper)

    return weekly


# ========= Modelagem semanal =========

def fit_models_weekly(
    y_train: pd.Series,
    y_test: pd.Series,
    exog_train: Optional[pd.Series] = None,
    exog_test: Optional[pd.Series] = None,
    freq: str = "W-MON",
) -> Dict[str, Dict]:
    """
    Ajusta modelos Holt-Winters e SARIMAX para a série semanal.

    - Se len(y_train) < 10: usa apenas Holt-Winters sem sazonalidade.
    - Se len(y_train) < 52: modelos sem sazonalidade (sem seasonal_periods=52).
    - Caso contrário, usa sazonalidade semanal (52).

    Retorna dict:
    {
      nome_modelo: {
        "fit": fitted_model,
        "forecast_test": Series alinhada a y_test,
        "metrics": {...},
        "description": str,
        "use_exog": bool
      },
      ...
    }
    """
    results: Dict[str, Dict] = {}
    n = len(y_train.dropna())
    use_seasonal = n >= 52

    # Holt-Winters
    if n >= 2:
        if use_seasonal:
            hw_model = ExponentialSmoothing(
                y_train,
                trend="add",
                seasonal="mul",
                seasonal_periods=52,
                initialization_method="estimated",
            )
        else:
            hw_model = ExponentialSmoothing(
                y_train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
        hw_fit = hw_model.fit(optimized=True)
        hw_forecast = hw_fit.forecast(steps=len(y_test))
        hw_forecast.index = y_test.index
        results["HoltWinters"] = {
            "fit": hw_fit,
            "forecast_test": hw_forecast,
            "metrics": avaliar(y_test, hw_forecast),
            "description": "Holt-Winters com tendência aditiva"
            + (" e sazonalidade semanal (52)" if use_seasonal else " sem sazonalidade"),
            "use_exog": False,
        }

    # SARIMAX sem/como sazonalidade
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52) if use_seasonal else (0, 0, 0, 0)
    try:
        sarimax_model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarimax_fit = sarimax_model.fit(disp=False)
        sarimax_forecast = sarimax_fit.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean
        sarimax_forecast.index = y_test.index
        results["SARIMAX"] = {
            "fit": sarimax_fit,
            "forecast_test": sarimax_forecast,
            "metrics": avaliar(y_test, sarimax_forecast),
            "description": f"SARIMAX{order} sazonal {seasonal_order}"
            + (" com exógena" if exog_train is not None else ""),
            "use_exog": exog_train is not None,
        }
    except Exception as exc:  # noqa: BLE001
        print("Falha ao ajustar SARIMAX:", exc)

    return results


def evaluate_models(
    y_train: pd.Series,
    y_test: pd.Series,
    results: Dict[str, Dict],
) -> Tuple[str, pd.DataFrame]:
    """
    Monta DataFrame de métricas e escolhe melhor modelo (menor RMSE).
    """
    rows = []
    for name, obj in results.items():
        m = obj["metrics"]
        rows.append({"modelo": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"]})
    metrics_df = pd.DataFrame(rows).sort_values("RMSE")
    best_model = metrics_df.iloc[0]["modelo"]
    return best_model, metrics_df


def forecast_weeks(
    y: pd.Series,
    best_model: str,
    results: Dict[str, Dict],
    steps: int = 12,
    exog_full: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Reajusta o melhor modelo em toda a série y e faz previsão de `steps` semanas.
    Usa exógena (se disponível) replicando o último valor para o horizonte futuro.
    """
    freq = y.index.freq or pd.infer_freq(y.index) or "W-MON"
    use_seasonal = len(y.dropna()) >= 52

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52) if use_seasonal else (0, 0, 0, 0)

    if best_model == "HoltWinters":
        if use_seasonal:
            hw_model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="mul",
                seasonal_periods=52,
                initialization_method="estimated",
            )
        else:
            hw_model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
        fit_full = hw_model.fit(optimized=True)
        future = fit_full.forecast(steps=steps)
    else:
        exog_full_fit = exog_full
        if exog_full_fit is not None:
            last_val = float(exog_full_fit.iloc[-1])
            exog_future = pd.Series([last_val] * steps, index=pd.date_range(y.index.max() + pd.Timedelta(days=7), periods=steps, freq=freq))
        else:
            exog_future = None

        model = SARIMAX(
            y,
            exog=exog_full_fit,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit_full = model.fit(disp=False)
        future = fit_full.get_forecast(steps=steps, exog=exog_future).predicted_mean

    # Garantir índice semanal contínuo
    last_idx = y.index.max()
    future_index = pd.date_range(last_idx + pd.Timedelta(days=7), periods=steps, freq=freq)
    future.index = future_index
    return pd.Series(np.maximum(future.values, 0.0), index=future_index)


# ========= Gráficos e HTML =========

def salvar_graficos_semanal(
    weekly: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_model: str,
    results: Dict[str, Dict],
    future_forecast: pd.Series,
    caminho_residuos: str,
) -> Dict[str, str]:
    """
    Gera e salva os gráficos principais, retornando caminhos relativos.
    """
    # Série completa
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    weekly["revenue"].plot(ax=ax1, label="Receita semanal")
    ax1.set_title("Receita Semanal (histórico)")
    ax1.set_xlabel("Semana")
    ax1.set_ylabel("Receita")
    ax1.legend()
    fig1.tight_layout()
    caminho_serie = PASTA_FIGS / "serie_semanal.png"
    fig1.savefig(caminho_serie, bbox_inches="tight", dpi=144)
    plt.close(fig1)

    # Treino vs teste
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    y_train.plot(ax=ax2, label="Treino")
    y_test.plot(ax=ax2, label="Teste", linestyle="--")
    results[best_model]["forecast_test"].plot(ax=ax2, label=f"{best_model} (previsão no teste)")
    ax2.set_title("Treino vs Teste com Previsão")
    ax2.set_xlabel("Semana")
    ax2.set_ylabel("Receita")
    ax2.legend()
    fig2.tight_layout()
    caminho_treino_teste = PASTA_FIGS / "treino_teste_semanal.png"
    fig2.savefig(caminho_treino_teste, bbox_inches="tight", dpi=144)
    plt.close(fig2)

    # Previsão 12 semanas
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    weekly["revenue"].plot(ax=ax3, label="Histórico")
    future_forecast.plot(ax=ax3, label=f"{best_model} (12 semanas futuras)", linestyle="--")
    ax3.set_title("Previsão de 12 Semanas à Frente")
    ax3.set_xlabel("Semana")
    ax3.set_ylabel("Receita")
    ax3.legend()
    fig3.tight_layout()
    caminho_prev = PASTA_FIGS / "previsao_12_semanas.png"
    fig3.savefig(caminho_prev, bbox_inches="tight", dpi=144)
    plt.close(fig3)

    # Decomposição + ACF/PACF (opcionais)
    caminho_decomp = ""
    caminho_acf_pacf = ""
    if len(weekly["revenue"].dropna()) >= 24:
        try:
            decomp = seasonal_decompose(weekly["revenue"], model="additive", period=52)
            fig4 = decomp.plot()
            fig4.set_size_inches(14, 8)
            fig4.tight_layout()
            caminho_decomp = PASTA_FIGS / "decomposicao_semanal.png"
            fig4.savefig(caminho_decomp, bbox_inches="tight", dpi=144)
            plt.close(fig4)
        except Exception as exc:  # noqa: BLE001
            print("Falha na decomposição semanal:", exc)

        fig5, ax = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(weekly["revenue"].dropna(), lags=30, ax=ax[0])
        plot_pacf(weekly["revenue"].dropna(), lags=30, ax=ax[1])
        ax[0].set_title("ACF - Receita Semanal")
        ax[1].set_title("PACF - Receita Semanal")
        fig5.tight_layout()
        caminho_acf_pacf = PASTA_FIGS / "acf_pacf_semanal.png"
        fig5.savefig(caminho_acf_pacf, bbox_inches="tight", dpi=144)
        plt.close(fig5)

    return {
        "serie": os.path.relpath(caminho_serie, PASTA_SAIDA),
        "treino_teste": os.path.relpath(caminho_treino_teste, PASTA_SAIDA),
        "previsao": os.path.relpath(caminho_prev, PASTA_SAIDA),
        "decomp": os.path.relpath(caminho_decomp, PASTA_SAIDA) if caminho_decomp else "",
        "acf_pacf": os.path.relpath(caminho_acf_pacf, PASTA_SAIDA) if caminho_acf_pacf else "",
        "residuos": caminho_residuos,
    }


def gerar_html_relatorio_semanal(
    weekly: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    results: Dict[str, Dict],
    best_model: str,
    future_forecast: pd.Series,
    metrics_df: pd.DataFrame,
    caminhos_imagens: Dict[str, str],
    has_exog: bool,
    adf_res: Tuple[float, float],
    kpss_res: Tuple[float, float],
    res_tests: Dict[str, float],
    arquivo_html: Path,
) -> None:
    n_weeks = len(weekly.dropna())
    periodo_txt = ""
    if len(weekly.index) > 0:
        inicio = weekly.index.min()
        fim = weekly.index.max()
        periodo_txt = f"{inicio:%Y-%m-%d} a {fim:%Y-%m-%d}"

    adf_stat, adf_p = adf_res
    kpss_stat, kpss_p = kpss_res
    estacionaria_txt = "Indeterminado"
    if not np.isnan(adf_p) and not np.isnan(kpss_p):
        if adf_p < 0.05 and kpss_p > 0.05:
            estacionaria_txt = "Provavelmente estacionária"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            estacionaria_txt = "Provavelmente não estacionária"
        else:
            estacionaria_txt = "Mista / inconclusiva"

    resumo = weekly["revenue"].describe().to_string()
    resumo += f"\n\nSemanas: {n_weeks}\nPeríodo: {periodo_txt}\nEstacionariedade (ADF/KPSS): {estacionaria_txt}"

    metrics_html = metrics_df.assign(
        MAE=lambda d: d["MAE"].map(fmt_moeda),
        RMSE=lambda d: d["RMSE"].map(fmt_moeda),
        MAPE=lambda d: d["MAPE"].map(lambda v: f"{fmt_num(v, 2)}%"),
    ).to_html(index=False)

    future_df = future_forecast.to_frame(name="previsao").copy()
    future_df.index.name = "semana"
    future_df.reset_index(inplace=True)
    future_df["semana"] = future_df["semana"].dt.strftime("%Y-%m-%d")
    future_df["previsao"] = future_df["previsao"].map(fmt_moeda)
    future_html = (
        "<table><tr><th>Semana</th><th>Previsão de Receitas</th></tr>"
        + "".join(
            f"<tr><td>{r['semana']}</td><td>{r['previsao']}</td></tr>" for _, r in future_df.iterrows()
        )
        + "</table>"
    )

    # Resumo dos testes de resíduos
    lb_p = res_tests.get("lb_p", float("nan"))
    jb_p = res_tests.get("jb_p", float("nan"))
    bp_p = res_tests.get("bp_p", float("nan"))
    testes_residuos_txt = (
        f"Ljung-Box p={fmt_num(lb_p, 4)}; "
        f"Jarque-Bera p={fmt_num(jb_p, 4)}; "
        f"Breusch-Pagan p={fmt_num(bp_p, 4)}"
    )

    html = f"""
<html>
<head>
  <meta charset="utf-8">
  <title>Relatório de Série Temporal Semanal - Receita</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2, h3 {{ color: #333; }}
    img {{ max-width: 960px; display:block; margin-bottom: 20px; }}
    pre {{ background:#f5f5f5; padding:10px; border-radius:4px; white-space: pre-wrap; }}
    table, th, td {{ border:1px solid #aaa; border-collapse: collapse; padding:6px; }}
    .muted {{ color:#666; font-size: 0.95em; }}
  </style>
</head>
<body>
  <h1>Relatório de Série Temporal Semanal - Receita</h1>
  <p class="muted">
    Dataset: <b>Updated_sales.csv</b> • Frequência: <b>Semanal (W-MON)</b> • Horizonte: <b>12 semanas</b>
    {"• Com variável exógena basket_count" if has_exog else ""}
  </p>

  <h2>Resumo da Série Semanal</h2>
  <pre>{resumo}</pre>

  <h2>Métricas por Modelo (Conjunto de Teste)</h2>
  {metrics_html}

  <h2>Modelo Vencedor</h2>
  <p><b>{best_model}</b>: {results[best_model]['description']}</p>

  <h2>Análise de Resíduos</h2>
  <pre>{testes_residuos_txt}</pre>
  {"<img src='" + caminhos_imagens.get('residuos','') + "' alt='Resíduos do modelo'>" if caminhos_imagens.get("residuos") else ""}

  <h2>Gráficos</h2>
  <h3>Série Semanal - Histórico</h3>
  <img src="{caminhos_imagens.get('serie', '')}" alt="Série semanal">

  <h3>Treino vs Teste com Previsão ({best_model})</h3>
  <img src="{caminhos_imagens.get('treino_teste', '')}" alt="Treino vs Teste">

  <h3>Previsão para as Próximas 12 Semanas ({best_model})</h3>
  <img src="{caminhos_imagens.get('previsao', '')}" alt="Previsão 12 semanas">
  <h3>Valores Previstos</h3>
  {future_html}

  {"<h3>Decomposição da Série</h3><img src='" + caminhos_imagens.get('decomp','') + "' alt='Decomposição'>" if caminhos_imagens.get("decomp") else ""}
  {"<h3>ACF / PACF</h3><img src='" + caminhos_imagens.get('acf_pacf','') + "' alt='ACF PACF'>" if caminhos_imagens.get("acf_pacf") else ""}

</body>
</html>
"""
    arquivo_html.write_text(html, encoding="utf-8")


def relatorio_demografia_semanal(path_customers: Path) -> Path:
    """
    Gera relatório de demografia similar ao mensal, mas em arquivo separado.
    """
    df = pd.read_csv(path_customers)
    df.columns = [c.strip() for c in df.columns]

    if "customer_age" in df.columns:
        df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    invalid_age_mask = df["customer_age"].notna() & ((df["customer_age"] < 0) | (df["customer_age"] > 120))
    n_invalid_age = int(invalid_age_mask.sum())
    df.loc[invalid_age_mask, "customer_age"] = np.nan

    invalid_tenure_mask = df["tenure"].notna() & (df["tenure"] < 0)
    n_invalid_tenure = int(invalid_tenure_mask.sum())
    df.loc[invalid_tenure_mask, "tenure"] = np.nan

    df = df.dropna(subset=["customer_age", "tenure"]).copy()

    resumo = df[["customer_age", "tenure"]].describe().to_string()
    resumo += f"\n\nObservação: removidos {n_invalid_age} registros com idade fora de [0,120] e {n_invalid_tenure} com tenure negativo."
    if "sex" in df.columns:
        share = df["sex"].value_counts(normalize=True).mul(100).round(1).to_string()
        resumo += f"\n\nDistribuição por sexo (%):\n{share}"

    figs_dir = PASTA_FIGS / "demografia_semanal"
    figs_dir.mkdir(exist_ok=True, parents=True)

    caminho_idade = figs_dir / "idade.png"
    plt.figure(figsize=(10, 4))
    df["customer_age"].plot(kind="hist", bins=30, alpha=0.8)
    plt.title("Distribuição de Idade")
    plt.xlabel("Idade")
    plt.tight_layout()
    plt.savefig(caminho_idade, bbox_inches="tight", dpi=144)
    plt.close()

    caminho_tenure = figs_dir / "tenure.png"
    plt.figure(figsize=(10, 4))
    df["tenure"].plot(kind="hist", bins=30, alpha=0.8)
    plt.title("Distribuição de Tempo de Relacionamento (meses)")
    plt.xlabel("Meses")
    plt.tight_layout()
    plt.savefig(caminho_tenure, bbox_inches="tight", dpi=144)
    plt.close()

    html_path = PASTA_SAIDA / "relatorio_demografia_semanal.html"
    html = f"""
<html>
<head>
  <meta charset="utf-8">
  <title>Relatório - Demografia de Clientes (Semanal)</title>
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
  <img src="{os.path.relpath(caminho_idade, PASTA_SAIDA)}" alt="Distribuição de Idade">
  <img src="{os.path.relpath(caminho_tenure, PASTA_SAIDA)}" alt="Distribuição de Tenure">
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ========= main =========

def main() -> None:
    print("Iniciando relatório semanal...")

    sales_path = BASE_DIR / "Updated_sales.csv"
    basket_path = BASE_DIR / "basket_details.csv"
    customers_path = BASE_DIR / "customer_details.csv"

    relatorios: List[Tuple[str, Path, Optional[Dict[str, float]]]] = []

    # 1) Série semanal de receita
    if not sales_path.exists():
        print("Arquivo de vendas não encontrado:", sales_path)
        return

    weekly_raw = load_sales_weekly(sales_path)
    weekly = clean_weekly(weekly_raw)
    y = weekly["revenue"].asfreq("W-MON")

    # Split 80/20
    n = len(y.dropna())
    train_size = max(int(n * 0.8), 1)
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Exógena opcional
    exog_full = None
    exog_train = exog_test = None
    has_exog = False
    if basket_path.exists():
        basket_weekly = load_basket_weekly(basket_path)
        basket_weekly = basket_weekly.reindex(y.index).interpolate(limit_direction="both")
        exog_full = basket_weekly
        exog_train = exog_full.iloc[:train_size]
        exog_test = exog_full.iloc[train_size:]
        has_exog = True

    results = fit_models_weekly(y_train, y_test, exog_train=exog_train if has_exog else None, exog_test=exog_test if has_exog else None)
    best_model, metrics_df = evaluate_models(y_train, y_test, results)

    future_forecast = forecast_weeks(y, best_model, results, steps=12, exog_full=exog_full if has_exog else None)

    # Resíduos do modelo vencedor (treino)
    fit_best = results[best_model]["fit"]
    if best_model == "HoltWinters":
        fitted_train = fit_best.fittedvalues
        resid_train = y_train - fitted_train
    else:  # SARIMAX
        resid_train = fit_best.resid
        fitted_train = fit_best.fittedvalues
        resid_train = resid_train.loc[y_train.index]
    testes_res, caminho_residuos = diagnosticos_residuos(resid_train, fitted_train, "residuos_semanal")

    # Testes ADF / KPSS para comentário
    adf_stat, adf_p = (np.nan, np.nan)
    kpss_stat, kpss_p = (np.nan, np.nan)
    try:
        adf_stat, adf_p, *_ = adfuller(y.dropna(), autolag="AIC")
    except Exception:
        pass
    try:
        kpss_stat, kpss_p, *_ = kpss(y.dropna(), regression="c", nlags="auto")
    except Exception:
        pass

    caminhos_imagens = salvar_graficos_semanal(weekly, y_train, y_test, best_model, results, future_forecast, caminho_residuos)
    rel_html = PASTA_SAIDA / "relatorio_semanal_receita.html"
    gerar_html_relatorio_semanal(
        weekly=weekly,
        y_train=y_train,
        y_test=y_test,
        results=results,
        best_model=best_model,
        future_forecast=future_forecast,
        metrics_df=metrics_df,
        caminhos_imagens=caminhos_imagens,
        has_exog=has_exog,
        adf_res=(adf_stat, adf_p),
        kpss_res=(kpss_stat, kpss_p),
        res_tests=testes_res,
        arquivo_html=rel_html,
    )
    relatorios.append(("Relatório Semanal de Receita", rel_html, {"MAPE": float(metrics_df.iloc[0]["MAPE"])}))

    # 2) Demografia
    demo_html = None
    if customers_path.exists():
        demo_html = relatorio_demografia_semanal(customers_path)
        relatorios.append(("Demografia de Clientes", demo_html, None))

    # 3) Index semanal
    index_path = PASTA_SAIDA / "index_semanal.html"
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Relatórios Semanais</title></head><body>",
        "<h1>Relatórios Semanais</h1><ul>",
    ]
    for nome, arq, mets in relatorios:
        rel_path = os.path.relpath(arq, PASTA_SAIDA)
        if mets and "MAPE" in mets:
            mape_txt = f"{fmt_num(mets['MAPE'], 2)}%"
            html_parts.append(f"<li><a href='{rel_path}'>{nome}</a> — MAPE teste: {mape_txt}</li>")
        else:
            html_parts.append(f"<li><a href='{rel_path}'>{nome}</a></li>")
    html_parts.append("</ul></body></html>")
    index_path.write_text("\n".join(html_parts), encoding="utf-8")

    print("Relatórios semanais concluídos. Abra o arquivo:", index_path)
    try:
        webbrowser.open(index_path.resolve().as_uri())
        for _, arq, _ in relatorios:
            webbrowser.open(Path(arq).resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()


