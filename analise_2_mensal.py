"""
ANÁLISE 2 – PREVISÃO MENSAL (12 MESES)

Modelos: Holt-Winters e SARIMAX
Treino: 9 meses | Teste: 3 meses
Saída: relatorios/analise_2_mensal.html
"""

import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


def mape(a, b):
    a, b = np.array(a), np.array(b)
    return np.mean(np.abs((a - b) / np.clip(np.abs(a), 1e-9, None))) * 100


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


def diagnosticos_residuos(y_train: pd.Series, fit, best_model: str, pasta_saida: Path) -> (dict, str):
    """
    Calcula Ljung-Box, Jarque-Bera, Breusch-Pagan e salva gráfico de resíduos.
    """
    if best_model == "HoltWinters":
        fitted = fit.fittedvalues
        resid = y_train - fitted
    else:  # SARIMAX
        resid = fit.resid
        fitted = fit.fittedvalues
        resid = resid.loc[y_train.index]
        fitted = fitted.loc[y_train.index]

    resid = resid.dropna()
    tests = {"lb_p": float("nan"), "jb_p": float("nan"), "bp_p": float("nan")}
    if resid.empty:
        return tests, ""

    # Ljung-Box
    try:
        lag = max(1, min(10, len(resid) - 1))
        lb_res = acorr_ljungbox(resid, lags=[lag], return_df=True)
        tests["lb_p"] = float(lb_res["lb_pvalue"].iloc[0])
    except Exception:
        pass

    # Jarque-Bera
    try:
        _, jb_p, _, _ = jarque_bera(resid)
        tests["jb_p"] = float(jb_p)
    except Exception:
        pass

    # Breusch-Pagan
    try:
        exog = sm.add_constant(fitted)
        _, bp_p, _, _ = het_breuschpagan(resid, exog)
        tests["bp_p"] = float(bp_p)
    except Exception:
        pass

    # Gráfico de resíduos
    fig, ax = plt.subplots(figsize=(12, 4))
    resid.plot(ax=ax, label="Resíduos")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Resíduos do Modelo Vencedor")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Resíduo")
    ax.legend()
    fig.tight_layout()
    caminho = pasta_saida / "residuos_analise2.png"
    fig.savefig(caminho, bbox_inches="tight", dpi=144)
    plt.close(fig)

    return tests, caminho.name


def main() -> None:
    base = Path.cwd()
    pasta_saida = base / "relatorios"
    pasta_saida.mkdir(exist_ok=True, parents=True)

    # ----------------------------
    # 1) Carregar todos os CSVs mensais
    # ----------------------------

    # Procurar arquivos Sales_*.csv na pasta Sales_Data (preferencialmente) ou na raiz
    sales_dir = base / "Sales_Data"
    search_dir = sales_dir if sales_dir.exists() else base
    files = sorted([f for f in os.listdir(search_dir) if f.startswith("Sales_") and f.endswith(".csv")])
    if not files:
        print("Nenhum arquivo Sales_*.csv encontrado.")
        return

    df_list = []
    for f in files:
        d = pd.read_csv(search_dir / f)
        d.columns = [c.strip() for c in d.columns]
        d["Order Date"] = pd.to_datetime(d["Order Date"], errors="coerce")
        d["Quantity Ordered"] = pd.to_numeric(d["Quantity Ordered"], errors="coerce")
        d["Price Each"] = pd.to_numeric(d["Price Each"], errors="coerce")
        d = d.dropna(subset=["Order Date", "Quantity Ordered", "Price Each"])
        d["Revenue"] = d["Quantity Ordered"] * d["Price Each"]
        df_list.append(d)

    sales = pd.concat(df_list, ignore_index=True)
    sales = sales.sort_values("Order Date")

    # ----------------------------
    # 2) Série mensal de receita
    # ----------------------------

    sales["month"] = sales["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = sales.groupby("month")["Revenue"].sum().asfreq("MS")

    # ----------------------------
    # 3) Split 9 meses → 3 meses
    # ----------------------------

    y = monthly.copy()
    n = len(y)
    if n < 12:
        print(f"Atenção: menos de 12 meses disponíveis ({n}). Ajuste o split se necessário.")

    train_size = 9
    y_train = y[:train_size]
    y_test = y[train_size:]

    h = 3  # horizonte fixo
    y_true = y_test[:h]

    # ----------------------------
    # 4) Modelos
    # ----------------------------

    results = {}

    # Holt-Winters (sem sazonalidade – série curta)
    hw = ExponentialSmoothing(
        y_train,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    hw_fit = hw.fit()
    hw_pred = hw_fit.forecast(h)
    results["HoltWinters"] = hw_pred

    # SARIMAX (sem sazonalidade forte – seasonal_order neutro)
    sarimax = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarimax_fit = sarimax.fit(disp=False)
    sarimax_pred = sarimax_fit.get_forecast(h).predicted_mean
    results["SARIMAX"] = sarimax_pred

    # ----------------------------
    # 5) Métricas
    # ----------------------------

    metrics = []
    for name, pred in results.items():
        pred.index = y_true.index
        mae = mean_absolute_error(y_true, pred)
        rmse = mean_squared_error(y_true, pred) ** 0.5
        mape_val = mape(y_true, pred)
        metrics.append([name, mae, rmse, mape_val])

    metrics_df = pd.DataFrame(metrics, columns=["Modelo", "MAE", "RMSE", "MAPE%"]).sort_values("RMSE")
    best_model = metrics_df.iloc[0]["Modelo"]

    # ----------------------------
    # 6) Previsão final – próximos 3 meses
    # ----------------------------

    if best_model == "SARIMAX":
        final_forecast = sarimax_fit.get_forecast(3).predicted_mean
        fit_best = sarimax_fit
    else:
        final_forecast = hw_fit.forecast(3)
        fit_best = hw_fit

    # ----------------------------
    # 7) Gerar relatório HTML
    # ----------------------------

    html_path = pasta_saida / "analise_2_mensal.html"

    fig, ax = plt.subplots(figsize=(12, 5))
    y.plot(ax=ax, label="Histórico")
    y_true.plot(ax=ax, label="Teste", color="orange")
    results[best_model].plot(ax=ax, label=f"Previsão (teste) – {best_model}", color="green")
    final_forecast.plot(ax=ax, label="Previsão futura (3 meses)", color="red")
    ax.set_title("Análise 2 – Previsão Mensal (12 meses de 2019)")
    ax.legend()
    plt.tight_layout()

    plot_path = pasta_saida / "plot_analise2.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=144)
    plt.close()

    # Diagnósticos de resíduos
    res_tests, caminho_residuos = diagnosticos_residuos(y_train, fit_best, best_model, pasta_saida)

    # Formatar métricas para exibição
    metrics_display = metrics_df.copy()
    metrics_display["MAE"] = metrics_display["MAE"].map(fmt_moeda)
    metrics_display["RMSE"] = metrics_display["RMSE"].map(fmt_moeda)
    metrics_display["MAPE%"] = metrics_display["MAPE%"].map(lambda v: f"{fmt_num(v, 2)}%")
    metrics_html = metrics_display.to_html(index=False, escape=False)

    # Formatar previsão futura
    forecast_df = final_forecast.to_frame("Previsão de Receita").copy()
    forecast_df.index.name = "Mês"
    forecast_df.reset_index(inplace=True)
    forecast_df["Mês"] = forecast_df["Mês"].dt.strftime("%Y-%m")
    forecast_df["Previsão de Receita"] = forecast_df["Previsão de Receita"].map(fmt_moeda)
    forecast_html = forecast_df.to_html(index=False, escape=False)

    testes_res_txt = (
        f"Ljung-Box p={fmt_num(res_tests.get('lb_p', float('nan')), 4)}; "
        f"Jarque-Bera p={fmt_num(res_tests.get('jb_p', float('nan')), 4)}; "
        f"Breusch-Pagan p={fmt_num(res_tests.get('bp_p', float('nan')), 4)}"
    )

    html = f"""
<html>
<head>
  <meta charset="utf-8">
  <title>Análise 2 – Previsão Mensal</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2, h3 {{ color: #333; }}
    img {{ max-width: 960px; display:block; margin-top: 10px; }}
    pre {{ background:#f5f5f5; padding:10px; border-radius:4px; white-space: pre-wrap; }}
    table, th, td {{ border:1px solid #aaa; border-collapse: collapse; padding:6px; }}
    th {{ background:#f0f0f0; }}
    .muted {{ color:#666; font-size: 0.95em; }}
  </style>
</head>
<body>
  <h1>Análise 2 – Previsão Mensal</h1>
  <p class="muted">Dataset: <b>Sales_2019 (12 meses)</b> • Horizonte: <b>3 meses</b></p>

  <h2>Métricas dos modelos</h2>
  {metrics_html}

  <h2>Melhor modelo</h2>
  <p><b>{best_model}</b></p>

  <h2>Análise de Resíduos</h2>
  <pre>{testes_res_txt}</pre>
  {"<img src='" + caminho_residuos + "' alt='Resíduos do modelo'>" if caminho_residuos else ""}

  <h2>Previsão dos próximos 3 meses</h2>
  {forecast_html}

  <h2>Série completa e previsões</h2>
  <img src="{plot_path.name}" alt="Análise 2 – Previsão Mensal (12 meses de 2019)">
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")

    print("Relatório gerado:", html_path)
    print("Modelo vencedor:", best_model)

    # Tentar abrir automaticamente o HTML no navegador padrão
    try:
        webbrowser.open(html_path.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()


