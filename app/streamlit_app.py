import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Dashboard de Backtesting Cuantitativo",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("BACKTEST_OUTPUT_DIR", BASE_DIR / "outputs"))
REQUIRED_FILES = {
    "summary": OUTPUT_DIR / "backtest_summary.parquet",
    "thresholds": OUTPUT_DIR / "backtest_threshold_validation.parquet",
}
OPTIONAL_FILES = {
    "curves": OUTPUT_DIR / "backtest_equity_curves.parquet",
    "model_report": OUTPUT_DIR / "validation_vs_test_model_report.parquet",
    "selection": OUTPUT_DIR / "model_selection_validation.parquet",
    "ranking": OUTPUT_DIR / "ranking_validation_vs_test.parquet",
    "classification": OUTPUT_DIR / "classification_metrics.parquet",
    "feature_importance": OUTPUT_DIR / "feature_importance.parquet",
}


@st.cache_data(show_spinner="Cargando resultados precomputados…")
def load_parquet(
    path_string: str, modified_ns: int, file_size: int
) -> pd.DataFrame:
    return pd.read_parquet(path_string)


@st.cache_data(show_spinner="Cargando curva seleccionada…")
def load_curve_slice(
    path_string: str,
    modified_ns: int,
    file_size: int,
    ticker: str,
    timeframe: str,
    split: str,
    strategy: str,
) -> pd.DataFrame:
    filters = [
        ("ticker", "==", ticker),
        ("timeframe", "==", timeframe),
        ("split", "==", split),
        ("strategy", "==", strategy),
    ]
    return prepare_curves(pd.read_parquet(path_string, filters=filters))


def parquet_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def load_outputs() -> dict[str, pd.DataFrame | None]:
    missing = [path for path in REQUIRED_FILES.values() if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        st.error(
            f"Faltan archivos requeridos: {names}. Ejecuta primero "
            "`python src/run_backtest.py`."
        )
        st.stop()

    outputs: dict[str, pd.DataFrame | None] = {
        name: load_parquet(*parquet_cache_key(path))
        for name, path in REQUIRED_FILES.items()
    }
    outputs.update({
        name: (
            None if name == "curves"
            else load_parquet(*parquet_cache_key(path)) if path.exists()
            else None
        )
        for name, path in OPTIONAL_FILES.items()
    })
    return outputs


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        st.error(
            f"El archivo `{name}` no contiene las columnas requeridas: "
            f"{', '.join(sorted(missing))}. Regenera los resultados con "
            "`python src/run_backtest.py`."
        )
        st.stop()


def prepare_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if "date" in curves.columns:
        result = curves.copy()
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        return result
    result = curves.reset_index()
    date_column = result.columns[0]
    result = result.rename(columns={date_column: "date"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result


def filter_by_selection(
    df: pd.DataFrame,
    ticker: str | None = None,
    timeframe: str | None = None,
    split: str | None = None,
    strategy: str | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    filters = {
        "ticker": ticker,
        "timeframe": timeframe,
        "split": split,
        "strategy": strategy,
    }
    for column, value in filters.items():
        if value is not None and column in df.columns:
            mask &= df[column].astype(str).eq(str(value))
    return df.loc[mask].copy()


def apply_strategy_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if "strategy_type" not in df.columns or scope == "Todos":
        return df
    expected = "model" if scope == "Solo modelos" else "baseline"
    return df[df["strategy_type"] == expected].copy()


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("strategy_type", "strategy", "split") if column in df]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not keys:
        return df.copy()
    return df.groupby(keys, as_index=False, dropna=False)[numeric].mean()


def compute_drawdown(equity: pd.Series) -> pd.Series:
    equity = pd.to_numeric(equity, errors="coerce")
    return equity / equity.cummax() - 1


def reduce_for_chart(df: pd.DataFrame, max_points: int = 50_000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    positions = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[np.unique(positions)]


def metric_value(row: pd.Series, column: str, percent: bool = False) -> str:
    value = row.get(column, np.nan)
    if pd.isna(value):
        return "N/D"
    if percent:
        return f"{value:.2%}"
    if column in {"n_trades", "n_exits"}:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def is_percentage_column(column: str) -> bool:
    percentage_terms = (
        "return",
        "drawdown",
        "exposure",
        "total_cost",
        "hit_rate",
        "avg_active_bar_return",
        "accuracy",
        "precision",
        "recall",
        "positive_rate",
    )
    return any(term in column for term in percentage_terms)


def table_number_config(df: pd.DataFrame) -> dict:
    config = {}
    integer_terms = (
        "n_trades", "n_exits", "rank", "n_rows", "n_series", "holding_period"
    )
    for column in df.select_dtypes(include=[np.number]).columns:
        if is_percentage_column(column) or column in {
            "f1", "roc_auc", "average_precision", "importance"
        }:
            config[column] = st.column_config.NumberColumn(
                format="percent", step=0.0001
            )
        elif any(term in column for term in integer_terms):
            config[column] = st.column_config.NumberColumn(format="%.0f")
        elif column in {"sharpe", "sortino", "threshold"} or column.startswith(
            ("sharpe_", "sortino_", "threshold_")
        ):
            config[column] = st.column_config.NumberColumn(format="%.3f")
        elif column == "cost_bps":
            config[column] = st.column_config.NumberColumn(format="%.1f")
        else:
            config[column] = st.column_config.NumberColumn(format="%.4f")
    return config


def official_validation_winner(
    selection: pd.DataFrame | None, ticker: str, timeframe: str
) -> pd.Series | None:
    if selection is None:
        return None
    required = {"ticker", "timeframe", "model", "is_winner"}
    if not required.issubset(selection.columns):
        return None
    series_selection = filter_by_selection(selection, ticker, timeframe)
    winners = series_selection[series_selection["is_winner"].fillna(False)]
    if winners.empty:
        return None
    return winners.iloc[0]


def build_validation_test_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sharpe",
        "annualized_excess_return",
        "max_drawdown",
        "exposure",
    ]
    metrics = [column for column in metrics if column in summary]
    data = summary[summary["split"].isin(["validation", "test"])].copy()
    if data.empty or not metrics:
        return pd.DataFrame()
    data = aggregate_summary(data)
    table = data.pivot_table(
        index=["strategy_type", "strategy"],
        columns="split",
        values=metrics,
        aggfunc="first",
    )
    table.columns = [f"{metric}_{split}" for metric, split in table.columns]
    table = table.reset_index()
    for metric in metrics:
        validation_col = f"{metric}_validation"
        test_col = f"{metric}_test"
        if validation_col in table and test_col in table:
            table[f"{metric}_test_minus_validation"] = (
                table[test_col] - table[validation_col]
            )
    return table

def line_figure(title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title_text="Serie",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def plot_equity_curve(curves: pd.DataFrame, show_buy_hold: bool) -> go.Figure:
    data = reduce_for_chart(curves.sort_values("date"))
    label = data.iloc[0]
    figure = line_figure(
        f"Equity — {label['ticker']} · {label['timeframe']} · "
        f"{label['split']} · {label['strategy']}",
        "Crecimiento de $1",
    )
    figure.add_trace(go.Scatter(
        x=data["date"], y=data["equity_strategy"], name=str(label["strategy"])
    ))
    if show_buy_hold and "equity_buy_hold" in data:
        figure.add_trace(go.Scatter(
            x=data["date"], y=data["equity_buy_hold"], name="Buy & Hold"
        ))
    return figure


def plot_drawdown(curves: pd.DataFrame, show_buy_hold: bool) -> go.Figure:
    data = curves.sort_values("date").copy()
    data["drawdown_strategy"] = compute_drawdown(data["equity_strategy"])
    if "equity_buy_hold" in data:
        data["drawdown_buy_hold"] = compute_drawdown(data["equity_buy_hold"])
    data = reduce_for_chart(data)
    figure = line_figure("Drawdown", "Drawdown")
    figure.add_trace(go.Scatter(
        x=data["date"], y=data["drawdown_strategy"], name="Estrategia",
        fill="tozeroy",
    ))
    if show_buy_hold and "drawdown_buy_hold" in data:
        figure.add_trace(go.Scatter(
            x=data["date"], y=data["drawdown_buy_hold"], name="Buy & Hold"
        ))
    figure.update_yaxes(tickformat=".0%")
    return figure

def plot_binary_series(curves: pd.DataFrame, column: str, title: str) -> go.Figure:
    data = reduce_for_chart(curves.sort_values("date"))
    figure = line_figure(title, column.capitalize())
    figure.add_trace(go.Scatter(
        x=data["date"], y=data[column], name=column.capitalize(),
        mode="lines", line_shape="hv",
    ))
    figure.update_yaxes(tickvals=[0, 1], ticktext=["0", "1"], range=[-0.1, 1.1])
    return figure

def plot_thresholds(data: pd.DataFrame, model: str) -> tuple[go.Figure, go.Figure]:
    data = data.sort_values("threshold")
    sharpe = go.Figure()
    sharpe.add_trace(go.Scatter(
        x=data["threshold"], y=data["sharpe"], mode="lines+markers", name="Sharpe"
    ))
    sharpe.update_layout(
        title=f"Sharpe por threshold — {model}", xaxis_title="Threshold",
        yaxis_title="Sharpe", hovermode="x unified",
    )

    performance = go.Figure()
    if "total_return" in data:
        performance.add_trace(go.Scatter(
            x=data["threshold"], y=data["total_return"], mode="lines+markers",
            name="Total return",
        ))
    if "exposure" in data:
        performance.add_trace(go.Scatter(
            x=data["threshold"], y=data["exposure"], mode="lines+markers",
            name="Exposure",
        ))
    performance.update_layout(
        title=f"Retorno y exposure por threshold — {model}",
        xaxis_title="Threshold", yaxis_title="Valor", hovermode="x unified",
    )
    performance.update_yaxes(tickformat=".0%")
    return sharpe, performance

outputs = load_outputs()
summary = outputs["summary"]
thresholds = outputs["thresholds"]

validate_columns(
    summary,
    {"ticker", "timeframe", "strategy_type", "strategy", "split", "sharpe", "exposure"},
    "backtest_summary.parquet",
)
validate_columns(
    thresholds,
    {"ticker", "timeframe", "model", "threshold", "sharpe", "exposure"},
    "backtest_threshold_validation.parquet",
)

st.title("Dashboard de Backtesting Cuantitativo")
st.caption("Resultados precomputados a partir de datos locales IBKR en Parquet.")

with st.sidebar:
    st.header("Filtros")
    if st.button("Recargar resultados", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    tickers = sorted(summary["ticker"].dropna().astype(str).unique())
    ticker = st.selectbox("Ticker", tickers)
    ticker_rows = summary[summary["ticker"].astype(str) == ticker]
    timeframes = sorted(ticker_rows["timeframe"].dropna().astype(str).unique())
    timeframe = st.selectbox("Timeframe", timeframes)
    series_rows = filter_by_selection(summary, ticker, timeframe)
    splits = sorted(series_rows["split"].dropna().astype(str).unique())
    default_split = splits.index("test") if "test" in splits else 0
    split = st.selectbox("Split", splits, index=default_split)
    scope = st.radio(
        "Estrategias visibles",
        ["Solo modelos", "Solo baselines", "Todos"],
        index=2,
        horizontal=False,
    )
    show_buy_hold = st.checkbox("Comparar con Buy & Hold", value=True)
    global_view = st.checkbox(
        "Resumen agregado global",
        value=False,
        help="Promedia métricas entre series. Las curvas siempre pertenecen a una serie concreta.",
    )

    scoped_rows = apply_strategy_scope(
        filter_by_selection(summary, ticker, timeframe, split), scope
    )
    strategy_options = sorted(scoped_rows["strategy"].dropna().astype(str).unique())
    if not strategy_options:
        st.warning("No hay estrategias para esta combinación de filtros.")
        st.stop()
    strategy = st.selectbox("Estrategia / modelo", strategy_options)

series_summary = filter_by_selection(summary, ticker, timeframe)
ranking_source = summary.copy() if global_view else series_summary.copy()
ranking_source = apply_strategy_scope(ranking_source, scope)
ranking = ranking_source[ranking_source["split"] == split].copy()
if global_view:
    ranking = aggregate_summary(ranking)
ranking = ranking.sort_values("sharpe", ascending=False, na_position="last")

selected_summary = filter_by_selection(
    summary, ticker, timeframe, split=split, strategy=strategy
)
selected_strategy_type = (
    selected_summary.iloc[0].get("strategy_type")
    if not selected_summary.empty else None
)

tabs = st.tabs([
    "Resumen",
    "Curvas",
    "Thresholds",
    "Validation vs Test",
    "Clasificación e importancia",
    "Metodología",
])

with tabs[0]:
    st.subheader("Resumen del experimento")
    kpi_columns = st.columns(5)
    kpi_columns[0].metric("Tickers", summary["ticker"].nunique())
    kpi_columns[1].metric("Timeframes", summary["timeframe"].nunique())
    kpi_columns[2].metric("Estrategias", summary["strategy"].nunique())
    kpi_columns[3].metric("Splits", summary["split"].nunique())
    series_count = summary[["ticker", "timeframe"]].drop_duplicates().shape[0]
    kpi_columns[4].metric("Series", series_count)

    st.subheader("Ranking de estrategias")
    requested_columns = [
        "ticker", "timeframe", "strategy_type", "model_scope", "strategy", "split",
        "threshold", "holding_period", "max_holding_bars", "profit_taking_mult",
        "stop_loss_mult", "cost_bps", "total_return", "buy_hold_return",
        "excess_return", "annualized_return", "annualized_excess_return",
        "max_drawdown", "buy_hold_max_drawdown", "sharpe", "sortino", "exposure",
        "n_trades", "total_cost",
    ]
    visible_columns = [column for column in requested_columns if column in ranking]
    ranking_display = ranking[visible_columns]
    st.dataframe(
        ranking_display,
        width="stretch",
        hide_index=True,
        column_config=table_number_config(ranking_display),
    )

    st.subheader("KPIs de la estrategia seleccionada")
    if selected_summary.empty:
        st.info("No hay métricas para la estrategia seleccionada.")
    else:
        row = selected_summary.iloc[0]
        first_row = st.columns(4)
        first_row[0].metric("Total return", metric_value(row, "total_return", True))
        first_row[1].metric("Buy & Hold", metric_value(row, "buy_hold_return", True))
        first_row[2].metric("Excess return", metric_value(row, "excess_return", True))
        first_row[3].metric("Sharpe", metric_value(row, "sharpe"))
        second_row = st.columns(4)
        second_row[0].metric("Max drawdown", metric_value(row, "max_drawdown", True))
        second_row[1].metric("Exposure", metric_value(row, "exposure", True))
        second_row[2].metric("Número de trades", metric_value(row, "n_trades"))
        second_row[3].metric("Threshold", metric_value(row, "threshold"))

with tabs[1]:
    curves_path = OPTIONAL_FILES["curves"]
    if not curves_path.exists():
        st.info(
            "El archivo `backtest_equity_curves.parquet` es opcional y no está "
            "disponible. El resto del dashboard funciona sin curvas."
        )
    else:
        selected_curve = load_curve_slice(
            *parquet_cache_key(curves_path),
            ticker,
            timeframe,
            split,
            strategy,
        )
        validate_columns(
            selected_curve,
            {
                "date", "ticker", "timeframe", "strategy", "split",
                "equity_strategy", "signal", "position",
            },
            "backtest_equity_curves.parquet",
        )
        if selected_curve.empty:
            st.warning("No hay curvas para la selección actual.")
        else:
            date_range = (
                f"{selected_curve['date'].min():%Y-%m-%d} → "
                f"{selected_curve['date'].max():%Y-%m-%d}"
            )
            st.caption(f"Rango seleccionado: {date_range}")
            st.plotly_chart(
                plot_equity_curve(selected_curve, show_buy_hold), width="stretch"
            )
            st.plotly_chart(
                plot_drawdown(selected_curve, show_buy_hold), width="stretch"
            )
            chart_columns = st.columns(2)
            chart_columns[0].plotly_chart(
                plot_binary_series(selected_curve, "signal", "Señal del modelo"),
                width="stretch",
            )
            chart_columns[1].plotly_chart(
                plot_binary_series(selected_curve, "position", "Posición de mercado"),
                width="stretch",
            )

with tabs[2]:
    threshold_data = filter_by_selection(thresholds, ticker, timeframe)
    threshold_data = threshold_data[threshold_data["model"].astype(str) == strategy]
    if threshold_data.empty:
        if selected_strategy_type == "baseline":
            st.info("Esta estrategia no tiene threshold de modelo porque es un baseline.")
        else:
            st.info("No hay resultados de thresholds para esta selección.")
    else:
        sharpe_figure, performance_figure = plot_thresholds(threshold_data, strategy)
        st.plotly_chart(sharpe_figure, width="stretch")
        st.plotly_chart(performance_figure, width="stretch")
        threshold_columns = [
            column for column in (
                "threshold", "sharpe", "total_return", "annualized_excess_return",
                "max_drawdown", "exposure",
            ) if column in threshold_data
        ]
        threshold_display = threshold_data[threshold_columns].sort_values("threshold")
        st.dataframe(
            threshold_display,
            width="stretch",
            hide_index=True,
            column_config=table_number_config(threshold_display),
        )

with tabs[3]:
    st.subheader("Comparación validation vs test")
    model_report = outputs.get("model_report")
    if model_report is not None and {"ticker", "timeframe"}.issubset(model_report.columns):
        comparison = filter_by_selection(model_report, ticker, timeframe)
    else:
        comparison = build_validation_test_table(series_summary)
    if comparison.empty:
        st.info("No hay observaciones simultáneas de validation y test.")
    else:
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
            column_config=table_number_config(comparison),
        )

    ranking_report = outputs.get("ranking")
    if ranking_report is not None:
        ranking_view = filter_by_selection(ranking_report, ticker, timeframe)
        if not ranking_view.empty:
            st.subheader("Predictividad del ranking")
            st.dataframe(
                ranking_view,
                width="stretch",
                hide_index=True,
                column_config=table_number_config(ranking_view),
            )

    st.subheader("Mejor modelo en validación")
    winner = official_validation_winner(outputs.get("selection"), ticker, timeframe)
    if winner is None:
        if outputs.get("selection") is None:
            st.warning(
                "No está disponible `model_selection_validation.parquet`; "
                "no se puede identificar el ganador oficial."
            )
        else:
            st.warning("Esta serie no tiene un ganador oficial elegible en validation.")
    else:
        winner_name = str(winner["model"])
        winner_test = filter_by_selection(
            summary, ticker, timeframe, split="test", strategy=winner_name
        )
        st.success(f"Ganador en validation: **{winner_name}**")
        validation_cards = st.columns(3)
        validation_cards[0].metric("Sharpe validation", metric_value(winner, "sharpe"))
        validation_cards[1].metric(
            "Exceso anual validation",
            metric_value(winner, "annualized_excess_return", True),
        )
        validation_cards[2].metric(
            "Exposure validation", metric_value(winner, "exposure", True)
        )
        if not winner_test.empty:
            test_row = winner_test.iloc[0]
            test_cards = st.columns(3)
            test_cards[0].metric("Sharpe test", metric_value(test_row, "sharpe"))
            test_cards[1].metric(
                "Exceso anual test",
                metric_value(test_row, "annualized_excess_return", True),
            )
            test_cards[2].metric(
                "Max drawdown test", metric_value(test_row, "max_drawdown", True)
            )
        st.caption(
            "Ganador oficial leído de model_selection_validation.parquet. "
            "Se seleccionó usando únicamente validation; test se reserva como "
            "evaluación fuera de muestra."
        )

with tabs[4]:
    st.subheader("Métricas de clasificación")
    classification = outputs.get("classification")
    if classification is None:
        st.info(
            "No está disponible `classification_metrics.parquet`. Regenera los "
            "resultados con `python src/run_backtest.py`."
        )
    elif strategy not in classification["model"].astype(str).unique():
        st.info("Las estrategias baseline no tienen métricas de clasificación.")
    else:
        classification_view = filter_by_selection(
            classification, ticker, timeframe
        )
        classification_view = classification_view[
            classification_view["model"].astype(str) == strategy
        ].copy()
        st.dataframe(
            classification_view,
            width="stretch",
            hide_index=True,
            column_config=table_number_config(classification_view),
        )
        metric_split = split if split in {"validation", "test"} else "test"
        metric_rows = classification_view[
            classification_view["split"] == metric_split
        ]
        if not metric_rows.empty:
            metric_row = metric_rows.iloc[0]
            cards = st.columns(5)
            cards[0].metric("Accuracy", metric_value(metric_row, "accuracy", True))
            cards[1].metric(
                "Balanced accuracy",
                metric_value(metric_row, "balanced_accuracy", True),
            )
            cards[2].metric("Precision", metric_value(metric_row, "precision", True))
            cards[3].metric("Recall", metric_value(metric_row, "recall", True))
            cards[4].metric("F1", metric_value(metric_row, "f1", True))

            confusion = np.array([
                [metric_row["tn"], metric_row["fp"]],
                [metric_row["fn"], metric_row["tp"]],
            ])
            confusion_figure = go.Figure(go.Heatmap(
                z=confusion,
                x=["Predicho 0", "Predicho 1"],
                y=["Real 0", "Real 1"],
                text=confusion,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ))
            confusion_figure.update_layout(
                title=f"Matriz de confusión — {metric_split}",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(confusion_figure, width="stretch")

    st.subheader("Feature importance")
    importance = outputs.get("feature_importance")
    if importance is None:
        st.info(
            "No está disponible `feature_importance.parquet`. Regenera los "
            "resultados con `python src/run_backtest.py`."
        )
    elif selected_strategy_type == "baseline":
        st.info("Las estrategias baseline no tienen feature importance.")
    else:
        importance_view = importance[
            importance["model"].astype(str) == strategy
        ].copy()
        if strategy.endswith("_generalist"):
            importance_view = filter_by_selection(
                importance_view, "ALL", "ALL"
            )
        else:
            importance_view = filter_by_selection(
                importance_view, ticker, timeframe
            )
        if importance_view.empty:
            st.info(
                "Este modelo no expone feature importance nativa. "
                "HistGradientBoosting queda sin importancia para evitar el coste "
                "masivo de permutation importance."
            )
        else:
            max_features = min(30, len(importance_view))
            top_features = st.slider(
                "Número de features", 5, max_features, min(15, max_features)
            ) if max_features >= 5 else max_features
            importance_view = importance_view.nsmallest(
                top_features, "importance_rank"
            ).sort_values("importance")
            importance_figure = go.Figure(go.Bar(
                x=importance_view["importance"],
                y=importance_view["feature"],
                orientation="h",
                customdata=importance_view[["raw_importance", "importance_method"]],
                hovertemplate=(
                    "%{y}<br>Importancia: %{x:.2%}<br>Valor bruto: "
                    "%{customdata[0]:.5f}<br>Método: %{customdata[1]}<extra></extra>"
                ),
            ))
            importance_figure.update_layout(
                title=f"Top features — {strategy}",
                xaxis_title="Importancia normalizada",
                yaxis_title="Feature",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            importance_figure.update_xaxes(tickformat=".0%")
            st.plotly_chart(importance_figure, width="stretch")
            st.dataframe(
                importance_view.sort_values("importance_rank"),
                width="stretch",
                hide_index=True,
                column_config=table_number_config(importance_view),
            )

with tabs[5]:
    st.subheader("Notas metodológicas")
    st.markdown(
        """
        - Los modelos fueron entrenados offline mediante `src/run_backtest.py`.
        - La aplicación consume exclusivamente resultados precomputados.
        - Test no interviene en la elección de modelos ni umbrales.
        - Los modelos, umbrales y resultados se calculan por ticker y timeframe.
        - Los horizontes están medidos en barras, no en días.
        - No existen conexiones a IBKR ni modelos en vivo en este dashboard.
        - El diseño prioriza una demostración estable y reproducible.
        - Para más información consulte readme.md
        """
    )
    st.caption(f"Directorio de resultados: `{OUTPUT_DIR}`")
