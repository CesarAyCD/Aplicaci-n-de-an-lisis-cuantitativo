from pathlib import Path

import numpy as np
import pandas as pd
import gc
from backtest import backtest_position, backtest_triple_barrier, summarize_backtest
from baselines import add_baseline_signal_features, make_baseline_signals
from carga_datos import bases_datos
from context_features import add_context_features
from feature_eng import TARGET, TRIPLE_BARRIER_PT, TRIPLE_BARRIER_SL, preprocesamiento
from markov_switching import add_regimes, fit_model
from model_metrics import evaluate_classification, extract_feature_importance, model_scope
from result_reporting import print_final_report, print_test_summary
from results_io import save_results, write_curve_frames
from train_models import TARGET_COL, fit_models, get_feature_columns, predict_model_probabilities

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
PURGE_BARS = TARGET

MAX_HOLDING_BARS = TARGET
COST_BPS = 1.0
THRESHOLDS = np.round(np.arange(0.50, 0.65, 0.01), 2)
MIN_EXPOSURE = 0.05
GROUP_COLS = ("ticker", "timeframe")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CURVES_PATH = OUTPUT_DIR / "backtest_equity_curves.parquet"
CURVES_TEMP_PATH = OUTPUT_DIR / "backtest_equity_curves.tmp.parquet"


# Divide los datos en los sets train, validación y test
def split(df, purge_bars=PURGE_BARS):
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO debe sumar 1")
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = train_end + int(n_rows * VAL_RATIO)
    if train_end <= purge_bars or val_end - train_end <= purge_bars or val_end >= n_rows:
        raise ValueError(f"No hay suficientes filas para dividir y purgar {n_rows} observaciones")
    train = df.iloc[:train_end - purge_bars].copy()
    validation = df.iloc[train_end:val_end - purge_bars].copy()
    test = df.iloc[val_end:].copy()
    return train, validation, test


def available_group_cols(df):
    return [column for column in GROUP_COLS if column in df.columns]


# Agrupa los datos por posiciones
def iter_groups(df, values=None):
    group_cols = available_group_cols(df)
    if not group_cols:
        yield {}, df, values
        return
    array = None if values is None else np.asarray(values)
    for key, positions in df.groupby(group_cols, sort=False).indices.items():
        labels = key if isinstance(key, tuple) else (key,)
        metadata = dict(zip(group_cols, labels))
        group_values = None if array is None else array[positions]
        yield metadata, df.iloc[positions].drop(columns=group_cols), group_values


# Crea y organiza un backtest
def run_single_backtest(df, signal, mode):
    if mode == "triple_barrier":
        return backtest_triple_barrier(
            df,
            signal,
            horizon=MAX_HOLDING_BARS,
            pt_mult=TRIPLE_BARRIER_PT,
            sl_mult=TRIPLE_BARRIER_SL,
            cost_bps=COST_BPS,
        )
    if mode == "position_shifted":
        return backtest_position(df, signal, COST_BPS, shift_signal=True)
    if mode == "position_same_bar":
        return backtest_position(df, signal, COST_BPS, shift_signal=False)
    raise ValueError(f"Modo de backtest no reconocido: {mode}")

# Backtest a todo el portafolio
def run_grouped_backtest(df, signal, mode):
    parts = []
    for metadata, group, group_signal in iter_groups(df, signal):
        result = run_single_backtest(group, group_signal, mode)
        for column, value in metadata.items():
            result[column] = value
        parts.append(result)
    if not parts:
        raise ValueError("No hay series para ejecutar el backtest")
    return pd.concat(parts)

# Estandariza los datos para registrar resultados
def make_summary_row(metadata, metrics, strategy_type, strategy_name,
                     split_name, threshold, mode, n_rows):
    row = {
        **metadata,
        "strategy_type": strategy_type,
        "strategy": strategy_name,
        "split": split_name,
        "threshold": threshold,
        "signal_mode": mode,
        "holding_period": np.nan,
        "max_holding_bars": MAX_HOLDING_BARS if mode == "triple_barrier" else np.nan,
        "profit_taking_mult": TRIPLE_BARRIER_PT if mode == "triple_barrier" else np.nan,
        "stop_loss_mult": TRIPLE_BARRIER_SL if mode == "triple_barrier" else np.nan,
        "cost_bps": COST_BPS,
        "n_rows": n_rows,
    }
    row.update(metrics)
    return row


def summarize_grouped(bt, strategy_type, strategy_name, split_name,
                      threshold, mode):
    rows = []
    for metadata, group, _ in iter_groups(bt):
        rows.append(make_summary_row(
            metadata, summarize_backtest(group), strategy_type, strategy_name,
            split_name, threshold, mode, len(group),
        ))
    return rows

# Genera la curva de equidad
def make_curve(bt, strategy_type, strategy_name, split_name, threshold, mode):
    columns = [
        "equity_strategy", "equity_buy_hold", "position", "signal",
        "strategy_log_return", "buy_hold_log_return",
    ]
    curve = bt[columns].copy()
    for column in GROUP_COLS:
        if column in bt:
            curve[column] = bt[column]
    curve["strategy_type"] = strategy_type
    curve["strategy"] = strategy_name
    curve["split"] = split_name
    curve["threshold"] = threshold
    curve["signal_mode"] = mode
    curve["holding_period"] = np.nan
    curve["max_holding_bars"] = (
        MAX_HOLDING_BARS if mode == "triple_barrier" else np.nan
    )
    return curve

# Ejecuta la simulación, extrae métricas y curvas
def backtest_signal(df, signal, strategy_type, strategy_name, split_name,
                    threshold=None, mode="triple_barrier"):
    bt = run_grouped_backtest(df, signal, mode)
    rows = summarize_grouped(
        bt, strategy_type, strategy_name, split_name, threshold, mode
    )
    curve = make_curve(
        bt, strategy_type, strategy_name, split_name, threshold, mode
    )
    return rows, curve

# Calcula la media agregada de los umbrales para seleccionar el mejor en validación
def aggregate_threshold_metrics(rows):
    metrics = pd.DataFrame(rows)
    numeric = metrics.select_dtypes(include=[np.number]).columns
    return metrics[numeric].mean().to_dict()

# Prueba los umbrales permitidos
def evaluate_thresholds(model_name, validation, probabilities):
    rows = []
    for threshold in THRESHOLDS:
        signal = (np.asarray(probabilities) >= threshold).astype(int)
        bt = run_grouped_backtest(validation, signal, "triple_barrier")
        per_series = [summarize_backtest(group) for _, group, _ in iter_groups(bt)]
        row = {
            "model": model_name,
            "threshold": threshold,
            "holding_period": np.nan,
            "max_holding_bars": MAX_HOLDING_BARS,
            "profit_taking_mult": TRIPLE_BARRIER_PT,
            "stop_loss_mult": TRIPLE_BARRIER_SL,
            "signal_mode": "triple_barrier",
            "threshold_selected_on": "validation_mean_across_series",
            "n_series": len(per_series),
        }
        row.update(aggregate_threshold_metrics(per_series))
        rows.append(row)
    return pd.DataFrame(rows)

# Elige el mejor umbral. Requiere una exposición mínima, retorno positivo y Sharpe
def best_threshold(validation_results):
    eligible = validation_results[
        (validation_results["exposure"] >= MIN_EXPOSURE)
        & (validation_results["annualized_excess_return"] > 0)
        & validation_results["sharpe"].notna()
    ].copy()
    candidates = eligible if not eligible.empty else validation_results.dropna(
        subset=["sharpe"]
    ).copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(
        ["sharpe", "annualized_excess_return", "max_drawdown"],
        ascending=[False, False, False],
    )
    selected = candidates.iloc[0].copy()
    selected["is_eligible"] = not eligible.empty
    return selected

# Prepara los datos usando feature_eng.py y markov_switching.py
def prepare_series(ticker, timeframe, raw, raw_datasets):
    features = preprocesamiento(raw)
    features = add_context_features(features, ticker, timeframe, raw, raw_datasets)
    if features.empty:
        raise ValueError("no quedan filas despues del preprocesamiento")
    features = add_baseline_signal_features(features)
    train_rows = int(len(features) * TRAIN_RATIO)
    if train_rows <= PURGE_BARS:
        raise ValueError("historial insuficiente")
    train_end_date = features.index[train_rows - 1]
    regime_model = fit_model(raw, train_end_date, 2)
    if regime_model.get("fallback", False):
        print(f"Markov neutro para {ticker} {timeframe}: {regime_model['reason']}")
    features = add_regimes(raw, features, regime_model)
    features["regime_daily"] = features["market_regime_adj"]
    train, validation, test = split(features)
    return {"train": train, "validation": validation, "test": test}


def prepare_datasets(raw_datasets):
    prepared = {}
    for (ticker, timeframe), raw in raw_datasets.items():
        try:
            splits = prepare_series(ticker, timeframe, raw, raw_datasets)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"Omitido {ticker} {timeframe}: {exc}")
            continue
        prepared[(ticker, timeframe)] = splits
        sizes = ", ".join(f"{name}={len(data)}" for name, data in splits.items())
        print(f"{ticker} {timeframe}: {sizes}")
    if not prepared:
        raise RuntimeError("Ninguna serie pudo prepararse")
    return prepared


def tag_splits(splits, ticker, timeframe):
    tagged = {}
    for split_name, data in splits.items():
        frame = data.copy()
        frame["ticker"] = ticker
        frame["timeframe"] = timeframe
        tagged[split_name] = frame
    return tagged

# Crea un dataset con todos los datos, se usa para entrenar el modelo generalista
def build_generalist_train(prepared):
    parts = []
    for (ticker, timeframe), splits in prepared.items():
        train = splits["train"].copy()
        train["ticker"] = ticker
        train["timeframe"] = timeframe
        parts.append(train)
    if not parts:
        raise RuntimeError("No hay observaciones de train para el modelo generalista")
    return pd.concat(parts, axis=0).sort_index(kind="stable")

def add_generalist_suffix(models):
    return {f"{name}_generalist": model for name, model in models.items()}

def merge_probabilities(*probability_maps):
    merged = {}
    for probabilities in probability_maps:
        overlap = set(merged).intersection(probabilities)
        if overlap:
            raise ValueError(f"Modelos duplicados al combinar probabilidades: {overlap}")
        merged.update(probabilities)
    return merged

# Crea las estrategias base
def evaluate_baselines(datasets):
    summary_rows, curves = [], []
    for split_name, data in datasets.items():
        for strategy_name, signal in make_baseline_signals(data).items():
            mode = (
                "position_same_bar"
                if strategy_name == "always_long"
                else "triple_barrier"
            )
            rows, curve = backtest_signal(
                data, signal, "baseline", strategy_name, split_name, mode=mode
            )
            summary_rows.extend(rows)
            curves.append(curve)
    return summary_rows, curves


def predict_all(models, datasets, feature_cols):
    return {
        split_name: predict_model_probabilities(models, data, feature_cols)
        for split_name, data in datasets.items()
    }

# Usa best_threshold para elegir el mejor umbral, como se mencionó en su sección
def select_models_on_validation(
    models, validation, validation_probabilities, ticker=None, timeframe=None
):
    threshold_frames, selected_rows = [], []
    for model_name in models:
        validation_results = evaluate_thresholds(
            model_name, validation, validation_probabilities[model_name]
        )
        validation_results["ticker"] = ticker
        validation_results["timeframe"] = timeframe
        validation_results["model_scope"] = (
            model_scope(model_name)
        )
        threshold_frames.append(validation_results)
        selected = best_threshold(validation_results)
        if selected is None:
            print(f"{model_name}: no tiene ningun threshold con Sharpe calculable")
            continue
        selected_rows.append(selected)
        status = "elegible" if selected["is_eligible"] else "solo diagnostico"
        print(
            f"{model_name}: threshold elegido en validation = "
            f"{selected['threshold']:.2f} ({status})"
        )

    if not selected_rows:
        raise RuntimeError(
            "Ningun modelo tiene un threshold con Sharpe calculable en validation."
        )

    selection = pd.DataFrame(selected_rows).sort_values(
        ["sharpe", "annualized_excess_return", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selection["validation_rank"] = np.arange(1, len(selection) + 1)
    eligible_selection = selection[selection["is_eligible"]]
    selection["is_winner"] = False
    if eligible_selection.empty:
        print(
            "Sin ganador elegible en validation; la serie continuara solo para "
            "evaluacion diagnostica."
        )
    else:
        winner_index = eligible_selection["sharpe"].idxmax()
        selection.loc[winner_index, "is_winner"] = True
        winner = selection.loc[winner_index, "model"]
        print(f"Ganador elegido exclusivamente en validation: {winner}")
    selection["selection_split"] = "validation"
    selection["ticker"] = ticker
    selection["timeframe"] = timeframe
    return selection, threshold_frames

# Aplica los umbrales calculados previamente al set de test
def evaluate_models(datasets, probabilities, selection):
    summary_rows, curves = [], []
    thresholds = selection.set_index("model")["threshold"].to_dict()
    for model_name, threshold in thresholds.items():
        for split_name, data in datasets.items():
            signal = (probabilities[split_name][model_name] >= threshold).astype(int)
            rows, curve = backtest_signal(
                data, signal, "model", model_name, split_name,
                threshold=threshold, mode="triple_barrier",
            )
            scope = model_scope(model_name)
            for row in rows:
                row["model_scope"] = scope
            curve["model_scope"] = scope
            summary_rows.extend(rows)
            curves.append(curve)
    return summary_rows, curves

# Función principal. Sirve como orquestador de todo el sistema. 
def main():
    print("Cargando datos...")
    raw = bases_datos()
    print("Aplicando preprocesamiento y Markov-Switching por serie...")
    prepared = prepare_datasets(raw)
    del raw
    gc.collect()
    all_rows, all_thresholds, all_selections = [], [], []
    all_classification_rows, all_feature_importance = [], []
    curve_writer = None
    CURVES_TEMP_PATH.unlink(missing_ok=True)

    print("\nEntrenamiento generalista con todos los splits de train")
    generalist_train = build_generalist_train(prepared)
    generalist_feature_cols = get_feature_columns(generalist_train)
    print(
        f"Entrenando modelos generalistas con {len(generalist_train):,} "
        f"observaciones de {len(prepared)} series..."
    )
    generalist_models = add_generalist_suffix(fit_models(generalist_train, generalist_feature_cols))
    generalist_importance = extract_feature_importance(generalist_models, generalist_feature_cols)
    generalist_importance["ticker"] = "ALL"
    generalist_importance["timeframe"] = "ALL"
    generalist_importance["model_scope"] = "generalist"
    all_feature_importance.append(generalist_importance)
    del generalist_train
    gc.collect()

    for ticker_timeframe in list(prepared):
        ticker, timeframe = ticker_timeframe
        splits = prepared.pop(ticker_timeframe)
        print(f"\nEntrenamiento de modelo especializado: {ticker} {timeframe}")

        series_splits = tag_splits(splits, ticker, timeframe)
        feature_cols = get_feature_columns(series_splits["train"])

        print("Entrenando modelos...")
        specific_models = fit_models(series_splits["train"], feature_cols)
        specific_importance = extract_feature_importance(
            specific_models, feature_cols
        )
        specific_importance["ticker"] = ticker
        specific_importance["timeframe"] = timeframe
        specific_importance["model_scope"] = "specific"
        all_feature_importance.append(specific_importance)
        print("Generando probabilidades de validación...")
        specific_validation_probabilities = predict_model_probabilities(
            specific_models, series_splits["validation"], feature_cols
        )
        generalist_validation_probabilities = predict_model_probabilities(
            generalist_models,
            series_splits["validation"],
            generalist_feature_cols,
        )
        validation_probabilities = merge_probabilities(
            specific_validation_probabilities,
            generalist_validation_probabilities,
        )
        candidate_models = {**specific_models, **generalist_models}
        print("Eligiendo umbrales en validation...")
        selection, thresholds = select_models_on_validation(
            candidate_models,
            series_splits["validation"],
            validation_probabilities,
            ticker=ticker,
            timeframe=timeframe,
        )
        print("Umbrales congelados. Evaluando test...")
        specific_probabilities = predict_all(
            specific_models, series_splits, feature_cols
        )
        generalist_probabilities = predict_all(
            generalist_models, series_splits, generalist_feature_cols
        )
        probabilities = {
            split_name: merge_probabilities(
                specific_probabilities[split_name],
                generalist_probabilities[split_name],
            )
            for split_name in series_splits
        }
        model_rows, model_curves = evaluate_models(
            series_splits, probabilities, selection
        )
        classification_rows = evaluate_classification(
            series_splits, probabilities, selection, TARGET_COL
        )
        baseline_rows, baseline_curves = evaluate_baselines(series_splits)

        all_rows.extend(baseline_rows + model_rows)
        curve_writer = write_curve_frames(
            model_curves + baseline_curves, CURVES_TEMP_PATH, curve_writer
        )
        all_thresholds.extend(thresholds)
        all_selections.append(selection)
        all_classification_rows.extend(classification_rows)

        # Liberamos RAM borrando lo que no se necesita ya
        del series_splits
        del feature_cols
        del specific_models
        del specific_importance
        del specific_validation_probabilities
        del generalist_validation_probabilities
        del validation_probabilities
        del candidate_models
        del specific_probabilities
        del generalist_probabilities
        del probabilities
        del model_rows
        del model_curves
        del classification_rows
        del baseline_rows
        del baseline_curves
        del splits
        gc.collect()

    if curve_writer is None:
        raise RuntimeError("No se generaron curvas de backtest")
    curve_writer.close()
    CURVES_TEMP_PATH.replace(CURVES_PATH)
    selection = pd.concat(all_selections, ignore_index=True)

    summary, comparison, ranking, paths = save_results(
        OUTPUT_DIR,
        CURVES_PATH,
        all_rows,
        all_thresholds,
        selection,
        all_classification_rows,
        all_feature_importance,
    )
    print("\nResultados guardados:")
    for path in paths.values():
        print(f"- {path}")
    print("\nResumen test ordenado por Sharpe:")
    print_test_summary(summary)
    print_final_report(comparison, ranking)


if __name__ == "__main__":
    main()
