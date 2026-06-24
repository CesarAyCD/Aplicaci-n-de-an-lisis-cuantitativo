import numpy as np
import pandas as pd


GROUP_COLS = ("ticker", "timeframe")

# Filtra y promedia las métricas numéricas de las estrategias
def aggregate_model_metrics(summary):
    models = summary[summary["strategy_type"] == "model"].copy()
    numeric = models.select_dtypes(include=[np.number]).columns.tolist()
    keys = [
        column for column in (*GROUP_COLS, "strategy", "split")
        if column in models.columns
    ]
    return models.groupby(keys, as_index=False)[numeric].mean()

# Hace una comparación del rendimiento en las fases de validación y test
# El objetivo es descubrir si el modelo logró generalizar
def build_validation_test_report(summary, selection):
    metrics = aggregate_model_metrics(summary)
    wanted = [
        "sharpe", "annualized_excess_return", "max_drawdown", "exposure"
    ]
    group_cols = [column for column in GROUP_COLS if column in metrics.columns]
    comparison = metrics[metrics["split"].isin(["validation", "test"])].pivot(
        index=group_cols + ["strategy"], columns="split", values=wanted
    )
    comparison.columns = [
        f"{metric}_{split_name}" for metric, split_name in comparison.columns
    ]
    comparison = comparison.reset_index().rename(columns={"strategy": "model"})
    selection_cols = group_cols + [
        "model", "model_scope", "threshold", "is_eligible", "validation_rank",
        "is_winner",
    ]
    merge_keys = group_cols + ["model"]
    comparison = selection[selection_cols].merge(
        comparison, on=merge_keys, how="left"
    )
    if group_cols:
        comparison["test_rank"] = comparison.groupby(group_cols)["sharpe_test"].rank(
            method="min", ascending=False
        ).astype("Int64")
    else:
        comparison["test_rank"] = comparison["sharpe_test"].rank(
            method="min", ascending=False
        ).astype("Int64")
    comparison["rank_shift"] = comparison["test_rank"] - comparison["validation_rank"]
    comparison["sharpe_degradation"] = (
        comparison["sharpe_validation"] - comparison["sharpe_test"]
    )
    comparison["annualized_excess_return_degradation"] = (
        comparison["annualized_excess_return_validation"]
        - comparison["annualized_excess_return_test"]
    )
    comparison["max_drawdown_deterioration"] = (
        comparison["max_drawdown_validation"] - comparison["max_drawdown_test"]
    )
    comparison["exposure_change"] = (
        comparison["exposure_test"] - comparison["exposure_validation"]
    )
    return comparison.sort_values(group_cols + ["validation_rank"]).reset_index(drop=True)

# Nuevamente hace comparaciones entre fases del entrenamiento
# Esta vez usa el coeficiente de correlación de Spearman para evaluar generalización
def ranking_diagnostics(report):
    group_cols = [column for column in GROUP_COLS if column in report.columns]
    groups = report.groupby(group_cols, sort=False) if group_cols else [((), report)]
    rows = []
    for key, group in groups:
        valid = group.dropna(subset=["sharpe_validation", "sharpe_test"])
        if valid.empty:
            continue
        validation_winner = valid.loc[valid["validation_rank"].idxmin()]
        test_winner = valid.loc[valid["test_rank"].idxmin()]
        evaluable = len(valid) >= 2
        labels = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(group_cols, labels)),
            "n_models": len(valid),
            "ranking_evaluable": evaluable,
            "validation_winner": validation_winner["model"],
            "test_winner": test_winner["model"],
            "top_model_match": (
                validation_winner["model"] == test_winner["model"]
                if evaluable else np.nan
            ),
            "validation_winner_test_rank": int(validation_winner["test_rank"]),
            "spearman_rank_correlation": (
                valid["validation_rank"].corr(valid["test_rank"], method="spearman")
                if evaluable else np.nan
            ),
        })
    return pd.DataFrame(rows)


def print_test_summary(summary):
    columns = [
        "ticker", "timeframe", "strategy_type", "strategy", "sharpe",
        "total_return", "buy_hold_return", "max_drawdown", "exposure",
    ]
    test = summary[summary["split"] == "test"].sort_values(
        "sharpe", ascending=False
    )
    print(test[columns].to_string(index=False))


def print_final_report(comparison, ranking):
    columns = [column for column in GROUP_COLS if column in comparison.columns] + [
        "model", "model_scope", "is_eligible", "is_winner", "validation_rank",
        "test_rank", "sharpe_validation", "sharpe_test", "sharpe_degradation",
        "annualized_excess_return_validation", "annualized_excess_return_test",
        "max_drawdown_validation", "max_drawdown_test", "exposure_validation",
        "exposure_test",
    ]
    print("\nTabla validation vs test (ordenada por ranking de validation):")
    print(comparison[columns].to_string(index=False))
    if not ranking.empty:
        print("\nPredictividad del ranking por ticker/timeframe:")
        print(ranking.to_string(index=False))
