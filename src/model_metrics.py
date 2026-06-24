import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

# Clasifica el modelo basándose en si es el general o un especialista en este activo
def model_scope(model_name):
    return "generalist" if model_name.endswith("_generalist") else "specific"

# Previene el desajuste de nombres
def fitted_feature_names(pipeline, feature_cols):
    imputer = pipeline.named_steps.get("imputer")
    if imputer is None or not hasattr(imputer, "get_feature_names_out"):
        return list(feature_cols)
    return list(imputer.get_feature_names_out(feature_cols))

# Genera un df de importancias
def extract_feature_importance(models, feature_cols):
    rows = []
    for model_name, pipeline in models.items():
        estimator = pipeline.named_steps["model"]
        model_feature_cols = fitted_feature_names(pipeline, feature_cols)
        # Modelos lineales
        if hasattr(estimator, "coef_"):
            raw_values = np.asarray(estimator.coef_)[0]
            values = np.abs(raw_values)
            method = "absolute_coefficient"
        # Árboles
        elif hasattr(estimator, "feature_importances_"):
            raw_values = np.asarray(estimator.feature_importances_)
            values = np.clip(raw_values, 0, None)
            method = "native_feature_importance"
        else:
            # Modelos que no muestren importancias
            continue

        # Aseguramos que las importancias sumen 1
        total = values.sum()
        normalized = values / total if total > 0 else np.zeros_like(values)

        model_rows = pd.DataFrame({
            "model": model_name,
            "feature": model_feature_cols,
            "raw_importance": raw_values,
            "importance": normalized,
            "importance_method": method,
        }).sort_values("importance", ascending=False)
        model_rows["importance_rank"] = np.arange(1, len(model_rows) + 1)
        rows.append(model_rows)

    columns = [
        "model", "feature", "raw_importance", "importance",
        "importance_method", "importance_rank",
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)

# Calcula las métricas de clasificación 
def compute_classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    has_two_classes = np.unique(y_true).size == 2
    return {
        "n_obs": len(y_true),
        "threshold": float(threshold),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if has_two_classes else np.nan,
        "average_precision": (
            average_precision_score(y_true, probabilities)
            if has_two_classes else np.nan
        ),
        "log_loss": (
            log_loss(y_true, probabilities, labels=[0, 1])
            if has_two_classes else np.nan
        ),
        "brier_score": brier_score_loss(y_true, probabilities),
        "matthews_corrcoef": matthews_corrcoef(y_true, predictions),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# Evalúa las predicciones usando los umbrales calculados previamente (sin reajustarlos)
def evaluate_classification(
    datasets,
    probabilities,
    selection,
    target_col,
    splits=("validation", "test"),
):
    rows = []
    thresholds = selection.set_index("model")["threshold"].to_dict()
    for model_name, threshold in thresholds.items():
        for split_name in splits:
            data = datasets[split_name]
            metrics = compute_classification_metrics(
                data[target_col], probabilities[split_name][model_name], threshold
            )
            rows.append({
                "ticker": data["ticker"].iloc[0],
                "timeframe": data["timeframe"].iloc[0],
                "model": model_name,
                "model_scope": model_scope(model_name),
                "split": split_name,
                **metrics,
            })
    return rows
