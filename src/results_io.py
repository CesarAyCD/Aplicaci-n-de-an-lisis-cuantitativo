from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from result_reporting import build_validation_test_report, ranking_diagnostics

# Este código es sólo para crear un archivo temporal que guarda las curvas incrementalmente
# No es necesario en un sistema pequeño. El problema que tuve es que se llenaba la RAM de mi PC con los datos.

def write_curve_frames(curves, temp_path, writer=None):
    temp_path = Path(temp_path)
    for curve in curves:
        frame = curve.copy(deep=False)
        if "model_scope" not in frame:
            frame["model_scope"] = "baseline"
        frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
        table = pa.Table.from_pandas(frame, preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(temp_path, table.schema, compression="snappy")
        else:
            table = table.cast(writer.schema, safe=False)
        writer.write_table(table)
    return writer


def save_results(
    output_dir,
    curves_path,
    summary_rows,
    threshold_frames,
    selection,
    classification_rows,
    feature_importance_frames,
):
    output_dir, curves_path = Path(output_dir), Path(curves_path)
    summary = pd.DataFrame(summary_rows)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    classification = pd.DataFrame(classification_rows)
    feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
    paths = {
        "summary": output_dir / "backtest_summary.parquet",
        "curves": curves_path,
        "thresholds": output_dir / "backtest_threshold_validation.parquet",
        "selection": output_dir / "model_selection_validation.parquet",
        "comparison": output_dir / "validation_vs_test_model_report.parquet",
        "ranking": output_dir / "ranking_validation_vs_test.parquet",
        "classification": output_dir / "classification_metrics.parquet",
        "feature_importance": output_dir / "feature_importance.parquet",
    }
    summary.to_parquet(paths["summary"], index=False)
    thresholds.to_parquet(paths["thresholds"], index=False)
    selection.to_parquet(paths["selection"], index=False)
    comparison = build_validation_test_report(summary, selection)
    comparison.to_parquet(paths["comparison"], index=False)
    ranking = ranking_diagnostics(comparison)
    ranking.to_parquet(paths["ranking"], index=False)
    classification.to_parquet(paths["classification"], index=False)
    feature_importance.to_parquet(paths["feature_importance"], index=False)
    return summary, comparison, ranking, paths
