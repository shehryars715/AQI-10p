"""Model evaluation — RMSE, MAE, R² per prediction horizon and averaged."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    horizon_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate a trained model and return a dict of metrics.

    For multi-output models, metrics are computed per horizon and averaged.

    Returns dict with keys like:
        rmse_day1, rmse_day2, rmse_day3, avg_rmse,
        mae_day1, ..., avg_mae, r2_day1, ..., avg_r2
    """
    y_pred = model.predict(X_test)

    # Ensure 2D arrays
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    n_outputs = y_test.shape[1]
    if horizon_labels is None:
        horizon_labels = [f"day{i + 1}" for i in range(n_outputs)]

    metrics: Dict[str, float] = {}

    for i in range(n_outputs):
        yt = y_test[:, i]
        yp = y_pred[:, i]
        label = horizon_labels[i]

        metrics[f"rmse_{label}"] = float(np.sqrt(mean_squared_error(yt, yp)))
        metrics[f"mae_{label}"] = float(mean_absolute_error(yt, yp))
        metrics[f"r2_{label}"] = float(r2_score(yt, yp))

    # Averages
    metrics["avg_rmse"] = float(np.mean([metrics[f"rmse_{l}"] for l in horizon_labels]))
    metrics["avg_mae"] = float(np.mean([metrics[f"mae_{l}"] for l in horizon_labels]))
    metrics["avg_r2"] = float(np.mean([metrics[f"r2_{l}"] for l in horizon_labels]))

    logger.info(
        "Evaluation: avg_rmse=%.2f, avg_mae=%.2f, avg_r2=%.4f",
        metrics["avg_rmse"],
        metrics["avg_mae"],
        metrics["avg_r2"],
    )
    return metrics


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "avg_rmse",
    lower_is_better: bool = True,
) -> str:
    """Pick the best model name from a dict of {name: metrics_dict}.

    Returns the name of the best model.
    """
    if not results:
        raise ValueError("No model results to compare")

    # Filter out entries where the metric is missing
    valid = {k: v for k, v in results.items() if metric in v}
    if not valid:
        raise ValueError(f"No model has metric '{metric}'")

    if lower_is_better:
        best = min(valid, key=lambda k: valid[k][metric])
    else:
        best = max(valid, key=lambda k: valid[k][metric])

    logger.info("Best model: '%s' with %s=%.4f", best, metric, valid[best][metric])
    return best


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    horizon_labels: Optional[List[str]] = None,
) -> str:
    """Return a pretty-printed ASCII table of all model metrics."""
    if horizon_labels is None:
        horizon_labels = ["day1", "day2", "day3"]

    header = (
        f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}  "
        + "  ".join(f"{l:>10}" for l in horizon_labels)
    )
    lines = [header, "-" * len(header)]

    for name, m in results.items():
        rmse_str = f"{m.get('avg_rmse', float('nan')):>8.2f}"
        mae_str = f"{m.get('avg_mae', float('nan')):>8.2f}"
        r2_str = f"{m.get('avg_r2', float('nan')):>8.4f}"
        per_h_str = "  ".join(
            f"{m.get(f'rmse_{l}', float('nan')):>10.2f}" for l in horizon_labels
        )
        lines.append(f"{name:<20} {rmse_str} {mae_str} {r2_str}  {per_h_str}")

    return "\n".join(lines)
