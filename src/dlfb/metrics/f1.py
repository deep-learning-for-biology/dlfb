from functools import partial
from typing import Literal

import jax
from jax import numpy as jnp

from dlfb.metrics.precision import calculate_precision_per_class
from dlfb.metrics.recall import calculate_recall_per_class


@partial(jax.jit, static_argnums=(2, 3))
def f1_score(
  y_true: jnp.ndarray,
  y_pred: jnp.ndarray,
  n_labels: jnp.ndarray,
  average: Literal["macro", "micro", "weighted"],
) -> jnp.ndarray:
  """Compute F1 score with macro, micro, or weighted averaging."""
  labels = jnp.arange(n_labels)
  f1_scores = jax.vmap(lambda lb: calculate_f1_per_class(y_true, y_pred, lb))(
    labels
  )

  match average:
    case "macro":
      f1 = jnp.mean(f1_scores)
    case "micro":
      tp = jnp.sum(y_pred == y_true)
      fn_fp = jnp.sum(y_pred != y_true)
      precision_micro = tp / (tp + fn_fp + 1e-8)
      recall_micro = tp / (tp + fn_fp + 1e-8)
      f1 = (
        2
        * (precision_micro * recall_micro)
        / (precision_micro + recall_micro + 1e-8)
      )
    case "weighted":
      support = jax.vmap(lambda lb: jnp.sum(y_true == lb))(labels)
      f1 = jnp.sum(f1_scores * (support / jnp.sum(support)))
    case _:
      raise ValueError(
        f"Unsupported average type '{average}'. Choose from 'macro', 'micro',"
        " or 'weighted'."
      )
  return f1


@jax.jit
def calculate_f1_per_class(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, class_label: int
) -> jnp.ndarray:
  """Calculate F1 score for a single class."""
  precision = calculate_precision_per_class(y_true, y_pred, class_label)
  recall = calculate_recall_per_class(y_true, y_pred, class_label)
  combined = precision + recall
  return jnp.where(
    combined > 0,
    # Avoids NaN when both precision and recall are zero.
    2 * (precision * recall) / (combined + 1e-8),
    jnp.array(0.0),
  )
