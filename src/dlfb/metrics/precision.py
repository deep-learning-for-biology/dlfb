from functools import partial
from typing import Literal

import jax
from jax import numpy as jnp

from dlfb.metrics import (
  calculate_false_positives,
  calculate_true_positives,
)


@partial(jax.jit, static_argnums=(2, 3))
def precision_score(
  y_true: jnp.ndarray,
  y_preds: jnp.ndarray,
  n_labels: jnp.ndarray,
  average: Literal["macro", "micro", "weighted"],
) -> jnp.ndarray:
  """Computes precision score with macro, micro, or weighted averaging."""
  labels = jnp.arange(n_labels)
  precisions = jax.vmap(
    lambda lb: calculate_precision_per_class(y_true, y_preds, lb)
  )(labels)

  match average:
    case "macro":
      precision = jnp.mean(precisions)
    case "micro":
      tp = jnp.sum(y_preds == y_true)
      fp = jnp.sum(y_preds != y_true)
      precision = tp / (tp + fp + 1e-8)  # Avoid division by zero.
    case "weighted":
      support = jax.vmap(lambda lb: jnp.sum(y_true == lb))(labels)
      precision = jnp.sum(precisions * (support / (jnp.sum(support) + 1e-8)))
    case _:
      raise ValueError(f"Unsupported average type: {average}")

  return precision


@jax.jit
def calculate_precision_per_class(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, label: int
) -> jnp.ndarray:
  """Computes precision for a single class."""
  tp = calculate_true_positives(y_true, y_pred, label)
  fp = calculate_false_positives(y_true, y_pred, label)
  return tp / (tp + fp + 1e-8)  # Avoid division by zero.
