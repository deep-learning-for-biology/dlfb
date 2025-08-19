from functools import partial
from typing import Literal

import jax
from jax import numpy as jnp

from dlfb.metrics import (
  calculate_false_negatives,
  calculate_true_positives,
)


@partial(jax.jit, static_argnums=(2, 3))
def recall_score(
  y_true: jnp.ndarray,
  y_preds: jnp.ndarray,
  n_labels: jnp.ndarray,
  average: Literal["macro", "micro", "weighted"],
) -> jnp.ndarray:
  """Computes recall score with macro, micro, or weighted averaging."""
  labels = jnp.arange(n_labels)
  recalls = jax.vmap(
    lambda lb: calculate_recall_per_class(y_true, y_preds, lb)
  )(labels)

  match average:
    case "macro":
      recall = jnp.mean(recalls)
    case "micro":
      tp = jnp.sum(y_preds == y_true)
      fn = jnp.sum(y_preds != y_true)
      recall = tp / (tp + fn + 1e-8)  # Avoid division by zero.
    case "weighted":
      support = jax.vmap(lambda lb: jnp.sum(y_true == lb))(labels)
      recall = jnp.sum(recalls * (support / (jnp.sum(support) + 1e-8)))
    case _:
      raise ValueError(f"Unsupported average type: {average}")

  return recall


@jax.jit
def calculate_recall_per_class(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, label: int
) -> jnp.ndarray:
  """Computes recall for a single class."""
  tp = calculate_true_positives(y_true, y_pred, label)
  fn = calculate_false_negatives(y_true, y_pred, label)
  return tp / (tp + fn + 1e-8)  # Avoid division by zero.
