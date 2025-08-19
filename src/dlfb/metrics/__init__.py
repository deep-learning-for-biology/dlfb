import jax
from jax import numpy as jnp


@jax.jit
def accuracy_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """Compute accuracy as the proportion of correct predictions."""
  return jnp.mean(jnp.equal(y_true, y_pred))


@jax.jit
def calculate_true_positives(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, label: int
) -> jnp.ndarray:
  """Count true positives for a given class label."""
  return jnp.sum((y_pred == label) & (y_true == label))


@jax.jit
def calculate_false_positives(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, label: int
) -> jnp.ndarray:
  """Count false positives for a given class label."""
  return jnp.sum((y_pred == label) & (y_true != label))


@jax.jit
def calculate_false_negatives(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, label: int
) -> jnp.ndarray:
  """Count false negatives for a given class label."""
  return jnp.sum((y_pred != label) & (y_true == label))


@jax.jit
def calculate_true_negatives(
  y_true: jnp.ndarray, y_pred: jnp.ndarray, label: int
) -> jnp.ndarray:
  """Count true negatives by excluding the given class label."""
  return jnp.sum((y_pred != label) & (y_true != label))
