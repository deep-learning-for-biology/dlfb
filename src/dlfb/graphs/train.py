from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from tqdm import tqdm

from dlfb.graphs.dataset import Dataset
from dlfb.graphs.model import TrainState
from dlfb.utils.metrics_logger import MetricsLogger
from dlfb.utils.restore import restorable


@restorable
def train(
  state: TrainState,
  rng: jax.Array,
  dataset_splits: dict[str, Dataset],
  num_epochs: int,
  loss_fn: Callable,
  norm_loss: bool = False,
  eval_every: int = 10,
) -> tuple[TrainState, dict[str, dict[str, list[dict[str, float]]]]]:
  """Training loop for the drug-drug interaction model."""
  # Initialize metrics and estimate optimal batch sizes.
  metrics = MetricsLogger()
  batch_size = optimal_batch_size(dataset_splits)

  # Epochs with progress bar.
  epochs = tqdm(range(num_epochs))
  for epoch in epochs:
    epochs.set_description(f"Epoch {epoch + 1}")
    rng, rng_shuffle, rng_sample = jax.random.split(rng, 3)

    # Training loop.
    for pairs_batch in dataset_splits["train"].pairs.get_train_batches(
      batch_size, rng_shuffle, rng_sample
    ):
      rng, rng_dropout = jax.random.split(rng, 2)
      state, batch_metrics = train_step(
        state,
        dataset_splits["train"].graph,
        pairs_batch,
        rng_dropout,
        loss_fn,
        norm_loss,
      )
      metrics.log_step(split="train", **batch_metrics)

    # Evaluation loop.
    if epoch % eval_every == 0:
      for pairs_batch in dataset_splits["valid"].pairs.get_eval_batches(
        batch_size
      ):
        batch_metrics = eval_step(
          state, dataset_splits["valid"].graph, pairs_batch, loss_fn, norm_loss
        )
        metrics.log_step(split="valid", **batch_metrics)

    metrics.flush(epoch=epoch)
    epochs.set_postfix_str(metrics.latest(["hits@20"]))

  return state, metrics.export()


def optimal_batch_size(
  dataset_splits: dict[str, Dataset], remainder_tolerance: float = 0.125
) -> int:
  """Calculates optimal batch size for optimizing JAX compilation."""
  # Calculate the minimum length of positive and negative pairs for each
  # dataset.
  lengths = [
    min(dataset.pairs.pos.shape[0], dataset.pairs.neg.shape[0])
    for dataset in dataset_splits.values()
  ]

  # Determine the allowable remainders per split based on the remainder
  # tolerance.
  remainder_thresholds = [
    int(length * remainder_tolerance) for length in lengths
  ]
  max_possible_batch_size = min(lengths)

  for batch_size in range(max_possible_batch_size, 0, -1):
    remainders = [length % batch_size for length in lengths]
    if all(
      remainder <= threshold
      for remainder, threshold in zip(remainders, remainder_thresholds)
    ):
      return batch_size
  return max_possible_batch_size


@jax.jit
def binary_log_loss(scores: dict[str, jax.Array]) -> jax.Array:
  """Computes the binary log loss for positive and negative drug pairs."""
  # Clip probabilities to avoid numerical instability.
  probs = jax.tree.map(
    lambda x: jnp.clip(nn.sigmoid(x), 1e-7, 1 - 1e-7), scores
  )

  # Compute positive and negative losses.
  pos_loss = -jnp.log(probs["pos"]).mean()
  neg_loss = -jnp.log(1 - probs["neg"]).mean()

  return pos_loss + neg_loss


@jax.jit
def auc_loss(scores: dict[str, jax.Array]) -> jax.Array:
  """Computes AUC-based loss for positive and negative drug pairs."""
  return jnp.square(1 - (scores["pos"] - scores["neg"])).sum()


@partial(jax.jit, static_argnames=["loss_fn", "norm_loss"])
def train_step(
  state: TrainState,
  graph: jraph.GraphsTuple,
  pairs: dict[str, jax.Array],
  rng_dropout: jax.Array,
  loss_fn: Callable = binary_log_loss,
  norm_loss: bool = False,
) -> tuple[TrainState, dict[str, jax.Array]]:
  """Performs a single training step, updating model parameters."""

  def calculate_loss(params):
    """Computes loss and hits@20 metric for the given model parameters."""
    scores = state.apply_fn(
      {"params": params},
      graph,
      pairs,
      is_training=True,
      rngs={"dropout": rng_dropout},
    )
    loss = loss_fn(scores)
    metric = evaluate_hits_at_20(scores)
    return loss, metric

  # Note that calculate_loss is defined as a scoped function to simplify access
  # to additional variables (e.g., state, graph, pairs) without requiring them
  # to be explicitly passed, while maintaining compatibility with
  # jax.value_and_grad.
  grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
  (loss, metric), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)

  metrics = {"loss": loss, "hits@20": metric}
  if norm_loss:
    metrics["loss"] = metrics["loss"] / (
      pairs["pos"].shape[0] + pairs["neg"].shape[0]
    )

  return state, metrics


@partial(jax.jit, static_argnames=["loss_fn", "norm_loss"])
def eval_step(
  state: TrainState,
  graph: jraph.GraphsTuple,
  pairs: dict[str, jax.Array],
  loss_fn: Callable = binary_log_loss,
  norm_loss: bool = False,
) -> dict[str, jax.Array]:
  """Performs an evaluation step, computing loss and hits@20 metric."""
  scores = state.apply_fn(
    {"params": state.params}, graph, pairs, is_training=False
  )
  metrics = {"loss": loss_fn(scores), "hits@20": evaluate_hits_at_20(scores)}
  if norm_loss:
    metrics["loss"] = metrics["loss"] / (
      pairs["pos"].shape[0] + pairs["neg"].shape[0]
    )

  return metrics


@jax.jit
def evaluate_hits_at_20(scores: dict[str, jax.Array]) -> jax.Array:
  """Computes the hits@20 metric capturing positive pairs ranking."""
  # Implementation inspired by the OGB benchmark: https://github.com/snap-stanford/ogb/blob/f631af76359c9687b2fe60905557bbb241916258/ogb/linkproppred/evaluate.py#L214
  # Find the 20th highest score among negative edges.
  kth_score_in_negative_edges = jnp.sort(scores["neg"])[-20]

  # Compute the proportion of positive scores greater than the threshold.
  return (
    jnp.sum(scores["pos"] > kth_score_in_negative_edges)
    / scores["pos"].shape[0]
  )
