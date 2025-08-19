from functools import partial
from typing import Any, Callable

import jax
import optax
import pandas as pd
from jax import numpy as jnp
from tqdm import tqdm

from dlfb.cancer.dataset import Dataset
from dlfb.cancer.dataset.preprocessors import crop
from dlfb.cancer.model import TrainStateWithBatchNorm
from dlfb.cancer.train.handlers import BatchHandler
from dlfb.cancer.train.handlers.samplers import repeating_sampler
from dlfb.metrics.precision import precision_score
from dlfb.metrics.recall import recall_score
from dlfb.utils.metrics_logger import MetricsLogger
from dlfb.utils.restore import restorable


@restorable
def train(
  state: TrainStateWithBatchNorm,
  rng: jax.Array,
  dataset_splits: dict[str, Dataset],
  num_steps: int,
  batch_size: int,
  preprocessor: Callable = crop,
  sampler: Callable = repeating_sampler,
  augmentor: Callable = None,
  eval_every: int = 10,
) -> tuple[TrainStateWithBatchNorm, dict]:
  """Trains a model using the provided dataset splits and logs metrics."""
  # Setup with metrics logger and classes numbers.
  num_classes = dataset_splits["train"].num_classes
  metrics = MetricsLogger()

  # Get train batch iterator from which to take batches.
  rng, rng_train, rng_eval = jax.random.split(rng, 3)
  train_batcher = BatchHandler(preprocessor, sampler, augmentor)
  train_batches = train_batcher.get_batches(
    dataset_splits["train"], batch_size, rng_train
  )

  steps = tqdm(range(num_steps))  # Steps with progress bar.
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    rng, rng_dropout = jax.random.split(rng, 2)
    train_batch = next(train_batches)
    state, batch_metrics = train_step(
      state, train_batch, rng_dropout, num_classes
    )
    metrics.log_step(split="train", **batch_metrics)

    if step % eval_every == 0:
      for batch in BatchHandler(preprocessor).get_batches(
        dataset_splits["valid"], batch_size, rng_eval
      ):
        batch_metrics = eval_step(state, batch, num_classes)
        metrics.log_step(split="valid", **batch_metrics)
      metrics.flush(step=step)

    steps.set_postfix_str(metrics.latest(["loss"]))

  return state, metrics.export()


@partial(jax.jit, static_argnums=(3,))
def train_step(
  state: TrainStateWithBatchNorm,
  batch: dict[str, list[Any]],
  rng_dropout: jax.Array,
  num_classes: int,
) -> tuple[TrainStateWithBatchNorm, dict[str, jax.Array]]:
  """Performs a single training step and returns updated state and metrics."""

  def calculate_loss(params, images, labels):
    variables, kwargs = {"params": params}, {"mutable": []}
    if state.batch_stats is not None:
      variables.update({"batch_stats": state.batch_stats})
      kwargs.update({"mutable": ["batch_stats"]})
    logits, updates = state.apply_fn(
      variables,
      x=images,
      is_training=True,
      rngs={"dropout": rng_dropout},
      **kwargs,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits, labels
    ).mean()
    return loss, (logits, updates)

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
  (loss, (logits, updates)), grads = grad_fn(
    state.params, batch["images"], batch["labels"]
  )
  state = state.apply_gradients(grads=grads)
  if state.batch_stats is not None:
    state = state.replace(batch_stats=updates["batch_stats"])

  metrics = {
    "loss": loss,
    **compute_metrics(batch["labels"], logits, num_classes),
  }

  return state, metrics


@partial(jax.jit, static_argnums=(2,))
def eval_step(
  state: TrainStateWithBatchNorm, batch: dict[str, Any], num_classes: int
):
  """Evaluates model performance on a batch and computes metrics."""
  variables, kwargs = {"params": state.params}, {}
  if state.batch_stats is not None:
    variables.update({"batch_stats": state.batch_stats})
    kwargs.update({"mutable": False})

  logits = state.apply_fn(
    variables, x=batch["images"], is_training=False, **kwargs
  )
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits, batch["labels"]
  ).mean()

  metrics = {
    "loss": loss,
    **compute_metrics(batch["labels"], logits, num_classes),
  }
  return metrics


@partial(jax.jit, static_argnums=(2,))
def compute_metrics(
  y_true: jax.Array, logits: jax.Array, n_labels: jax.Array
) -> dict[str, jax.Array]:
  """Computes weighted precision and recall metrics from logits and labels."""
  y_scores = jax.nn.softmax(logits)
  y_pred = jnp.argmax(y_scores, axis=1)
  metrics = {
    "recall_weighted": recall_score(
      y_true, y_pred, n_labels, average="weighted"
    ),
    "precision_weighted": precision_score(
      y_true, y_pred, n_labels, average="weighted"
    ),
  }
  return metrics


def get_predictions(
  state: TrainStateWithBatchNorm,
  dataset: Dataset,
  preprocessor: Callable,
  batch_size: int = 32,
) -> pd.DataFrame:
  """Generates predictions for entire dataset using the current model state."""
  dfs = []
  for batch in BatchHandler(preprocessor).get_batches(
    dataset, batch_size, jax.random.PRNGKey(42)
  ):
    pred = predict(state, batch["images"]).tolist()
    dfs.append(pd.DataFrame({"frame_id": batch["frame_ids"], "pred": pred}))
  predictions = pd.concat(dfs).merge(
    dataset.metadata[["frame_id", "class", "label"]], on="frame_id"
  )
  return predictions


@jax.jit
def predict(state, images):
  """Returns predicted class labels from model logits."""
  # NOTE: because of jit we cannot return a Python list.
  variables, kwargs = {"params": state.params}, {}
  if state.batch_stats is not None:
    variables.update({"batch_stats": state.batch_stats})
    kwargs.update({"mutable": False})
  logits = state.apply_fn(variables, images, **kwargs)
  return jnp.argmax(logits, axis=-1)
