import jax
import optax
from tqdm.auto import tqdm

from dlfb.localization.dataset import Dataset
from dlfb.localization.model import TrainState
from dlfb.metrics import accuracy_score
from dlfb.utils.metrics_logger import MetricsLogger
from dlfb.utils.restore import restorable


@restorable
def train(
  state: TrainState,
  rng: jax.Array,
  dataset_splits: dict[str, Dataset],
  num_epochs: int,
  batch_size: int,
  classification_weight: float,
  eval_every: int = 10,
) -> tuple[TrainState, dict[str, dict[str, list[dict[str, float]]]]]:
  """Train the VQ-VAE model with optional classification."""
  # Setup metrics logging
  metrics = MetricsLogger()

  epochs = tqdm(range(num_epochs))
  for epoch in epochs:
    epochs.set_description(f"Epoch {epoch + 1}")
    rng, rng_batch = jax.random.split(rng, 2)

    # Perform a training step on a batch of train data and log metrics.
    for batch in dataset_splits["train"].get_batches(
      rng_batch, batch_size=batch_size
    ):
      rng, rng_dropout = jax.random.split(rng, 2)
      state, batch_metrics = train_step(
        state, batch, rng_dropout, classification_weight
      )
      metrics.log_step(split="train", **batch_metrics)

    # Evaluate on the validation split
    if epoch % eval_every == 0:
      rng, rng_batch = jax.random.split(rng, 2)
      for batch in dataset_splits["valid"].get_batches(
        rng_batch, batch_size=batch_size
      ):
        batch_metrics = eval_step(state, batch, classification_weight)
        metrics.log_step(split="valid", **batch_metrics)

    metrics.flush(epoch=epoch)
    epochs.set_postfix_str(metrics.latest(["total_loss"]))

  return state, metrics.export()


@jax.jit
def train_step(
  state: TrainState,
  batch: dict[str, jax.Array],
  rng_dropout: jax.Array,
  classification_weight: float,
) -> tuple[TrainState, dict[str, float]]:
  """Train for a single step."""

  def calculate_loss(params: dict) -> tuple[jax.Array, dict[str, float]]:
    """Forward pass and loss computation."""
    (
      x_recon,
      perplexity,
      codebook_loss,
      commitment_loss,
      logits,
    ) = state.apply_fn(
      {"params": params},
      batch["images"],
      is_training=True,
      rngs={"dropout": rng_dropout},
    )

    loss_components = {
      "recon_loss": optax.squared_error(
        predictions=x_recon, targets=batch["images"]
      ).mean(),
      "codebook_loss": codebook_loss,
      "commitment_loss": commitment_loss,
      "classification_loss": classification_weight
      * optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["labels"]
      ).mean(),
    }

    metrics = {
      "total_loss": sum_loss_components(**loss_components),
      "perplexity": perplexity,
      "accuracy": accuracy_score(batch["labels"], y_pred=logits.argmax(-1)),
      **loss_components,
    }
    return metrics["total_loss"], metrics

  # Compute gradients and apply update.
  grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
  (_, metrics), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, metrics


@jax.jit
def eval_step(state, batch, classification_weight):
  (
    x_recon,
    perplexity,
    codebook_loss,
    commitment_loss,
    logits,
  ) = state.apply_fn(
    {"params": state.params}, batch["images"], is_training=False
  )

  loss_components = {
    "recon_loss": optax.l2_loss(
      predictions=x_recon, targets=batch["images"]
    ).mean(),
    "codebook_loss": codebook_loss,
    "commitment_loss": commitment_loss,
    "classification_loss": classification_weight
    * optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch["labels"]
    ).mean(),
  }

  metrics = {
    "total_loss": sum_loss_components(**loss_components),
    "perplexity": perplexity,
    "accuracy": accuracy_score(batch["labels"], y_pred=logits.argmax(-1)),
    **loss_components,
  }

  return metrics


@jax.jit
def sum_loss_components(
  recon_loss, codebook_loss, commitment_loss, classification_loss
):
  total_loss = (
    recon_loss + codebook_loss + commitment_loss + classification_loss
  )
  return total_loss
