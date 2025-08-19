import jax
import optax
from flax import linen as nn
from flax import traverse_util
from jax import numpy as jnp

from dlfb.cancer.model import (
  PRETRAINED_RESNETS,
  SkinLesionClassifierHead,
  TrainStateWithBatchNorm,
)


class ResNetFromScratch(nn.Module):
  """ResNet model initialized from scratch with a custom classification head."""

  num_classes: int
  layers: int = 50
  dropout_rate: float = 0.0

  def setup(self):
    """Initializes the backbone and classification head."""
    self.backbone = PRETRAINED_RESNETS[self.layers].module
    self.head = SkinLesionClassifierHead(self.num_classes, self.dropout_rate)

  def __call__(self, x, is_training: bool = False):
    """Runs a forward pass through the model."""
    x = self.backbone(x, deterministic=not is_training).pooler_output
    x = jnp.squeeze(x, axis=(2, 3))
    x = self.head(x, is_training=is_training)
    return x

  def create_train_state(
    self, rng: jax.Array, dummy_input, tx
  ) -> TrainStateWithBatchNorm:
    """Initializes model parameters and optimizer state."""
    rng, rng_init, rng_dropout = jax.random.split(rng, 3)
    variables = self.init(rng_init, dummy_input, is_training=False)
    variables = self.transfer_parameters(variables)
    tx = self.set_trainable_parameters(tx, variables)
    state = TrainStateWithBatchNorm.create(
      apply_fn=self.apply,
      tx=tx,
      params=variables["params"],
      batch_stats=variables["batch_stats"],
      key=rng_dropout,
    )
    return state

  def transfer_parameters(_, variables):
    """Returns variables unchanged (no transfer learning)."""
    return variables

  @staticmethod
  def set_trainable_parameters(tx, _):
    """Returns optimizer configuration with all parameters trainable."""
    return tx


class FinetunedResNet(ResNetFromScratch):
  """ResNet model with pretrained weights and full fine-tuning."""

  def transfer_parameters(self, variables):
    """Replaces model parameters with pretrained ResNet weights."""
    resnet_variables = PRETRAINED_RESNETS[self.layers].params
    variables["params"]["backbone"] = resnet_variables["params"]
    variables["batch_stats"]["backbone"] = resnet_variables["batch_stats"]
    return variables


class FinetunedHeadResNet(FinetunedResNet):
  """ResNet model with a frozen backbone and trainable classification head."""

  @staticmethod
  def set_trainable_parameters(tx, variables):
    """Freezes backbone parameters and trains only the classification head."""
    return optax.multi_transform(
      transforms={"trainable": tx, "frozen": optax.set_to_zero()},
      param_labels=traverse_util.path_aware_map(
        lambda path, _: "frozen" if "backbone" in path else "trainable",
        variables["params"],
      ),
    )


class PartiallyFinetunedResNet(FinetunedResNet):
  """ResNet model with selective fine-tuning of deeper layers."""

  @staticmethod
  def set_trainable_parameters(tx, variables):
    """Freezes early layers, fine-tunes other layers at variable LR."""

    def label_fn(path, _):
      joined = "/".join(path)
      if "backbone" in path:
        if "stages/3" in joined:
          return "reduced_lr"
        return "frozen"
      return "trainable"

    return optax.multi_transform(
      transforms={
        "trainable": tx,
        "reduced_lr": optax.adam(learning_rate=1e-5),
        "frozen": optax.set_to_zero(),
      },
      param_labels=traverse_util.path_aware_map(label_fn, variables["params"]),
    )
