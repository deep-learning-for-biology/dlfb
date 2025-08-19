import jax
from flax import linen as nn
from flax.training import train_state
from transformers import FlaxResNetModel

# Dictionary of pretrained ResNet models from Hugging Face.
PRETRAINED_RESNETS = {
  18: FlaxResNetModel.from_pretrained("microsoft/resnet-18"),
  34: FlaxResNetModel.from_pretrained("microsoft/resnet-34"),
  50: FlaxResNetModel.from_pretrained("microsoft/resnet-50"),
}


class TrainStateWithBatchNorm(train_state.TrainState):
  """Train state that tracks batch statistics and a PRNG key."""

  batch_stats: dict | None
  key: jax.Array


class SkinLesionClassifierHead(nn.Module):
  """Skin lesion classification MLP head."""

  num_classes: int
  dropout_rate: float

  @nn.compact
  def __call__(self, x, is_training: bool):
    x = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform())(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
    x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform())(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
    x = nn.Dense(
      self.num_classes, kernel_init=nn.initializers.xavier_uniform()
    )(x)
    return x
