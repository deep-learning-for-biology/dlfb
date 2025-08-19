import jax
from flax import linen as nn

from dlfb.cancer.model import SkinLesionClassifierHead, TrainStateWithBatchNorm


class SimpleCnn(nn.Module):
  """Simple CNN model with small convolutional backbone and classifier head."""

  num_classes: int
  dropout_rate: float = 0.0

  def setup(self):
    """Initializes the CNN backbone and classification head."""
    self.backbone = CnnBackbone()
    self.head = SkinLesionClassifierHead(self.num_classes, self.dropout_rate)

  @nn.compact
  def __call__(self, x, is_training: bool = False):
    """Applies the backbone and classifier head to the input."""
    x = self.backbone(x)
    x = self.head(x, is_training=is_training)
    return x

  def create_train_state(
    self, rng: jax.Array, dummy_input, tx
  ) -> TrainStateWithBatchNorm:
    """Creates the training state with initialized parameters."""
    rng, rng_init, rng_dropout = jax.random.split(rng, 3)
    variables = self.init(rng_init, dummy_input)
    state = TrainStateWithBatchNorm.create(
      apply_fn=self.apply,
      tx=tx,
      params=variables["params"],
      batch_stats=None,
      key=rng_dropout,
    )
    return state


class CnnBackbone(nn.Module):
  """Compact convolutional feature extractor for image data."""

  @nn.compact
  def __call__(self, x):
    """Applies two conv-pool blocks and a dense layer to the input."""
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    return x
