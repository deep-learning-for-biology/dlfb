from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


@dataclass
class Images:
  """Stores image data and size information."""

  loaded: dict[str, np.memmap]
  size: tuple[int, int, int]


@dataclass
class Dataset:
  """Dataset class storing images and corresponding metadata."""

  metadata: pd.DataFrame
  images: Images
  num_classes: int

  def get_dummy_input(self) -> jax.Array:
    """Returns dummy input with the correct shape for the model."""
    return jnp.empty((1,) + self.images.size)

  def num_samples(self) -> int:
    """Returns the number of samples in the dataset."""
    return self.metadata.shape[0]

  def get_images(self, preprocessor: Callable, indices: jax.Array) -> jax.Array:
    """Returns preprocessed images for the given indices."""
    return self.images.loaded[preprocessor.__name__][indices]

  def get_labels(self, indices: list[int]) -> jax.Array:
    """Returns integer class labels for the given indices."""
    return jnp.int16(self.metadata.loc[indices]["label"].values)
