from dataclasses import dataclass
from typing import Generator

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Pairs:
  """Represents positive and negative pairs of drug-drug interactions."""

  pos: jax.Array
  neg: jax.Array

  def get_eval_batches(
    self, batch_size: int
  ) -> Generator[dict[str, jax.Array], None, None]:
    """Generates evaluation batches of positive and negative pairs."""
    indices = jnp.arange(self._n_pairs())
    for i in range(self._n_batches(batch_size)):
      batch_indices = jnp.array(indices[i * batch_size : (i + 1) * batch_size])
      yield Pairs(
        pos=self.pos[batch_indices], neg=self.neg[batch_indices]
      ).to_dict()

  def _n_batches(self, batch_size: int) -> int:
    """Calculates number of batches in the dataset given a batch size."""
    return int(np.floor(self._n_pairs() / batch_size))

  def _n_pairs(self) -> int:
    """Returns the smaller number of positive or negative pairs."""
    return int(min(self.pos.shape[0], self.neg.shape[0]))

  def get_train_batches(
    self, batch_size: int, rng_shuffle: jax.Array, rng_sample: jax.Array
  ) -> Generator[dict[str, jax.Array], None, None]:
    """Generates shuffled training batches with sampled negative pairs."""
    # Shuffle indices for positive pairs.
    indices = jax.random.permutation(rng_shuffle, jnp.arange(self._n_pairs()))

    # Get sample of negative pairs.
    neg_sample = self._global_negative_sampling(rng_sample)

    for i in range(self._n_batches(batch_size)):
      batch_indices = jnp.array(indices[i * batch_size : (i + 1) * batch_size])
      yield Pairs(
        pos=self.pos[batch_indices], neg=neg_sample[batch_indices]
      ).to_dict()

  def _global_negative_sampling(self, rng_sample: jax.Array) -> jax.Array:
    """Samples negative pairs from the entire set to match positive set size."""
    return jax.random.choice(
      rng_sample, self.neg, (self.pos.shape[0],), replace=False
    )

  def get_dummy_input(self) -> dict[str, jax.Array]:
    """Returns a small dummy subset of positive and negative pairs."""
    return Pairs(pos=self.pos[:2], neg=self.neg[:(2)]).to_dict()

  def to_dict(self) -> dict:
    """Converts the Pairs object back to a dictionary."""
    return {"pos": self.pos, "neg": self.neg}
