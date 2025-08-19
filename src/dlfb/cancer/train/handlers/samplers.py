from typing import Iterator

import jax
import jax.numpy as jnp
import pandas as pd


def epoch_sampler(
  metadata: pd.DataFrame, batch_size: int, rng: jax.Array
) -> Iterator[jax.Array]:
  """Yields batches of indices from metadata sequentially."""
  frame_ids = metadata["frame_id"].to_numpy()
  rng, rng_shuffle = jax.random.split(rng, 2)
  shuffled_frame_ids = jax.random.permutation(rng_shuffle, frame_ids)
  for i in range(0, len(shuffled_frame_ids), batch_size):
    yield shuffled_frame_ids[i : i + batch_size]


def repeating_sampler(
  metadata: pd.DataFrame, batch_size: int, rng: jax.Array
) -> Iterator[jax.Array]:
  """Continuously generates random batches of data by sampling without
  replacement.

  Unlike `default_sampler`, this generator does not iterate through the
  dataset once, instead, it samples batches indefinitely,
  """
  frame_ids = metadata["frame_id"].to_numpy()

  while True:
    rng, rng_select = jax.random.split(rng, 2)
    batch_indices = jax.random.choice(
      rng_select, a=frame_ids, shape=(batch_size,), replace=False
    )

    yield batch_indices


def balanced_sampler(
  metadata: pd.DataFrame, batch_size: int, rng: jax.Array
) -> Iterator[jax.Array]:
  """Yields balanced batches by sampling equal number of instances per class."""
  labels = metadata["label"].unique()
  samples_per_class = (batch_size // len(labels)) + 1

  while True:
    batch_indices: list[jax.Array] = []
    for label in labels:
      rng, rng_select = jax.random.split(rng, 2)
      frame_ids = metadata.loc[
        metadata["label"] == label, "frame_id"
      ].to_numpy()
      sampled_frame_ids = jax.random.choice(
        rng_select, a=frame_ids, shape=(samples_per_class,), replace=False
      )
      batch_indices.extend(sampled_frame_ids)

    rng, rng_shuffle = jax.random.split(rng, 2)
    shuffled_batch_indices = jax.random.permutation(
      rng_shuffle, jnp.array(batch_indices)
    )
    yield shuffled_batch_indices[:batch_size]
