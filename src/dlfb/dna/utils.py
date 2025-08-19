import jax
import jax.numpy as jnp
import numpy as np


def dna_to_one_hot(dna_sequence: str) -> np.ndarray:
  """Convert DNA into a one-hot encoded format with channel ordering ACGT."""
  base_to_one_hot = {
    "A": (1, 0, 0, 0),
    "C": (0, 1, 0, 0),
    "G": (0, 0, 1, 0),
    "T": (0, 0, 0, 1),
    "N": (1, 1, 1, 1),  # N represents any unknown or ambiguous base.
  }
  one_hot_encoded = np.array([base_to_one_hot[base] for base in dna_sequence])
  return one_hot_encoded


def one_hot_to_dna(one_hot_encoded: np.ndarray) -> str:
  """Convert one-hot encoded array back to DNA sequence."""
  one_hot_to_base = {
    (1, 0, 0, 0): "A",
    (0, 1, 0, 0): "C",
    (0, 0, 1, 0): "G",
    (0, 0, 0, 1): "T",
    (1, 1, 1, 1): "N",  # N represents any unknown or ambiguous base.
  }

  dna_sequence = "".join(
    one_hot_to_base[tuple(base)] for base in one_hot_encoded
  )
  return dna_sequence


@jax.jit
def compute_input_gradient(state, sequence):
  """Compute input gradient for a one-hot DNA sequence."""
  if len(sequence.shape) != 2:
    raise ValueError("Input must be a single one-hot encoded DNA sequence.")

  sequence = jnp.asarray(sequence, dtype=jnp.float32)[None, :]

  def predict(sequence):
    # We take the mean to ensure we have a single scalar to take the grad of.
    return jnp.mean(state.apply_fn({"params": state.params}, sequence))

  gradient = jax.grad(lambda x: predict(x))(sequence)
  return jnp.squeeze(gradient)


def filter_sequences_by_label(dataset, target_label, max_count):
  """Filter up to N sequences matching a target label."""
  sequences = []
  for i, label in enumerate(dataset["labels"]):
    if label == target_label:
      sequences.append(dataset["sequences"][i])
      if len(sequences) == max_count:
        break
  return sequences
