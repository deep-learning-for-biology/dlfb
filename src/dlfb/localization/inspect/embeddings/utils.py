import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp
from scipy.cluster.hierarchy import fcluster, linkage

from dlfb.localization.dataset import Dataset
from dlfb.localization.model import (
  LocalizationModel,
  TrainState,
  get_num_embeddings,
)


def get_frame_embeddings(
  state: TrainState,
  dataset_split: Dataset,
  batch_size: int = 256,
) -> dict[str, np.ndarray]:
  """Returns per-frame histograms of codebook encoding indices."""
  num_embeddings = get_num_embeddings(state)
  frame_ids, frame_histograms = [], []

  rng = jax.random.PRNGKey(42)
  for batch in dataset_split.get_batches(rng, batch_size):
    frame_ids.append(batch["frame_ids"])
    encoding_indices = pluck_encodings(state, batch)

    # Reshape and count codebook usage per frame.
    frame_histograms.append(
      np.apply_along_axis(
        lambda x: np.histogram(x, bins=np.arange(0, num_embeddings + 0.5))[0],
        axis=1,
        arr=jnp.reshape(encoding_indices, (batch_size, -1)),
      )
    )

  return {
    "frame_ids": np.concatenate(frame_ids),
    "frame_histograms": np.concatenate(frame_histograms, axis=0),
  }


@jax.jit
def pluck_encodings(state, batch: dict[str, jax.Array]) -> jax.Array:
  """Returns encoding indices for a batch of images."""
  encoding_indices = state.apply_fn(
    {"params": state.params},
    batch["images"],
    method=LocalizationModel.get_encoding_indices,
  )
  return encoding_indices


def aggregate_proteins(dataset: Dataset, frame_ids, frame_histograms):
  protein_histograms = []
  protein_ids = []
  lookup = (
    dataset.labels.frame_protein_lookup().set_index("frame_id").loc[frame_ids]
  )
  lookup["idx"] = np.arange(len(frame_ids))
  for protein_id, rows in lookup.groupby("protein_id"):
    protein_ids.append(protein_id)
    protein_histograms.append(
      frame_histograms[rows["idx"].tolist(), :].mean(axis=0)
    )
  protein_histograms = np.stack(protein_histograms, axis=0)
  return protein_ids, protein_histograms


def aggregate_localizations(dataset: Dataset, protein_ids, protein_histograms):
  proteins = dataset.filter_for_single_localization_frames(
    pd.DataFrame({"protein_id": protein_ids, "idx": range(len(protein_ids))})
  )
  proteins = proteins.merge(
    dataset.labels.localization_lookup(), how="inner", on="protein_id"
  )
  proteins = dataset.filter_for_localization_of_interest(proteins)
  localizations = []
  localization_histograms = []
  for localization, rows in proteins[["localization", "idx"]].groupby(
    "localization"
  ):
    localizations.append(localization)
    localization_histograms.append(
      protein_histograms[rows["idx"], :].mean(axis=0)
    )
  localization_histograms = np.stack(localization_histograms, axis=0)
  return localizations, localization_histograms


def np_pearson_cor(x, y):
  # NOTE: see -- https://cancerdatascience.org/blog/posts/pearson-correlation/
  xv = x - x.mean(axis=0)
  yv = y - y.mean(axis=0)
  xvss = (xv * xv).sum(axis=0)
  yvss = (yv * yv).sum(axis=0)
  result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
  # NOTE: bound the values to -1 to 1 in the event of precision issues
  return np.nan_to_num(np.maximum(np.minimum(result, 1.0), -1.0))


def np_euclidian_distance(x, y):
  # NOTE: see -- https://jaykmody.com/blog/distance-matrices-with-numpy/
  x2 = np.sum(x**2, axis=1)
  y2 = np.sum(y**2, axis=1)
  xy = np.matmul(x, y.T)
  x2 = x2.reshape(-1, 1)
  # NOTE: bound the values to 0 in the event of precision issues
  return np.nan_to_num(np.sqrt(np.maximum(x2 - 2 * xy + y2, 0.0)))


def cluster_feature_spectrums(
  protein_histograms: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Cluster proteins based on similarity in codebook usage patterns."""
  corr_idx_idx = np_pearson_cor(protein_histograms, protein_histograms)
  tree = linkage(
    corr_idx_idx,
    method="average",
    metric="euclidean",
    optimal_ordering=True,
  )
  encoding_clusters = fcluster(tree, n_clusters, criterion="maxclust")
  return corr_idx_idx, tree, encoding_clusters
