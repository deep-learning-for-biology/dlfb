from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dlfb.localization.constants import LOCALIZATIONS_OF_INTEREST
from dlfb.localization.utils import calculate_grid_dimensions


@dataclass
class Images:
  frames: np.ndarray

  def n(self) -> int:
    return self.frames.shape[0]


@dataclass
class Labels:
  lookup: pd.DataFrame

  def subset(self, frames) -> "Labels":
    return Labels(
      lookup=self.lookup.iloc[frames["frame_id"]].reset_index(drop=True)
    )

  def get_all_frames(self) -> pd.DataFrame:
    return self.lookup[["frame_id", "fov_id", "protein_id"]]

  def get_frames_of_random_proteins(
    self, rng_proteins: jax.Array, max_proteins: int
  ) -> pd.DataFrame:
    all_proteins = self.protein_lookup()
    sample_idx = jax.random.choice(
      rng_proteins, all_proteins.shape[0], (max_proteins,), replace=False
    )
    proteins = all_proteins.loc[sample_idx]
    protein_frames = proteins.merge(
      self.frame_protein_lookup(), how="left", on="protein_id"
    ).sort_values(["protein_id", "frame_id", "fov_id"])
    frame_ids = protein_frames[["frame_id", "fov_id", "protein_id"]]
    return frame_ids

  def get_frames_of_selected_proteins(
    self, gene_symbols: list[str]
  ) -> pd.DataFrame:
    lookup = self.protein_lookup()
    proteins = lookup[lookup["gene_symbol"].isin(gene_symbols)]
    protein_frames = proteins.merge(
      self.frame_protein_lookup(), how="left", on="protein_id"
    ).sort_values(["protein_id", "frame_id", "fov_id"])
    frame_ids = protein_frames[["frame_id", "fov_id", "protein_id"]]
    return frame_ids

  def get_frames_of_selected_localizations(self, localizations) -> pd.DataFrame:
    lookup = self.localization_lookup()
    localizations = lookup[lookup["localization"].isin(localizations)]
    localization_frames = localizations.merge(
      self.frame_protein_lookup(), how="left", on="protein_id"
    ).sort_values(["protein_id", "frame_id", "fov_id"])
    frame_ids = localization_frames[["frame_id", "fov_id", "protein_id"]]
    return frame_ids

  def localization_lookup(self):
    raw_localizations = self.lookup[
      ["protein_id", "loc_grade1", "loc_grade2", "loc_grade3"]
    ].drop_duplicates()
    df = pd.wide_to_long(
      raw_localizations, stubnames="loc_grade", i="protein_id", j="grade"
    )
    df["localization"] = df["loc_grade"].str.split(";")
    df.drop(columns=["loc_grade"], inplace=True)
    df = df.explode("localization")
    df.reset_index(inplace=True)
    df.dropna(subset=["localization"], inplace=True)
    df.sort_values("protein_id", inplace=True)
    return df

  def protein_lookup(self):
    return (
      self.lookup[["protein_id", "ensembl_id", "gene_symbol"]]
      .drop_duplicates()
      .reset_index(drop=True)
    )

  def frame_protein_lookup(self):
    return self.lookup[["frame_id", "protein_id", "fov_id"]].sort_values(
      "frame_id"
    )

  def get_frame_ids(self) -> np.ndarray:
    return self.lookup["frame_id"].to_numpy()

  def get_n_proteins(self) -> int:
    return len(self.lookup["protein_id"].unique())


@dataclass
class Dataset:
  images: Images
  labels: Labels

  def count_unique_proteins(self) -> int:
    return self._get_unique_proteins().shape[0]

  def get_unique_protein_symbols(self) -> list[str]:
    return self.labels.protein_lookup()["gene_symbol"].to_list()

  def _get_unique_proteins(self) -> np.ndarray:
    return (
      self._get_proteins_across_frames()["protein_id"]
      .drop_duplicates()
      .to_numpy()
    )

  def _get_proteins_across_frames(self) -> pd.DataFrame:
    lookup = self.labels.frame_protein_lookup()
    return lookup[lookup["frame_id"].isin(self.labels.get_frame_ids())]

  def get_batches(
    self,
    rng: jax.Array,
    batch_size: int,
  ):
    """Yields batches of image and label data for training or evaluation."""
    frame_ids = self.labels.get_frame_ids()

    n_frames = len(frame_ids)
    batches_per_epoch = n_frames // batch_size

    # Shuffle data.
    _, rng_perm = jax.random.split(rng, num=2)
    shuffled_idx = jax.random.permutation(rng_perm, n_frames)

    # The model has a softmax layer and expects consecutive integers.
    all_labels = self.labels.lookup[["frame_id", "code"]].set_index("frame_id")

    for idx_set in shuffled_idx[: batches_per_epoch * batch_size].reshape(
      (batches_per_epoch, batch_size)
    ):
      frame_id_set = frame_ids[idx_set]
      yield {
        "frame_ids": frame_id_set,
        "images": self.images.frames[frame_id_set],
        "labels": all_labels.loc[frame_id_set]["code"].to_numpy(dtype=int),
      }

  def get_dummy_input(self):
    return jnp.ones([1, 100, 100, 1])

  def get_random_annotated_frames(
    self, n: int, rng: jax.Array, gene_symbols: list[str] = []
  ) -> list["AnnotatedFrame"]:
    from dlfb.localization.dataset.utils import summarize_localization

    if gene_symbols:
      frames = self.labels.get_frames_of_selected_proteins(gene_symbols)
      frame_ids = frames["frame_id"].to_numpy(np.int32)
    else:
      frame_ids = self.labels.get_frame_ids()

    selected_frame_ids = jax.random.choice(rng, frame_ids, (n,), replace=False)
    lookup = (
      self.labels.frame_protein_lookup()
      .set_index("frame_id", drop=False)
      .loc[selected_frame_ids]
    )
    localization_lookup = (
      self.labels.localization_lookup()
      .groupby("protein_id")
      .apply(summarize_localization, include_groups=False)
      .reset_index(name="localization")
    )
    lookup = lookup.merge(
      localization_lookup, how="left", on="protein_id"
    ).merge(self.labels.protein_lookup(), how="left", on="protein_id")
    annotated_frames = []
    for _, row in lookup.iterrows():
      annotated_frames.append(
        AnnotatedFrame(
          image=self.images.frames[row["frame_id"]],
          localization=row["localization"],
          gene_symbol=row["gene_symbol"],
        )
      )
    return annotated_frames

  def add_frame_localization_to(self, df: pd.DataFrame):
    return df.merge(
      self.labels.frame_protein_lookup(), how="left", on="frame_id"
    ).merge(self.labels.localization_lookup(), how="left", on="protein_id")

  def assign_only_most_predominant_localization(self, df: pd.DataFrame):
    localization_lookup = self.labels.localization_lookup()
    top_grade = (
      localization_lookup.groupby("protein_id")["grade"].min().reset_index()
    )
    top_grade_lookup = (
      localization_lookup.merge(
        top_grade, on=["protein_id", "grade"], how="inner"
      )
      .groupby("protein_id", as_index=False)
      .first()
      .drop(columns=["grade"])
    )
    return df.drop(columns=["grade", "localization"]).merge(
      top_grade_lookup, on=["protein_id"], how="inner"
    )

  def filter_for_single_localization_frames(self, df: pd.DataFrame):
    localization_lookup = self.labels.localization_lookup()
    localization_counts = localization_lookup.groupby("protein_id").nunique()
    single_localization_proteins = localization_counts[
      localization_counts["localization"] == 1
    ].index.to_list()
    return df[df["protein_id"].isin(single_localization_proteins)]

  def filter_for_localization_of_interest(self, df: pd.DataFrame):
    return df[df["localization"].isin(LOCALIZATIONS_OF_INTEREST)]

  def plot_random_frames(
    self,
    n: int,
    rng: jax.Array,
    gene_symbols: list[str] = [],
    figsize: tuple[float, float] = (12, 12),
    with_labels: bool = True,
  ):
    nrows, ncols = calculate_grid_dimensions(n)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1 or ncols == 1:
      axs = np.array(axs).reshape(nrows, ncols)
    af: AnnotatedFrame
    for i, af in enumerate(
      self.get_random_annotated_frames(n, rng, gene_symbols)
    ):
      row, col = divmod(i, ncols)
      axs[row, col].imshow(af.image, cmap="binary_r")
      axs[row, col].axis("off")
      if with_labels:
        label = f"{af.gene_symbol}\n{af.localization}"
        axs[row, col].set_title(
          label,
          fontsize=14,
          x=0.5,
          y=0.9,
          color="black",
          ha="center",
          va="top",
          bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3"),
        )
      # NOTE: Hides any unused subplot axes
      for ax in axs.flat[i + 1 :]:
        ax.axis("off")
    plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
    return fig


@dataclass
class AnnotatedFrame:
  image: np.ndarray
  localization: str
  gene_symbol: str

  def plot(self):
    plt.figure(figsize=(7, 7))
    plt.imshow(self.image)
    plt.axis("off")
    plt.title(
      f"{self.gene_symbol} | {self.localization}",
      fontsize=8,
      x=0.5,
      y=0.9,
      color="white",
      ha="center",
      va="top",
    )
    plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
    plt.show()
