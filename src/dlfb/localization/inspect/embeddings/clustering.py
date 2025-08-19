import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt

from dlfb.localization.dataset import Dataset
from dlfb.localization.inspect.utils import (
  LOCALIZATIONS_COLORS,
  LOCALIZATIONS_MARKERS,
)


def calculate_projection(
  frame_embeddings: dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
  first = next(iter(frame_embeddings.values()))
  if isinstance(first, np.ndarray):
    projection = calculate_single_projection(frame_embeddings)
  elif isinstance(first, dict):
    projection = calculate_aligned_projection(frame_embeddings)
  return projection


def calculate_single_projection(frame_embeddings) -> pd.DataFrame:
  mapper = umap.UMAP(random_state=42).fit(frame_embeddings["frame_histograms"])
  projection = pd.DataFrame(
    {
      "run_name": 0,
      "frame_id": frame_embeddings["frame_ids"],
      "UMAP1": mapper.embedding_[:, 0],
      "UMAP2": mapper.embedding_[:, 1],
    }
  )
  return projection


def calculate_aligned_projection(
  frame_embeddings: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
  mapper = umap.AlignedUMAP(random_state=42).fit(
    X=[e["frame_histograms"] for e in frame_embeddings.values()],
    relations=set_frame_relations(frame_embeddings),
  )
  dfs = []
  for i, umap_embeddings in enumerate(mapper.embeddings_):
    dfs.append(
      pd.DataFrame(
        {
          "run_name": i,
          "frame_id": list(frame_embeddings.values())[i]["frame_ids"],
          "UMAP1": umap_embeddings[:, 0],
          "UMAP2": umap_embeddings[:, 1],
        }
      )
    )
  projections = pd.concat(dfs)
  return projections


def set_frame_relations(
  frame_embeddings: dict[str, dict[str, np.ndarray]],
) -> list[dict[int, int]]:
  emb = list(frame_embeddings.values())
  frame_relations = [
    {i: i for i in range(emb[n]["frame_ids"].shape[0])}
    for n in range(len(emb) - 1)
  ]
  return frame_relations


def plot_projection(
  projections: pd.DataFrame,
  dataset: Dataset,
  titles: list[str] = [],
  subset_mode: str = "predominant",
  localizations_color: dict[str, str] = LOCALIZATIONS_COLORS,
  localizations_markers: dict[str, str] = LOCALIZATIONS_MARKERS,
  random_seed: int = 42,
  point_highlight_fraction: float = 0.01,
  highlight_scale: float = 60,
):
  plot_data = dataset.add_frame_localization_to(projections)
  match subset_mode:
    case "single":
      plot_data = dataset.filter_for_single_localization_frames(plot_data)
    case "predominant":
      plot_data = dataset.assign_only_most_predominant_localization(plot_data)
  plot_data = dataset.filter_for_localization_of_interest(plot_data)
  run_names = projections["run_name"].unique().tolist()
  n_subplots = len(run_names)
  np.random.seed(random_seed)
  if n_subplots == 1:
    fig, ax = plt.subplots(figsize=(7, 7))
    axs = [ax]
  else:
    fig, axs = plt.subplots(
      1, n_subplots, figsize=(n_subplots * 7, 7), sharex=True, sharey=True
    )
  for i, ax in enumerate(axs):
    subset = projections[projections["run_name"] == run_names[i]]
    ax.scatter(subset["UMAP1"], subset["UMAP2"], c="lightgray", s=1, marker="o")
    for localization in plot_data["localization"].unique():
      subset = plot_data[
        (plot_data["localization"] == localization)
        & (plot_data["run_name"] == run_names[i])
      ]
      mask = np.random.rand(len(subset)) < point_highlight_fraction
      sub_high = subset[mask]
      sub_norm = subset[~mask]
      ax.scatter(
        x=sub_norm["UMAP1"],
        y=sub_norm["UMAP2"],
        s=1,
        alpha=0.9,
        color=localizations_color[localization],
        marker=localizations_markers[localization],
      )
      ax.scatter(
        x=sub_high["UMAP1"],
        y=sub_high["UMAP2"],
        label=localization,
        alpha=0.9,
        color=localizations_color[localization],
        marker=localizations_markers[localization],
        s=highlight_scale,
        edgecolors="black",
        linewidths=0.8,
      )
    if titles:
      ax.set_title(titles[i], fontsize=16)
    if i == 0:
      ax.legend(
        title="Localization",
        frameon=False,
        loc="best",
        fontsize=14,
        title_fontsize=14
      )
      ax.set_ylabel("UMAP2", fontsize=14)
    ax.set_xlabel("UMAP1", fontsize=14)
  plt.tight_layout()
  return fig
