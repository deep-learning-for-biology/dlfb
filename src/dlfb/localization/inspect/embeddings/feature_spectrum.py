import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage

from dlfb.localization.constants import LOCALIZATIONS_OF_INTEREST
from dlfb.localization.inspect.embeddings.utils import np_euclidian_distance
from dlfb.localization.inspect.utils import (
  LOCALIZATIONS_COLORS,
  map_labels_to_color,
)
from dlfb.utils import int_to_roman


def plot_encoding_corr_heatmap(corr_idx_idx, tree, encoding_clusters):
  heatmap = sns.clustermap(
    corr_idx_idx,
    cmap="viridis",
    vmin=-1,
    vmax=1,
    xticklabels=False,
    yticklabels=False,
    col_linkage=tree,
    row_linkage=tree,
    col_colors=map_labels_to_color(encoding_clusters),
  )
  heatmap.ax_col_dendrogram.set_title("Pearson Correlation Hierarchy Link")
  heatmap.ax_heatmap.set_xlabel("VQ Index")
  heatmap.ax_heatmap.set_ylabel("VQ Index")
  heatmap.ax_row_dendrogram.set_visible(False)
  return heatmap


def plot_stacked_histrograms(
  localizations,
  localization_histograms,
  tree,
  encoding_clusters,
  localization_colors=LOCALIZATIONS_COLORS,
):
  loc_idx = [
    i for i, lo in enumerate(localizations) if lo in LOCALIZATIONS_OF_INTEREST
  ]
  localizations = [localizations[i] for i in loc_idx]
  localization_histograms = localization_histograms[loc_idx, :]

  indices = leaves_list(tree)
  dfs = []
  for i, localization in enumerate(localizations):
    dfs.append(
      pd.DataFrame(
        {
          "localization": localization,
          "cluster": encoding_clusters[indices],
          "vq_indices_str": [f"vq-{idx}" for idx in indices],
          "freq": localization_histograms[i, indices],
        }
      )
    )
  df = pd.concat(dfs)

  n_localizations = len(localizations)
  localization_order = get_localization_order(localization_histograms)
  localizations = [localizations[i] for i in localization_order]

  n_clusters = len(np.unique(encoding_clusters))

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_gridspec(
    nrows=n_localizations,
    ncols=n_clusters,
    hspace=0,
    wspace=0,
    width_ratios=calcuate_width_ratios(df),
  )
  for i, localization in enumerate(localizations):
    df_panel_row = df[(df["localization"] == localization)]
    panel_row_ylim = df_panel_row["freq"].agg(["min", "max"])
    for j, cluster in enumerate(range(1, n_clusters + 1)):
      df_panel = df_panel_row[(df_panel_row["cluster"] == cluster)]
      ax1 = fig.add_subplot(ax[i, j])
      ax1.set_ylim(panel_row_ylim)
      ax1.bar(
        x="vq_indices_str",
        height="freq",
        data=df_panel,
        color=localization_colors[localization],
      )
      if j == 0:
        ax1.text(
          -0.01,
          0.5,
          localization,
          transform=ax1.transAxes,
          rotation=45,
          verticalalignment="top",
          horizontalalignment="right",
          color=localization_colors[localization],
        )
      if i == 0:
        ax1.set_title(
          int_to_roman(cluster),
          color=map_labels_to_color(range(1, n_clusters + 1))[j],
        )
      ax1.set(xticks=[], yticks=[])
  return fig


def get_localization_order(localization_histograms):
  dists = np_euclidian_distance(
    localization_histograms, localization_histograms
  )
  tree = linkage(
    dists, method="average", metric="euclidean", optimal_ordering=True
  )
  return leaves_list(tree)


def calcuate_width_ratios(df):
  width_ratios = {}
  for cluster, rows in df[["cluster", "vq_indices_str"]].groupby(["cluster"]):
    width_ratios.update(
      {
        cluster: len(np.unique(rows["vq_indices_str"]))
        / len(np.unique(df["vq_indices_str"]))
      }
    )
  return [width_ratios[k] for k in sorted(width_ratios.keys())]
