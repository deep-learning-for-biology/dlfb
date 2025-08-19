import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dlfb.dna.utils import compute_input_gradient, filter_sequences_by_label
from dlfb.utils.metric_plots import MetricsPlotter


def describe_change(
  idx, deltas: np.ndarray, sequence: np.ndarray, bases=["A", "C", "G", "T"]
) -> str:
  """Describe the effect of mutating one base to another."""
  seq_pos, base_idx = idx
  original_base = bases[np.argmax(sequence[seq_pos])]
  new_base = bases[base_idx]
  perc_delta = deltas[idx].item() * 100
  direction = "increase" if perc_delta >= 0 else "decrease"
  mutation = f"{original_base}→{new_base}"
  change = f"position {seq_pos} with {mutation} ({perc_delta:.2f}% {direction})"
  return change


def plot_binding_site(panels, highlight: tuple[int, int] | None = None):
  """Plot a line plot and heatmap of contribution scores in a DNA sequence."""
  fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(7, 3),
    gridspec_kw={"height_ratios": [1, 3]},
    sharex=True,
    constrained_layout=True,
  )

  ax1, ax2 = axes

  # Position-wise line plot.
  ax1.plot(panels["line"]["values"], c="black")
  ax1.set_ylabel(panels["line"]["label"])
  for spine in ax1.spines.values():
    spine.set_visible(False)
  ax1.set_xticks([])

  # Heatmap without colorbar.
  heatmap = sns.heatmap(
    panels["tiles"]["values"].T,
    ax=ax2,
    center=0,
    cbar=False,
    cmap="viridis",
    yticklabels=["A", "C", "G", "T"],
  )
  ax2.set_xlabel("Position in DNA sequence")
  ax2.set_ylabel("DNA Base")

  # Add shared vertical colorbar.
  cbar = fig.colorbar(
    heatmap.collections[0],
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.02,
  )
  cbar.set_label(panels["tiles"]["label"])

  # Optional highlight region.
  if highlight:
    start, end = highlight
    rect = plt.Rectangle(
      xy=(start, 0),
      width=end - start,
      height=panels["tiles"]["values"].shape[1],
      linewidth=3,
      edgecolor="black",
      facecolor="none",
    )
    ax2.add_patch(rect)

  return fig


def plot_10_gradients(state, dataset, target_label, max_count=10):
  """Plot saliency maps for 10 sequences of a given label."""
  input_grads = [
    compute_input_gradient(state, sequence)
    for sequence in filter_sequences_by_label(dataset, target_label, max_count)
  ]
  fig, axes = plt.subplots(nrows=int(max_count / 2), ncols=2, figsize=(12, 6))
  for ax, input_grad in zip(axes.flatten(), input_grads):
    sns.heatmap(
      input_grad.T,
      cmap="viridis",
      center=0,
      xticklabels=False,
      yticklabels=False,
      cbar=False,
      ax=ax,
    )
  plt.tight_layout()
  return fig


def plot_learning(metrics, tf):
  """Visualize loss and metrics over time for a given TF."""
  return MetricsPlotter(metrics).plot(
    panels=[
      {
        "title": f"{tf}: Learning Curves",
        "metrics": ["loss"],
        "splits": ["train", "valid"],
        "ylim": (0, None),
        "no_std": False,
      },
      {
        "title": f"{tf}: Accuracy & auROC",
        "metrics": ["accuracy", "auc"],
        "splits": ["valid"],
        "ylim": (0, 1),
        "no_std": True,
      },
    ],
    panel_size=(4, 4),
  )
