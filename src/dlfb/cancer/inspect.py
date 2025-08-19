from collections import defaultdict
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dlfb.cancer.dataset import Dataset
from dlfb.utils import wrap_text
from dlfb.utils.metric_plots import MetricsPlotter


def plot_learning(metrics):
  """Plot training/validation loss and precision/recall over time."""
  return MetricsPlotter(metrics).plot(
    panels=[
      {
        "title": "Learning Curves",
        "metrics": ["loss"],
        "splits": ["train", "valid"],
        "ylim": (0, None),
      },
      {
        "title": "Precision / Recall",
        "metrics": ["precision_weighted", "recall_weighted"],
        "splits": ["train", "valid"],
        "ylim": (0, 1),
        "no_std": True,
      },
    ],
    panel_size=(4, 4),
  )


def display_augmented_images(
  labels: list[str], images: np.ndarray, ncols: int = 6
) -> plt.Figure:
  """Display a grid of augmented images with their corresponding labels."""
  nrows = (len(labels) + ncols - 1) // ncols
  fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.0 * nrows))
  axes = axes.flatten()

  for label, image, ax in zip(labels, images, axes):
    ax.imshow(image)
    ax.set_title(label)
    ax.axis("off")

  # Hide any unused subplots.
  for ax in axes[len(labels) :]:
    ax.axis("off")

  plt.tight_layout()
  plt.show()
  return fig


def plot_confusion(predictions, normalize: bool = False):
  """Display a (normalized) confusion matrix for predicted vs. true classes."""
  cm = confusion_matrix(predictions["label"], predictions["pred"])
  if normalize:
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  lookup_table = predictions[["class", "label"]]
  lookup_dict = dict(zip(lookup_table["label"], lookup_table["class"]))
  class_names = list(dict(sorted(lookup_dict.items())).values())
  disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
  disp.plot()
  plt.show()


def plot_classified_images(
  predictions: pd.DataFrame,
  dataset: Dataset,
  preprocessor: Callable,
  max_images: int = 4,
):
  """Plot classified images in a grid with precision/recall annotations."""
  cm = confusion_matrix(predictions["label"], predictions["pred"])
  precision = np.diag(cm) / np.sum(cm, axis=0)
  recall = np.diag(cm) / np.sum(cm, axis=1)
  image_grid = defaultdict(list)
  for pair, group in predictions.groupby(["label", "pred"]):
    image_grid[pair] = group["frame_id"].tolist()
  # TODO: dedup with plot_confusion's
  lookup_table = predictions[["class", "label"]]
  lookup_dict = dict(zip(lookup_table["label"], lookup_table["class"]))
  unique_labels = list(lookup_dict.keys())
  fig, axes = plt.subplots(
    len(unique_labels), len(unique_labels), figsize=(10, 10)
  )
  fig.suptitle("Predicted")
  fig.supylabel("Truth")

  for i in unique_labels:
    for j in unique_labels:
      ax = axes[i, j]
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_frame_on(True)
      if (i, j) in image_grid:
        composite_image = create_composite_image(
          dataset.get_images(preprocessor, jnp.array(image_grid[(i, j)])),
          max_images,
        )
        ax.imshow(composite_image)
      if i == 0:
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(
          wrap_text(lookup_dict[j], 10) + f"\nP: {precision[j]:.2f}",
          fontsize=12,
        )
      if j == 0:
        ax.set_ylabel(
          wrap_text(lookup_dict[i], 10) + f"\nR: {recall[i]:.2f}",
          fontsize=12,
        )
      if i == j:
        color = "green"
      else:
        if cm[i, j] == 0:
          color = "green"
        else:
          color = "red"
      if cm[i, j] != 0:
        ax.text(
          0.95,
          0.05,
          cm[i, j],
          verticalalignment="bottom",
          horizontalalignment="right",
          transform=ax.transAxes,
          color=color,
          fontsize=12,
          weight="bold",
        )
      for a in ["top", "bottom", "left", "right"]:
        ax.spines[a].set_linewidth(1)
        ax.spines[a].set_color(color)

  plt.tight_layout()
  plt.subplots_adjust(wspace=0.05, hspace=0.05)
  plt.show()
  return fig


def create_composite_image(images, max_images):
  """Combine a list of images into a single grid-based composite image."""
  nrow, ncol = calculate_grid_dimensions(max_images)
  img_height, img_width, num_channels = images[0].shape
  composite_image = np.ones((img_height * nrow, img_width * ncol, num_channels))
  for idx, img in enumerate(images[:max_images]):
    if idx >= nrow * ncol:
      break
    row = idx // ncol
    col = idx % ncol
    composite_image[
      row * img_height : (row + 1) * img_height,
      col * img_width : (col + 1) * img_width,
      :,
    ] = img
  return composite_image


def calculate_grid_dimensions(n, ratio=1):
  """Compute rows/cols to arrange n items in a grid with given aspect ratio."""
  num_cols = int(np.ceil(np.sqrt(n) * ratio))
  num_rows = int(np.ceil(n / num_cols))
  return num_rows, num_cols
