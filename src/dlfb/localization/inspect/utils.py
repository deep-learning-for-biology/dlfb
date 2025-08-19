import itertools

import numpy as np
from matplotlib import pyplot as plt

from dlfb.localization.constants import LOCALIZATIONS_OF_INTEREST


def map_labels_to_color(labels, scale=plt.get_cmap("plasma")):
  mapping = get_label_to_color_mapping(labels, scale)
  return [mapping[f] for f in labels]


def get_label_to_color_mapping(labels, scale):
  unique_labels = sorted(np.unique(labels).tolist())
  return dict(zip(unique_labels, scale(np.linspace(0, 1, len(unique_labels)))))


LOCALIZATIONS_COLORS: dict[str, str] = {
  localization: color
  for localization, color in zip(
    LOCALIZATIONS_OF_INTEREST,
    plt.get_cmap("rainbow")(np.linspace(0, 1, len(LOCALIZATIONS_OF_INTEREST))),
  )
}


LOCALIZATIONS_MARKERS: dict[str, str] = {
  localization: marker
  for localization, marker in zip(
    LOCALIZATIONS_OF_INTEREST,
    itertools.cycle(["o", "s", "^", "v", "P", "X", "D", "*", "h"]),
  )
}
