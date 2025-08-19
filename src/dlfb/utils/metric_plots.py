import itertools
from collections import OrderedDict
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from dlfb.utils import groom

NAMED_COLORS = OrderedDict(
  [
    ("red", "#e41a1c"),
    ("blue", "#377eb8"),
    ("green", "#4daf4a"),
    ("purple", "#984ea3"),
    ("orange", "#ff7f00"),
    ("yellow", "#ffff33"),
    ("brown", "#a65628"),
    ("pink", "#f781bf"),
    ("gray", "#999999"),
  ]
)

NAMED_LINESTYLES = OrderedDict(
  [
    ("solid", "solid"),
    ("dashed", "dashed"),
    ("dashdot", "dashdot"),
    ("dotted", "dotted"),
    ("densely-dotted", (0, (1, 1))),
    ("long-dash-with-offset", (5, (10, 3))),
    ("loosely-dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("loosely-dotted", (0, (1, 10))),
    ("dotted", (0, (1, 5))),
    ("densely-dashed", (0, (5, 1))),
    ("loosely-dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely-dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely-dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely-dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
  ]
)


DEFAULT_SPLIT_COLORS = {
  "train": NAMED_COLORS["blue"],
  "valid": NAMED_COLORS["green"],
  "test": NAMED_COLORS["orange"],
}


PANEL_DEFAULTS = {"no_std": False, "ylim": (None, None)}


class MissingData(Exception): ...


class DifferingUnits(Exception): ...


class MixedEpochStepMetrics(Exception): ...


class MetricsPlotter:
  def __init__(
    self,
    metrics: dict,
    color_opts: list[str] = None,
    line_opts: list = None,
  ):
    self.metrics: pd.DataFrame = to_df(metrics)
    self.color_opts = color_opts or list(NAMED_COLORS.values())
    self.line_opts = line_opts or list(NAMED_LINESTYLES.values())

  def plot(
    self, panels: list[dict[str, Any]], panel_size: tuple[int, int] = (4, 4)
  ):
    panels = self._fill_defaults(panels)
    all_splits, all_metrics = self._inspect_panels(panels)
    fig, axes = self._setup_plot(panels, panel_size)
    style_maps = self._get_style_maps(all_splits, all_metrics)
    for ax, (panel) in zip(axes[0], panels):
      self._plot_panel(ax, style_maps, panel)
    self._finalize_plot()
    return fig

  def _fill_defaults(self, panels):
    return [{**PANEL_DEFAULTS, **panel} for panel in panels]

  def _inspect_panels(
    self, panels: list[dict[str, Any]]
  ) -> tuple[list[str], list[str]]:
    all_splits, all_metrics = set(), set()
    data = self.metrics[["split", "metric"]].drop_duplicates()
    if not len(self.metrics["unit"].unique()) == 1:
      raise DifferingUnits
    for panel in panels:
      for split, metric in itertools.product(panel["splits"], panel["metrics"]):
        if not ((data["split"] == split) & (data["metric"] == metric)).any():
          raise MissingData
        else:
          all_splits.add(split)
          all_metrics.add(metric)
    return (sorted(list(i)) for i in [all_splits, all_metrics])

  def _setup_plot(self, panels, panel_size):
    n_panels = len(panels)
    fig, axes = plt.subplots(
      1,
      n_panels,
      figsize=(panel_size[0] * n_panels, panel_size[1]),
      squeeze=False,
    )
    return fig, axes

  def _get_style_maps(self, all_splits, all_metrics):
    """Gets plot-level consistent styling of panels."""
    if len(all_splits) == 1:
      color_iter = itertools.cycle(self.color_opts)

      def get_color(_):
        return next(color_iter)

      def get_linestyle(_):
        return "-"
    else:
      line_iter = itertools.cycle(self.line_opts)

      def get_color(split):
        return DEFAULT_SPLIT_COLORS.get(split.lower())

      linestyle_map = {m: next(line_iter) for m in all_metrics}

      def get_linestyle(metric):
        return linestyle_map[metric]

    style_maps = {}
    for m in all_metrics:
      for s in all_splits:
        label = f"{groom(m)} ({groom(s)})" if len(all_metrics) > 1 else groom(s)
        style_maps[f"{s}_{m}"] = {
          "label": label,
          "color": get_color(s),
          "linestyle": get_linestyle(m),
        }
    return style_maps

  def _plot_panel(self, ax: Axes, style_maps, panel: dict[str, Any]):
    for metric in panel["metrics"]:
      for split in panel["splits"]:
        self._plot_data(
          ax, split, metric, style_maps[f"{split}_{metric}"], panel["no_std"]
        )
    self._finalize_panel(ax, panel)

  def _plot_data(self, ax: Axes, split, metric, style_map, no_std):
    x, y, std = self._extract_values(split, metric)
    ax.plot(x, y, **style_map)
    if not no_std:
      ax.fill_between(
        x, y - std, y + std, color=style_map["color"], alpha=0.2, linewidth=0
      )

  def _extract_values(self, split, metric):
    data = self.metrics[
      (self.metrics["split"] == split) & (self.metrics["metric"] == metric)
    ]
    x, y, std = [data[col].to_numpy() for col in ["round", "mean", "std"]]
    return x, y, std

  def _finalize_panel(self, ax: Axes, panel: dict[str, Any]):
    self._set_axes(ax, panel)
    if "title" in panel:
      ax.set_title(panel["title"])
    ax.grid(True)
    ax.legend()

  def _set_axes(self, ax: Axes, panel: dict[str, Any]):
    self._set_xaxis(ax)
    self._set_yaxis(ax, panel)

  def _set_xaxis(self, ax: Axes):
    ax.set_xlabel(self.metrics["unit"].iloc[0].title())
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

  def _set_yaxis(self, ax: Axes, panel: dict[str, Any]):
    metric = list(set(panel["metrics"]))
    ax.set_ylabel(groom(metric[0]) if len(metric) == 1 else "Value")
    ax.set_yscale("linear")
    ax.set_ylim(*self._get_ylim(panel))

  def _get_ylim(self, panel: dict[str, Any]) -> tuple[int, int]:
    data = self.metrics[self.metrics["metric"].isin(panel["metrics"])]
    if not panel["no_std"]:
      y_min = (data["mean"] - data["std"]).min()
      y_max = (data["mean"] + data["std"]).max()
    else:
      y_min, y_max = data["mean"].min(), data["mean"].max()
    y_min, y_max = (
      panel["ylim"][0] if panel["ylim"][0] is not None else y_min,
      panel["ylim"][1] if panel["ylim"][1] is not None else y_max,
    )
    return y_min, y_max

  def _finalize_plot(self):
    sns.despine()
    plt.tight_layout()


def to_df(exported_metrics: dict[str, dict[str:float]]) -> pd.DataFrame:
  rows = []
  for split, metrics in exported_metrics.items():
    for metric, records in metrics.items():
      for record in records:
        rows.append(
          {
            "split": split,
            "metric": metric,
            "round": record["round"],
            "mean": record["mean"],
            "std": record["std"],
            "unit": record["unit"],
          }
        )
  return pd.DataFrame(rows)


def from_df(df: pd.DataFrame) -> dict[str, dict[str, list[dict[str, float]]]]:
  nested = {}
  for (split, metric), group in df.groupby(["split", "metric"]):
    records = group.sort_values("round")[
      ["mean", "std", "round", "unit"]
    ].to_dict(orient="records")
    nested.setdefault(split, {})[metric] = records
  return nested
