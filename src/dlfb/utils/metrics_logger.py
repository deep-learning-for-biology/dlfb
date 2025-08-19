from collections import defaultdict

import numpy as np


class MetricsLogger:
  def __init__(self):
    self.history = defaultdict(
      lambda: defaultdict(list)
    )  # split -> metric -> list of (mean, std)
    self.step_buffer = defaultdict(
      lambda: defaultdict(list)
    )  # temporary step logs

  def log_step(self, split, **metrics):
    for metric, value in metrics.items():
      self.step_buffer[split][metric].append(value)

  def flush(self, **kwargs):
    unit, round = next(iter(kwargs.items()))
    for split in list(self.step_buffer.keys()):
      for metric, values in self.step_buffer[split].items():
        if values:
          mean = float(np.mean(values))
          std = float(np.std(values))
          self.history[split][metric].append(
            {"mean": mean, "std": std, "unit": unit, "round": round}
          )
      self.step_buffer[split].clear()

  def latest(self, metrics: list) -> str:
    self.ensure_metrics_exist(metrics)
    grouped = defaultdict(dict)
    for split, metric_dict in self.history.items():
      for metric, history in metric_dict.items():
        if metric not in metrics:
          continue
        latest = history[-1]
        grouped[metric][split] = f"{latest['mean']:.4f}±{latest['std']:.4f}"

    parts = []
    for metric, split_vals in grouped.items():
      joined = ", ".join(f"{split}={val}" for split, val in split_vals.items())
      parts.append(f"{metric}: {joined}")

    return " | ".join(parts)

  def ensure_metrics_exist(self, metrics):
    requested = set(metrics)
    if requested:
      all_available = {
        metric for split in self.history.values() for metric in split.keys()
      }
      missing = requested - all_available
      if missing:
        raise ValueError(f"Missing metrics in history: {sorted(missing)}")

  def export(self) -> dict[str, dict[str, list[dict[str, float]]]]:
    return {
      split: {metric: list(values) for metric, values in metrics.items()}
      for split, metrics in self.history.items()
    }
