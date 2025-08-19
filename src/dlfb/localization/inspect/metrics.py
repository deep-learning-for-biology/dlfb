from dlfb.utils.metric_plots import MetricsPlotter


def plot_losses(metrics, splits=["train", "valid"]):
  return MetricsPlotter(metrics).plot(
    panels=[
      {
        "title": "Evolution of the 4 Loss Components on the Training Set",
        "metrics": [
          "recon_loss",
          "codebook_loss",
          "commitment_loss",
          "classification_loss",
        ],
        "splits": ["train"],
      },
      {"title": "Learning Curves", "metrics": ["total_loss"], "splits": splits},
    ]
  )


def plot_perplexity(metrics, splits=["train", "valid"]):
  return MetricsPlotter(metrics).plot(
    panels=[
      {
        "title": "Evolution of the Perplexity (~Usage of the Codebook)",
        "metrics": ["perplexity"],
        "splits": splits,
      }
    ]
  )
