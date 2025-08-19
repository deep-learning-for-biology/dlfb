import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from dlfb.graphs.dataset import Dataset
from dlfb.utils.metric_plots import MetricsPlotter


def plot_graph(dataset: Dataset) -> plt.Figure:
  """Plots a circular layout graph with annotated drug names."""
  fig = plt.figure(figsize=(20, 20))

  # Build the graph using networkx.
  G: nx.Graph = nx.Graph()
  G.add_edges_from(
    np.stack((dataset.graph.senders, dataset.graph.receivers), axis=1)
  )
  pos = nx.circular_layout(G)

  # Compute node label angles for improved readability.
  theta = {k: np.arctan2(v[1], v[0]) * 180 / np.pi for k, v in pos.items()}

  nx.draw(
    G,
    pos,
    with_labels=False,
    node_color="lightgray",
    edge_color="gray",
    node_size=10,
    alpha=0.1,
  )

  labels = (
    dataset.annotation[dataset.annotation["node_id"].isin(G.nodes)]
    .set_index("node_id")["drug_name"]
    .to_dict()
  )
  label_objects = nx.draw_networkx_labels(
    G, pos, labels=labels, font_size=10, horizontalalignment="left"
  )

  # Adjust label rotation based on node position.
  for key, text in label_objects.items():
    if 90 < theta[key] or theta[key] < -90:
      angle = 180 + theta[key]
      text.set_horizontalalignment("right")
    else:
      angle = theta[key]
      text.set_horizontalalignment("left")
    text.set_va("center")
    text.set_rotation(angle)
    text.set_rotation_mode("anchor")

  return fig


def plot_learning(metrics, splits=["train", "valid"]) -> plt.Figure:
  """Plots learning metrics such as loss and hits@20 over training epochs."""
  return MetricsPlotter(metrics).plot(
    panels=[
      {"title": "Learning Curves", "metrics": ["loss"], "splits": splits},
      {
        "title": "Hits@20",
        "metrics": ["hits@20"],
        "splits": splits,
        "ylim": (0, 1),
      },
    ]
  )
