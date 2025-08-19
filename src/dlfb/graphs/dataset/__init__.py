from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd

from dlfb.graphs.dataset.pairs import Pairs


@dataclass
class Dataset:
  """Graph dataset with nodes, pairs, and optional annotations."""

  n_nodes: int
  graph: jraph.GraphsTuple
  pairs: Pairs
  annotation: pd.DataFrame = field(default_factory=pd.DataFrame)

  def subset(
    self, node_ids: jax.Array, keep_original_ids: bool = True
  ) -> "Dataset":
    """Creates a subset of the dataset based on given node IDs."""
    if keep_original_ids:
      gid = node_ids
      n_nodes = self.n_nodes
    else:
      gid = jnp.arange(node_ids.shape[0])
      n_nodes = node_ids.shape[0]

    # Subset graph, pairs, and annotations.
    lookup = {k: i for i, k in enumerate((node_ids.tolist()))}
    graph = self.subset_graph(lookup, gid)
    pairs = self.subset_pairs(lookup)
    annotation = self.subset_annotation(lookup)
    return Dataset(n_nodes, graph, pairs, annotation)

  def subset_graph(self, lookup, gid) -> jraph.GraphsTuple:
    """Generates a subgraph by filtering nodes and reindexing edges."""
    ids = list(lookup.keys())
    edge_mask = np.isin(self.graph.senders, ids) & np.isin(
      self.graph.receivers, ids
    )
    graph = jraph.GraphsTuple(
      nodes={"gid": gid},
      edges=None,
      senders=jnp.array(
        [lookup[k] for k in self.graph.senders[edge_mask].tolist()]
      ),
      receivers=jnp.array(
        [lookup[k] for k in self.graph.receivers[edge_mask].tolist()]
      ),
      n_node=jnp.array([len(ids)]),
      n_edge=jnp.array([edge_mask.sum()]),
      globals=None,
    )
    return graph

  def subset_pairs(self, lookup: dict[int, int]) -> Pairs:
    """Subsets the positive and negative pairs by filtering and re-indexing."""
    ids = list(lookup.keys())
    pairs = {}

    for pair_type in ["pos", "neg"]:
      # Mask pairs to include only valid node combinations.
      pairs_mask = np.isin(getattr(self.pairs, pair_type)[:, 0], ids) & np.isin(
        getattr(self.pairs, pair_type)[:, 1], ids
      )
      pairs[pair_type] = jnp.stack(
        [
          jnp.array(
            [
              lookup[k]
              for k in getattr(self.pairs, pair_type)[:, 0][pairs_mask].tolist()
            ]
          ),
          jnp.array(
            [
              lookup[k]
              for k in getattr(self.pairs, pair_type)[:, 1][pairs_mask].tolist()
            ]
          ),
        ],
        axis=1,
      )

    return Pairs(pos=pairs["pos"], neg=pairs["neg"])

  def subset_annotation(self, lookup: dict[int, int]) -> pd.DataFrame:
    """Subsets and reindexes node annotations using lookup dictionary."""
    # Convert the lookup dictionary to a DataFrame for merging.
    lookup_df = pd.DataFrame(
      {"node_id": lookup.keys(), "new_node_id": lookup.values()}
    )
    annotation = pd.merge(lookup_df, self.annotation, on="node_id", how="left")
    annotation = annotation.drop(columns=["node_id"]).rename(
      columns={"new_node_id": "node_id"}
    )
    return annotation
