import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd
from ogb.linkproppred import LinkPropPredDataset

from dlfb.graphs.dataset import Dataset
from dlfb.graphs.dataset.pairs import Pairs


class DatasetBuilder:
  """Builds dataset splits for drug-drug interaction analysis and modeling."""

  def __init__(self, path):
    """Initializes the dataset builder with a path to the dataset."""
    self.path = path

  def build(
    self,
    node_limit: int | None = None,
    rng: jax.Array | None = None,
    keep_original_ids: bool = False,
  ) -> dict[str, Dataset]:
    """Builds and returns a dictionary of dataset splits."""
    dataset_splits = {}
    n_nodes, split_pairs = self.download()
    annotation = self.prepare_annotation()

    for name, split in split_pairs.items():
      pos_pairs, neg_pairs = split["edge"], split["edge_neg"]
      graph = self.prepare_graph(n_nodes, pos_pairs)
      pairs = self.prepare_pairs(graph, pos_pairs, neg_pairs)
      dataset_splits.update({name: Dataset(n_nodes, graph, pairs, annotation)})

    if node_limit and (rng is not None):
      dataset_splits = self.subset(
        dataset_splits, rng, node_limit, keep_original_ids
      )

    return dataset_splits

  def download(self) -> tuple[int, dict]:
    """Downloads the dataset and returns the number of nodes and edge splits."""
    raw = LinkPropPredDataset(name="ogbl-ddi", root=self.path)
    # Note that the full graph is available in raw.graph
    n_nodes = raw[0]["num_nodes"]
    split_pairs = raw.get_edge_split()
    split_pairs["train"]["edge_neg"] = None  # Placeholder for negative edges.
    return n_nodes, split_pairs

  def prepare_annotation(self) -> pd.DataFrame:
    """Annotates nodes by mapping node IDs to database IDs and drug names."""
    ddi_descriptions = pd.read_csv(
      f"{self.path}/ogbl_ddi/mapping/ddi_description.csv.gz"
    )
    node_to_dbid_lookup = pd.read_csv(
      f"{self.path}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    )
    # Merge first and second drug descriptions into a single lookup.
    first_drug = ddi_descriptions.loc[
      :, ["first drug id", "first drug name"]
    ].rename(columns={"first drug id": "dbid", "first drug name": "drug_name"})

    second_drug = ddi_descriptions.loc[
      :, ["second drug id", "second drug name"]
    ].rename(
      columns={"second drug id": "dbid", "second drug name": "drug_name"}
    )
    dbid_to_name_lookup = (
      pd.concat([first_drug, second_drug])
      .drop_duplicates()
      .reset_index(drop=True)
    )

    # Merge with node-to-DBID lookup.
    annotation = pd.merge(
      node_to_dbid_lookup.rename(
        columns={"drug id": "dbid", "node idx": "node_id"}
      ),
      dbid_to_name_lookup,
      on="dbid",
      how="inner",
    )
    return annotation

  def prepare_graph(
    self, n_nodes: int, pos_pairs: jax.Array
  ) -> jraph.GraphsTuple:
    """Prepares a Jraph graph from positive edge pairs."""
    senders, receivers = self.make_undirected(pos_pairs[:, 0], pos_pairs[:, 1])
    graph = jraph.GraphsTuple(
      nodes={"gid": jnp.arange(n_nodes)},  # Optional global node ID.
      edges=None,
      senders=senders,
      receivers=receivers,
      n_node=jnp.array([n_nodes]),
      n_edge=jnp.array([len(senders)]),
      globals=None,
    )
    return graph

  @staticmethod
  def make_undirected(
    senders: jax.Array, receivers: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Makes an undirected graph by duplicating edges in both directions."""
    # Jraph requires undirected graphs to have both A->B and B->A edges
    # explicitly.
    senders_undir = jnp.concatenate((senders, receivers))
    receivers_undir = jnp.concatenate((receivers, senders))
    return senders_undir, receivers_undir

  def prepare_pairs(
    self, graph: int, pos_pairs: jax.Array, neg_pairs: jax.Array | None = None
  ) -> Pairs:
    """Prepares positive and negative edge pairs."""
    if neg_pairs is None:
      neg_pairs = self.infer_negative_pairs(graph)
    return Pairs(pos=pos_pairs, neg=neg_pairs)

  def infer_negative_pairs(self, graph: jraph.GraphsTuple) -> jax.Array:
    """Infers negative edge pairs in a graph."""
    # Initialize a matrix where all possible edges are marked as potential
    # negative edges (1).
    neg_adj_mask = np.ones((graph.n_node[0], graph.n_node[0]), dtype=np.uint8)

    # Mask out existing edges in the graph (set to 0).
    neg_adj_mask[graph.senders, graph.receivers] = 0

    # Use the upper triangular part of the matrix to avoid duplicate pairs and
    # self-loops.
    neg_adj_mask = np.triu(neg_adj_mask, k=1)
    neg_pairs = jnp.array(neg_adj_mask.nonzero()).T  # Extract indices.
    return neg_pairs

  def subset(
    self,
    dataset_splits: dict[str, Dataset],
    rng: jax.Array,
    node_limit: int,
    keep_original_ids: bool = False,
  ) -> dict[str, Dataset]:
    """Creates subset of dataset splits by sampling a fixed number of nodes."""
    # Get a random subset of node_ids.
    node_ids = jax.random.choice(
      rng, dataset_splits["train"].n_nodes, (node_limit,), replace=False
    )

    # Subset every dataset split by the same node_ids.
    dataset_subset_splits = {}
    for name, dataset in dataset_splits.items():
      dataset_subset_splits[name] = dataset.subset(node_ids, keep_original_ids)

    return dataset_subset_splits
