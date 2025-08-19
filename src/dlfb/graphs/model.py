import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from flax import linen as nn
from flax.training import train_state


class TrainState(train_state.TrainState):
  """Extends TrainState to include a random key for RNG."""

  key: jax.Array


class DdiModel(nn.Module):
  """Graph-based model for predicting drug-drug interactions (DDIs)."""

  n_nodes: int
  embedding_dim: int
  dropout_rate: float
  last_layer_self: bool
  degree_norm: bool
  n_mlp_layers: int = 2

  def setup(self):
    """Initializes the node encoder and link predictor modules."""
    self.node_encoder = NodeEncoder(
      self.n_nodes,
      self.embedding_dim,
      self.last_layer_self,
      self.degree_norm,
      self.dropout_rate,
    )
    self.link_predictor = LinkPredictor(
      self.embedding_dim, self.n_mlp_layers, self.dropout_rate
    )

  def __call__(
    self,
    graph: jraph.GraphsTuple,
    pairs: dict,
    is_training: bool,
    is_pred: bool = False,
  ):
    """Generates interaction scores for node pairs."""
    # Compute node embeddings. The 'h' stands for hidden state or embedding.
    h = self.node_encoder(graph, is_training)

    if is_pred:
      scores = self.link_predictor(h[pairs[:, 0]], h[pairs[:, 1]], False)

    else:
      pos_senders, pos_receivers = pairs["pos"][:, 0], pairs["pos"][:, 1]
      neg_senders, neg_receivers = pairs["neg"][:, 0], pairs["neg"][:, 1]
      scores = {
        "pos": self.link_predictor(
          h[pos_senders], h[pos_receivers], is_training
        ),
        "neg": self.link_predictor(
          h[neg_senders], h[neg_receivers], is_training
        ),
      }
    return scores

  def create_train_state(self, rng: jax.Array, dummy_input, tx) -> TrainState:
    """Initializes the training state with model parameters."""
    rng, rng_init, rng_dropout = jax.random.split(rng, 3)
    variables = self.init(rng_init, is_training=False, **dummy_input)
    return TrainState.create(
      apply_fn=self.apply, params=variables["params"], tx=tx, key=rng_dropout
    )

  @staticmethod
  def add_mean_embedding(embeddings: jax.Array) -> jax.Array:
    """Concatenates a mean embedding to the existing embeddings."""
    mean_embeddings = jnp.mean(embeddings, axis=0, keepdims=True)
    embeddings = jnp.concatenate([embeddings, mean_embeddings], axis=0)
    return embeddings


class NodeEncoder(nn.Module):
  """Encodes nodes into embeddings using a two-layer GraphSAGE model."""

  n_nodes: int
  embedding_dim: int
  last_layer_self: bool
  degree_norm: bool
  dropout_rate: float

  def setup(self):
    """Initializes node embeddings, which cover the full graph's n_nodes."""
    self.node_embeddings = nn.Embed(
      num_embeddings=self.n_nodes,
      features=self.embedding_dim,
      embedding_init=jax.nn.initializers.glorot_uniform(),
    )

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple, is_training: bool) -> jax.Array:
    """Encodes the nodes of a graph into embeddings."""
    # Graph can be a subgraph and thus we use a subset of embeddings
    x = self.node_embeddings(graph.nodes["gid"])

    # First convolutional layer.
    x = SAGEConv(
      self.embedding_dim, with_self=True, degree_norm=self.degree_norm
    )(graph, x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(x)

    # Second convolutional layer.
    x = SAGEConv(
      self.embedding_dim,
      with_self=self.last_layer_self,
      degree_norm=self.degree_norm,
    )(graph, x)

    return x


class SAGEConv(nn.Module):
  """GraphSAGE convolutional layer with optional self-loops."""

  embedding_dim: int
  with_self: bool
  degree_norm: bool

  @nn.compact
  def __call__(self, graph: jraph.GraphsTuple, x) -> jax.Array:
    n_nodes = self.get_n_nodes(graph)

    # Add self-loops if enabled.
    if self.with_self:
      senders, receivers = self._add_self_edges(graph, n_nodes)
    else:
      senders, receivers = graph.senders, graph.receivers

    # Aggregate node features from neighbors.
    if not self.degree_norm:
      x_updated = jraph.segment_mean(
        x[senders], receivers, num_segments=n_nodes
      )
    else:

      def get_degree(n):
        return jax.ops.segment_sum(jnp.ones_like(senders), n, n_nodes)

      x_updated = self.normalize_by_degree(x, get_degree(senders))
      x_updated = jraph.segment_mean(
        x_updated[senders], receivers, num_segments=n_nodes
      )
      x_updated = self.normalize_by_degree(x_updated, get_degree(receivers))

    # Combine node and neighbor embeddings by concatenation.
    combined_embeddings = jnp.concatenate([x, x_updated], axis=-1)

    return nn.Dense(self.embedding_dim)(combined_embeddings)

  @staticmethod
  def _add_self_edges(
    graph: jraph.GraphsTuple, n_nodes: int
  ) -> tuple[jax.Array, jax.Array]:
    """Adds self-loops to the graph."""
    all_nodes = jnp.arange(n_nodes)
    senders = jnp.concatenate([graph.senders, all_nodes])
    receivers = jnp.concatenate([graph.receivers, all_nodes])
    return senders, receivers

  @staticmethod
  def normalize_by_degree(x: jax.Array, degree: jax.Array) -> jax.Array:
    """Normalizes node features by the square root of the degree."""
    # We set the the degree to a minimum of 1.
    return x * jax.lax.rsqrt(jnp.maximum(degree, 1.0))[:, None]

  @staticmethod
  def get_n_nodes(graph):
    """Returns the number of nodes in the graph in a jittable way."""
    return tree.tree_leaves(graph.nodes)[0].shape[0]


class LinkPredictor(nn.Module):
  """Predicts interaction scores for pairs of node embeddings."""

  embedding_dim: int
  n_layers: int
  dropout_rate: float

  @nn.compact
  def __call__(
    self,
    sender_embeddings: jax.Array,
    receiver_embeddings: jax.Array,
    is_training: bool,
  ) -> jax.Array:
    """Computes scores for node pairs."""
    x = sender_embeddings * receiver_embeddings  # Element-wise multiplication.

    # Apply MLP layers with ReLU activation and dropout.
    for _ in range(self.n_layers)[:-1]:
      x = nn.Dense(self.embedding_dim)(x)
      x = nn.relu(x)
      x = nn.Dropout(self.dropout_rate, deterministic=not is_training)(x)

    # Final output layer is a single neuron. Logit output used for binary link
    # classification.
    x = nn.Dense(1)(x)

    return jnp.squeeze(x)
