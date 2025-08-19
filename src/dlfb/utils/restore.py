from functools import wraps
from pathlib import Path

import jax
import orbax.checkpoint as ocp

from dlfb.log import log


def restore(store_path, state):
  with ocp.CheckpointManager(Path(store_path).resolve()) as mngr:
    state, metrics = restore_checkpoint(mngr, state)
  return state, metrics


def store(store_path, state, metrics) -> None:
  with ocp.CheckpointManager(Path(store_path).resolve()) as mngr:
    save_final_checkpoint(mngr, state, metrics)


def restorable(train_fn):
  # NOTE: wraps required for proper display with 'inspect'
  @wraps(train_fn)
  def wrapper(state, store_path: str | None = None, **kwargs):
    if store_path:
      mngr = ocp.CheckpointManager(Path(store_path).resolve())
      try:
        state, metrics = restore_checkpoint(mngr, state)
      except FileNotFoundError:
        log.debug("Training new model checkpointing state and metrics...")
        state, metrics = train_fn(state, **kwargs)
        save_final_checkpoint(mngr, state, metrics)
    else:
      log.debug("Training new model without checkpointing...")
      state, metrics = train_fn(state, **kwargs)
    return state, metrics

  return wrapper


def restore_checkpoint(mngr: ocp.CheckpointManager, state):
  # NOTE: benefit of approach is that we can reconstruct a model
  #       built on a GPU back in a CPU environment.
  #       see: https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpointing_pytrees.html#change-sharding
  restored = mngr.restore(
    0,
    args=ocp.args.Composite(
      state=ocp.args.StandardRestore(get_abstract_state(state)),
      extra_metadata=ocp.args.JsonRestore(),
    ),
  )
  log.debug("Train state and metrics restored from checkpoint")
  state, metrics = restored.state, restored.extra_metadata
  return state, metrics


def get_abstract_state(state):
  """Returns a shape/dtype-only version of the model's state"""
  return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)


def save_final_checkpoint(mngr, state, metrics):
  mngr.save(
    0,
    args=ocp.args.Composite(
      state=ocp.args.StandardSave(state),
      extra_metadata=ocp.args.JsonSave(metrics),
    ),
  )
  mngr.wait_until_finished()
