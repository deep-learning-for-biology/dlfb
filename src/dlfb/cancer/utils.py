from flax.traverse_util import flatten_dict, unflatten_dict


def decay_mask(params):
  """Creates a weight decay mask that excludes biases and norm layers."""
  flat_params = flatten_dict(params)
  mask = {}
  for path, _ in flat_params.items():
    key = "/".join(path).lower()
    if "bias" in key or "bn" in key or "batchnorm" in key:
      mask[path] = False
    else:
      mask[path] = True
  return unflatten_dict(mask)
