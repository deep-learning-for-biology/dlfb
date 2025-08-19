import textwrap
from pathlib import Path

import jax.numpy as jnp

from dlfb.log import log


def mkdir_p(path):
  try:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
  except OSError as error:
    log.error(f"Failed to create directory '{path}': {error}")


ROMAN = [
  (1000, "M"),
  (900, "CM"),
  (500, "D"),
  (400, "CD"),
  (100, "C"),
  (90, "XC"),
  (50, "L"),
  (40, "XL"),
  (10, "X"),
  (9, "IX"),
  (5, "V"),
  (4, "IV"),
  (1, "I"),
]


def int_to_roman(number):
  # NOTE: see -- https://stackoverflow.com/a/47713392
  result = []
  for arabic, roman in ROMAN:
    (factor, number) = divmod(number, arabic)
    result.append(roman * factor)
    if number == 0:
      break
  return "".join(result)


def groom(label: str) -> str:
  return label.replace("_", " ").title()


def validate_splits(splits: dict[str, float]) -> None:
  """Make sure the split fractions sum up to 1."""
  total = sum(splits.values())
  if not jnp.isclose(total, 1.0):
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def wrap_text(text, width):
  return "\n".join(textwrap.wrap(text, width))
