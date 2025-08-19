import json
import os
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image

from dlfb.cancer.dataset import Dataset, Images
from dlfb.cancer.dataset.preprocessors import crop


class DatasetBuilder:
  """Builds a dataset with metadata, loaded images, and class mappings."""

  def __init__(self, data_dir: str, out_dir: str | None = None) -> None:
    self.data_dir = data_dir
    self.out_dir = out_dir or data_dir

  def build(
    self,
    rng: jax.Array,
    splits: dict[str, float],
    preprocessors: list[Callable] = [crop],
    image_size: tuple[int, int, int] = (224, 224, 3),
    class_map: dict[str, Any] | None = None,
  ) -> dict[str, Dataset]:
    """Builds the dataset splits from loaded metadata and loaded images."""
    metadata = MetadataLoader(self.data_dir, self.out_dir).load(class_map)
    images = ImageLoader(metadata, self.out_dir).load(preprocessors, image_size)

    # Shuffle the dataset and assign each example to one of the dataset splits.
    num_samples = len(metadata)
    rng, rng_perm = jax.random.split(rng, 2)
    shuffled_indices = jax.random.permutation(rng_perm, num_samples)

    # Create each dataset split using the shuffled indices and store the
    # corresponding metadata and image data in a Dataset object.
    dataset_splits, start = {}, 0
    for name, size in self._get_split_sizes(splits, num_samples):
      indices = np.array(shuffled_indices[start : (start + size)])
      dataset_splits[name] = Dataset(
        metadata=metadata.iloc[indices],
        images=images,
        num_classes=metadata["class"].nunique(),
      )
      start += size
    return dataset_splits

  def _get_split_sizes(self, splits: dict[str, float], num_samples: int):
    """Convert fractional split sizes to integer counts that sum to total."""
    names = list(splits)
    sizes = [int(num_samples * splits[name]) for name in names[:-1]]
    sizes.append(num_samples - sum(sizes))  # last split gets the remainder
    yield from zip(names, sizes)


class MetadataLoader:
  """Loads and processes metadata for image classification."""

  def __init__(self, data_dir: str, out_dir: str | None = None):
    """Initializes loader with input and output directories."""
    self.data_dir = data_dir
    self.out_dir = out_dir or data_dir
    self.outfile = f"{self.out_dir}/metadata.csv"
    self.metadata: pd.DataFrame

  def load(self, class_map) -> pd.DataFrame:
    """Loads metadata from a CSV file or scans image files if missing."""
    if not os.path.exists(self.outfile):
      self._scan_image_dir()
    self.metadata = pd.read_csv(self.outfile)
    self._apply_labels(class_map)
    return self.metadata

  def _scan_image_dir(self) -> None:
    """Scans image directory and writes metadata (split, class, path) to CSV."""
    found = []
    for path in sorted(
      Path(self.data_dir).rglob("*.jpg"), key=lambda p: str(p)
    ):
      split, class_name, _ = path.parts[-3:]
      found.append(
        {"split_orig": split, "class_orig": class_name, "full_path": str(path)}
      )
    return (
      pd.DataFrame(found)
      .rename_axis("frame_id")
      .reset_index()
      .to_csv(self.outfile, index=False)
    )

  def _apply_labels(self, class_map) -> None:
    """Applies class mapping to assign numeric labels and merged class names."""

    class_map = self._ensure_class_map(class_map)
    self.metadata["label"] = [
      class_map["mapping"][c] for c in self.metadata["class_orig"]
    ]
    # We now refer to records according to their new merged classes.
    self.metadata["class"] = [
      class_map["names"][lb] for lb in self.metadata["label"]
    ]

  def _ensure_class_map(self, class_map) -> dict[str, dict]:
    """Creates a default class map if none is provided."""
    if not class_map:
      unique_classes = self.metadata["class_orig"].unique()
      class_map = {
        "mapping": {c: i for i, c in enumerate(unique_classes)},
        "names": {i: c for i, c in enumerate(unique_classes)},
      }
    return class_map


class ImageLoader:
  """Loads and preprocesses image data into memory-mapped arrays."""

  def __init__(self, metadata: pd.DataFrame, out_dir: str):
    """Initializes loader with metadata and output directory."""
    self.metadata = metadata
    self.outdir = out_dir
    self.raw_image_file = f"{out_dir}/images_raw.bin"
    self.shapes_file = f"{out_dir}/image_shapes.json"

  def load(
    self, preprocessors: list[Callable], image_size: tuple[int, int, int]
  ) -> Images:
    """Processes and stores images as memory-mapped arrays for access."""
    if not os.path.exists(self.raw_image_file):
      self._assemble_raw_images()
    return self._preprocess(preprocessors, image_size)

  def _assemble_raw_images(self) -> None:
    """Assembles and stores raw images as a flat memory-mapped array."""
    if not os.path.exists(self.shapes_file):
      self._store_image_dims()
    _, sizes, offsets = self._load_image_dims()
    memmap = np.memmap(
      self.raw_image_file, mode="w+", dtype=np.uint8, shape=(sum(sizes),)
    )
    for i, path in enumerate(self.metadata["full_path"]):
      image = Image.open(path).convert("RGB")
      flat_image = np.array(image, dtype=np.uint8).flatten()
      memmap[offsets[i] : (offsets[i] + flat_image.size)] = flat_image
    memmap.flush()

  def _store_image_dims(self) -> None:
    """Checks all image files for their size to infer memmap size."""
    shapes = []
    for path in self.metadata["full_path"]:
      with Image.open(path) as img:
        w, h = img.size  # This does not load the image
        shape = (h, w, 3)
        shapes.append(shape)
    sizes = np.array([np.prod(shape) for shape in shapes])
    offsets = np.concatenate([[0], np.cumsum(sizes[:-1])])
    with open(self.shapes_file, "w") as f_out:
      json.dump(
        {
          "shapes": shapes,
          "sizes": sizes.tolist(),
          "offsets": offsets.tolist(),
        },
        f_out,
      )

  def _load_image_dims(self):
    """Loads shape and offset metadata from JSON."""
    with open(self.shapes_file, "r") as f_out:
      dims = json.load(f_out)
    return dims["shapes"], dims["sizes"], dims["offsets"]

  def _load_raw_images(self) -> np.memmap:
    """Loads the flat memory-mapped raw image array."""
    _, sizes, _ = self._load_image_dims()
    return np.memmap(
      self.raw_image_file, mode="r", dtype=np.uint8, shape=(sum(sizes),)
    )

  def _preprocess(
    self, preprocessors: list[Callable], image_size: tuple[int, int, int]
  ) -> Images:
    """Applies preprocessing to raw images and stores as memory-mapped array."""
    preprocessed = {}
    for fn in preprocessors:
      out_file = f"{self.outdir}/images_{fn.__name__}.npy"
      if not os.path.exists(out_file):
        self._apply_preprocessing_fn(out_file, fn, image_size)
      memmap = np.memmap(out_file, mode="r", **self.memmap_config(image_size))
      preprocessed.update({fn.__name__: memmap})
    return Images(loaded=preprocessed, size=image_size)

  def _apply_preprocessing_fn(self, out_file, fn, image_size):
    """Fill memmap with processed images."""
    raw = self._load_raw_images()
    shapes, _, offsets = self._load_image_dims()
    memmap = np.memmap(out_file, mode="w+", **self.memmap_config(image_size))
    for i, (shape, offset) in enumerate(zip(shapes, offsets)):
      raw_image = self._get_raw_image(raw, tuple(shape), offset)
      memmap[i, :, :, :] = fn(raw_image)
    memmap.flush()

  def _get_raw_image(self, raw, shape, offset) -> jax.Array:
    """Extracts and reshapes a raw image from memmap storage."""
    size = np.prod(shape)
    flat = raw[offset : (offset + size)]
    image = flat.reshape(shape)
    image = jnp.array(image, dtype=jnp.float32)
    return image

  def memmap_config(self, image_size):
    """Returns memmap config dictionary for given image size."""
    return {
      "dtype": jnp.float32,
      "shape": (self.metadata.shape[0], *image_size),
    }
