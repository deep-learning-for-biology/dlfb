import os

import numpy as np
import pandas as pd

from dlfb.localization.constants import DATASET_BLOCKS
from dlfb.localization.dataset import Images, Labels
from dlfb.log import log


class DataLoader:
  data_path: str
  file: str

  def is_assembled(self):
    status = False
    if os.path.exists(self.file):
      if os.stat(self.file).st_size > 0:
        status = True
    return status

  @staticmethod
  def get_blocks():
    return [f"{i:02d}" for i in range(DATASET_BLOCKS)]


class LabelLoader(DataLoader):
  LABEL_DATA_PREFIX = "Label_data"

  def __init__(
    self,
    data_path: str,
    file: str = "labels.npy",
  ):
    self.data_path = data_path
    self.file = f"{data_path}/{file}"

  def load(self, force_recreate: bool = False) -> Labels:
    if not self.is_assembled() or force_recreate:
      log.info("Labels to be assembled from blocks...")
      self.assemble()
    return Labels(lookup=self._load_from_disk())

  def assemble(self) -> None:
    labels: list[np.ndarray] = []
    for block in self.get_blocks():
      log.info(f"Processing block{block}")
      labels.append(self._load_labels_block(block))
    np.save(self.file, np.concatenate(labels))

  def _load_labels_block(self, block: str) -> np.ndarray:
    return pd.read_csv(
      f"{self.data_path}/{self.LABEL_DATA_PREFIX}{block}.csv"
    ).to_numpy()

  def _load_from_disk(self) -> pd.DataFrame:
    lookup = pd.DataFrame(
      np.load(self.file, allow_pickle=True),
      columns=[
        "ensembl_id",
        "gene_symbol",
        "loc_grade1",
        "loc_grade2",
        "loc_grade3",
        "protein_id",
        "fov_id",
      ],
    ).reset_index(names="frame_id")
    return lookup


class ImageLoader(DataLoader):
  IMAGE_DATA_PREFIX = "Image_data"
  CHANNELS = {"pro": 0}

  def __init__(
    self,
    data_path: str,
    file: str = "images.npy",
  ):
    self.data_path = data_path
    self.file = f"{data_path}/{file}"

  def load(self, force_recreate: bool = False) -> Images:
    if not self.is_assembled() or force_recreate:
      log.info("Images were not yet assembled from blocks...")
      self.assemble()
    return Images(frames=self._load_on_disk())

  def assemble(self) -> None:
    total_frames = self._get_total_frames_across_blocks()
    images_on_disk = np.memmap(
      self.file,
      dtype="float32",
      mode="w+",
      shape=(total_frames, 100, 100, 1),
    )
    start = 0
    for block in self.get_blocks():
      log.info(f"Processing block{block}")
      images_block_on_disk = np.load(
        f"{self.data_path}/{self.IMAGE_DATA_PREFIX}{block}.npy",
        mmap_mode="r",
      )
      end = start + images_block_on_disk.shape[0]
      images_on_disk[start:end, :, :, :] = images_block_on_disk[
        :, :, :, :1  # NOTE: we pluck channel 0 (with ':1')
      ]
      images_on_disk.flush()
      start = end

  def _get_total_frames_across_blocks(self) -> int:
    frames_per_block = [shape[0] for shape in self._get_block_shapes()]
    log.debug(f"Frames across blocs: {frames_per_block}")
    total_frames = sum(frames_per_block)
    log.debug(f"Total Frames across blocs: {total_frames}")
    return total_frames

  def _get_block_shapes(self) -> list[tuple[int, int, int, int]]:
    return [
      np.load(
        f"{self.data_path}/{self.IMAGE_DATA_PREFIX}{block}.npy",
        mmap_mode="r",
      ).shape
      for block in self.get_blocks()
    ]

  def _load_on_disk(self) -> np.ndarray:
    # NOTE: full dataset contains 1134592 images
    return np.memmap(self.file, dtype="float32", mode="r").reshape(
      -1, 100, 100, 1
    )
