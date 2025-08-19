from dlfb.localization.dataset import Dataset
from dlfb.localization.dataset.loaders import ImageLoader, LabelLoader
from dlfb.utils import groom


def get_dataset(data_path: str) -> Dataset:
  dataset = Dataset(
    images=ImageLoader(data_path).load(), labels=LabelLoader(data_path).load()
  )
  return dataset


def count_unique_proteins(dataset: dict[str, Dataset]) -> int:
  protein_ids: set[int] = set()
  for dataset_split in dataset.values():
    protein_ids.update(dataset_split.labels.lookup["protein_id"].unique())
  return len(protein_ids)


def summarize_localization(group) -> str:
  top_grade = group["grade"].min()
  selected = group[group["grade"] == top_grade]["localization"].tolist()
  groomed = [groom(s) for s in selected]
  base = " & ".join(groomed)
  n_hidden = len(group) - len(selected)
  return f"{base} (+{n_hidden})" if n_hidden > 0 else base
