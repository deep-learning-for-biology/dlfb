import os
from typing import List, Tuple
from urllib.parse import urljoin

from parfive import Downloader, SessionConfig
import requests
import typer


app = typer.Typer(no_args_is_help=True)


@app.command()
def cli_provision_assets(
  chapter: str = typer.Option(
    ..., help="Name of the chapter dataset to download."
  ),
  base_url: str = typer.Option(
    "https://deep-learning-for-biology.com",
    help="Public URL where data is hosted."
  ),
  destination: str = typer.Option(
    "/content/assets",
    help="Destination of the dataset download.",
  ),
  models: bool = typer.Option(
    True, help="Whether to download pretrained models."
  ),
  chunk: int = typer.Option(
    256 * 1024 * 1024, help="Chunk size of larger file downloads."
  ),
):
  """Download the required dataset and models for a chapter."""
  provision_assets(chapter, base_url, destination, models, chunk)


def provision_assets(chapter, base_url, destination, models, chunk) -> None:
  sized_prefixes = get_sized_prefixes(base_url, chapter, models)
  dl_small = Downloader(
    max_conn=64,
    max_splits=1,
    progress=True,
    config=SessionConfig(file_progress=False)
  )
  dl_big = Downloader(max_conn=4, max_splits=8, progress=True)
  for size, prefix in sized_prefixes:
    url = urljoin(base_url, prefix)
    dest = os.path.dirname(os.path.join(destination, prefix))
    os.makedirs(dest, exist_ok=True)
    if size <= chunk:
      dl_small.enqueue_file(url=url, path=dest)
    else:
        dl_big.enqueue_file(url=url, path=dest)
  if dl_small.queued_downloads:
    dl_small.download()
  if dl_big.queued_downloads:
    dl_big.download()


def get_sized_prefixes(base_url: str, chapter: str, models: bool) -> List[Tuple[int, str]]:
  r = requests.get(urljoin(base_url, "urls.txt"), timeout=60)
  r.raise_for_status()
  sized_prefixes: List[Tuple[int, str]] = []
  for raw in r.text.splitlines():
    line = raw.strip()
    parts = line.split(None, 1)
    size, prefix = int(parts[0]), parts[1]
    c, k, _ = prefix.split("/", 2)
    if c == chapter and (k == "datasets" or (models and k == "models")):
      sized_prefixes.append((size, prefix))
  return sized_prefixes
