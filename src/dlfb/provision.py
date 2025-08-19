import os
import textwrap

import ipinfo
import typer
from google.cloud.storage import Client, transfer_manager

from dlfb.log import log
from dlfb.utils import mkdir_p


app = typer.Typer(no_args_is_help=True)


@app.command()
def cli_provision_assets(
  chapter: str = typer.Option(
    ..., help="Name of the chapter dataset to download."
  ),
  bucket_name: str = typer.Option(
    "dlfb-assets", help="Name of public bucket hosting datasets."
  ),
  destination: str = typer.Option(
    "/content/assets",
    help="Destination of the dataset download (note trailing slash).",
  ),
  models: bool = typer.Option(
    True, help="Whether to download pretrained models."
  ),
  workers: int = typer.Option(
    4, help="Number of files to download concurrently."
  ),
  chunk_size: int = typer.Option(
    512 * 1024 * 1024, help="Chunk size of larger file downloads (512MB)."
  ),
):
  """Download the required dataset and models for a chapter."""
  provision_assets(
    chapter, bucket_name, destination, models, workers, chunk_size
  )


def provision_assets(
  chapter, bucket_name, destination, models, workers, chunk_size
) -> None:
  warn_about_slow_transfers()
  storage_client = Client()
  bucket = storage_client.bucket(bucket_name)
  provision_datasets(chapter, bucket, destination, workers, chunk_size)
  if models:
    provision_models(chapter, bucket, destination, workers)


def warn_about_slow_transfers():
  # NOTE: inspired by https://github.com/googlecolab/colabtools/issues/1722
  client_details = ipinfo.getHandler().getDetails()
  if client_details.country != "UK":
    log.warning(
      textwrap.dedent(
        f"""
        Your currently allocated Google Colab VM is not located in the same
        region as where the datasets are stored. This usually means that data
        access will be substantially slower (~20 min up from ~5 min). If
        possible, reset your runtime in the menu to request a new VM and try
        again or grab a coffee and read ahead.

        Current location: {client_details.city} ({client_details.country_name})
        """
      )
    )


def provision_datasets(chapter, bucket, destination, workers, chunk_size):
  blobs = bucket.list_blobs(prefix=f"{chapter}/datasets")
  match chapter:
    case "localization":
      # NOTE: this chapter has a few large files
      download_large_files_from_gcs(blobs, destination, workers, chunk_size)
    case _:
      download_many_files_from_gcs(blobs, bucket, destination, workers)


def provision_models(chapter, bucket, destination, workers):
  blobs = bucket.list_blobs(prefix=f"{chapter}/models")
  download_many_files_from_gcs(blobs, bucket, destination, workers)


def download_large_files_from_gcs(
  blobs, destination: str, workers: int, chunk_size: int
):
  # NOTE see: https://cloud.google.com/storage/docs/sliced-object-downloads
  for blob in blobs:
    filename = f"{destination}/{blob.name}"
    if not os.path.exists(filename):
      mkdir_p(os.path.dirname(filename))
      transfer_manager.download_chunks_concurrently(
        blob,
        filename,
        chunk_size=chunk_size,
        max_workers=workers,
      )


def download_many_files_from_gcs(blobs, bucket, destination: str, workers: int):
  transfer_manager.download_many_to_path(
    bucket,
    [blob.name for blob in blobs],
    destination_directory=f"{destination}/",
    max_workers=workers,
    skip_if_exists=True,
  )
