import os


def assets(subdir: str | None = None) -> str:
  ensure_context()
  assets_dir = os.environ["ASSETS_DIR"]
  if subdir:
    assets_dir += f"/{subdir}"
  return assets_dir


def ensure_context() -> None:
  context = detect_context()
  assets_dir = {
    "local": "/content/drive/MyDrive/dlfb/assets",
    "colab": "/content/assets",
  }[context]
  if not os.path.exists(assets_dir):
    raise FileNotFoundError("Could not find the assets directory.")
  os.environ["ASSETS_DIR"] = assets_dir


def detect_context() -> str:
  is_colab = any(env_var.startswith("COLAB_") for env_var in os.environ)
  return "colab" if is_colab else "local"
