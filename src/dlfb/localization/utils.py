import numpy as np


def ensure_empty_json_file(file_path):
  try:
    with open(file_path, "x") as file:
      file.write("{}")
  except FileExistsError:
    print(f"The file '{file_path}' already exists.")


def calculate_grid_dimensions(n, ratio=1):
  num_cols = int(np.ceil(np.sqrt(n) * ratio))
  num_rows = int(np.ceil(n / num_cols))
  return num_rows, num_cols
