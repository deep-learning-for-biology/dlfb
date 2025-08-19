import matplotlib.pyplot as plt
import pandas as pd


def configure_figures_and_table_display():
  plt.rcParams["figure.dpi"] = 300
  pd.set_option("display.max_colwidth", 20)
  pd.set_option("display.max_columns", 10)
  pd.set_option("display.max_rows", 6)
  pd.set_option("display.min_rows", 6)
