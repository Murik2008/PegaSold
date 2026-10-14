"""Loading, cleaning, splitting and labelling the bina.az sale data.

Every statistic used to transform the data is measured on the TRAINING
SPLIT ONLY, so nothing leaks from validation or test into the model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42 #for reproducibility
DATA_PATH = "data/bina_az_sale.csv"
OUTPUT_DIR = "data/processed"

LEAKAGE_COLUMNS = ["total_price", "unit_price","description"] #leakage columns

def load(path: str|Path = DATA_PATH) -> pd.DataFrame:
    #Read the raw scv
    df = pd.read_csv(path, low_memory=False)
    print(f"  [1] loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df






