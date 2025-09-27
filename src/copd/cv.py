import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

def build_folds(csv_path, n_splits=5, seed=42):
    df = pd.read_csv(csv_path)
    cols = {c.lower():c for c in df.columns}
    path_col = cols.get("path") or cols.get("filepath")
    label_col = cols.get("label")
    pid_col = cols.get("patient_id")

    y = df[label_col].astype(int).values
    if pid_col is not None:
        groups = df[pid_col].astype(str).values
        gkf = GroupKFold(n_splits)
        idx = [(tr,val) for tr,val in gkf.split(df, y, groups=groups)]
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        idx = [(tr,val) for tr,val in skf.split(df, y)]
    return df, idx
