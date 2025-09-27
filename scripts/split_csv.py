# scripts/split_csv.py
import argparse, os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):5d} -> {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with columns path(or filepath), label (and optional patient_id)")
    ap.add_argument("--out_prefix", required=True, help="Prefix for outputs, e.g. artifacts/splits/icbhi_binary")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_col", default=None, help="e.g., patient_id for patient-wise splitting")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    path_col = "path" if "path" in df.columns else "filepath"
    if path_col not in df.columns or "label" not in df.columns:
        raise SystemExit(f"CSV must have columns 'path' (or 'filepath') and 'label'. Got {df.columns.tolist()}")

    if args.group_col and args.group_col in df.columns:
        groups = df[args.group_col].astype(str).values
        gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
        idx_tr, idx_te = next(gss_test.split(df, df["label"], groups))
        df_tr = df.iloc[idx_tr].reset_index(drop=True)
        df_te = df.iloc[idx_te].reset_index(drop=True)

        adj_val = args.val_frac / (1.0 - args.test_frac)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val, random_state=args.seed)
        idx_tr2, idx_va = next(gss_val.split(df_tr, df_tr["label"], df_tr[args.group_col]))
        df_tr, df_va = df_tr.iloc[idx_tr2].reset_index(drop=True), df_tr.iloc[idx_va].reset_index(drop=True)
    else:
        # stratified by label
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
        y = df["label"].values
        idx_tr, idx_te = next(sss1.split(df, y))
        df_tr, df_te = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)

        adj_val = args.val_frac / (1.0 - args.test_frac)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adj_val, random_state=args.seed)
        y_tr = df_tr["label"].values
        idx_tr2, idx_va = next(sss2.split(df_tr, y_tr))
        df_tr, df_va = df_tr.iloc[idx_tr2].reset_index(drop=True), df_tr.iloc[idx_va].reset_index(drop=True)

    save(df_tr, f"{args.out_prefix}_train.csv")
    save(df_va, f"{args.out_prefix}_val.csv")
    save(df_te, f"{args.out_prefix}_test.csv")

if __name__ == "__main__":
    main()
