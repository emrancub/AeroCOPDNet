# scripts/merge_csvs.py
import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("csvs", nargs="+", help="Two or more CSVs to merge")
    args = ap.parse_args()

    dfs = []
    for c in args.csvs:
        df = pd.read_csv(c)
        if "path" not in df.columns and "filepath" in df.columns:
            df = df.rename(columns={"filepath":"path"})
        missing = {"path","label"} - set(df.columns)
        if missing:
            raise SystemExit(f"{c} is missing {missing}")
        if "dataset" not in df.columns:
            df["dataset"] = os.path.splitext(os.path.basename(c))[0]
        dfs.append(df[["path","label","dataset"]])

    out = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved {len(out)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
