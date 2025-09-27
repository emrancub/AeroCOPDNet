# scripts/merge_to_pooled.py
import argparse, os, re
import pandas as pd

def _read_binary_csv(p, dataset_name):
    df = pd.read_csv(p)
    # normalize col names
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("path") or cols.get("filepath")
    if path_col is None:
        raise RuntimeError(f"{p} must have a 'path' (or 'filepath') column.")
    df = df.rename(columns={path_col: "path"})
    if "label" not in df.columns:
        raise RuntimeError(f"{p} must have a 'label' column (0/1).")

    # force absolute paths (Windows-safe)
    df["path"] = df["path"].apply(lambda x: os.path.abspath(os.path.expanduser(str(x))))
    # force int labels {0,1}
    df["label"] = df["label"].astype(int).clip(0, 1)
    df["dataset"] = dataset_name

    # best-effort group_id to help with patient-level analysis later
    if "patient_id" in df.columns:
        df["group_id"] = df["patient_id"].astype(str)
    else:
        if dataset_name == "icbhi":
            # filenames like 101_1b1_Al_sc_Meditron.wav -> group_id=101
            df["group_id"] = df["path"].apply(lambda s: re.match(r".*[\\/](\d+)_", s).group(1) if re.match(r".*[\\/](\d+)_", s) else "unk")
        elif dataset_name == "fraiwan":
            # filenames like BP123_... -> group_id=BP123
            df["group_id"] = df["path"].apply(lambda s: re.match(r".*[\\/](BP\d+)_", s).group(1) if re.match(r".*[\\/](BP\d+)_", s) else "unk")
        else:
            df["group_id"] = "unk"
    return df[["path","label","dataset","group_id"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icbhi_csv", required=True)
    ap.add_argument("--fraiwan_csv", required=True)
    ap.add_argument("--out_csv", default="artifacts/splits/pooled_icbhi_fraiwan_binary.csv")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate paths if present.")
    args = ap.parse_args()

    icbhi = _read_binary_csv(args.icbhi_csv, "icbhi")
    fraiwan = _read_binary_csv(args.fraiwan_csv, "fraiwan")
    pooled = pd.concat([icbhi, fraiwan], ignore_index=True)

    if args.dedupe:
        pooled = pooled.drop_duplicates(subset=["path"])

    # keep only rows whose files exist
    pooled["exists"] = pooled["path"].apply(os.path.exists)
    missing = pooled[~pooled["exists"]]
    if len(missing):
        print(f"[WARN] {len(missing)} rows dropped (files missing). First few:\n{missing.head()}")
        pooled = pooled[pooled["exists"]].drop(columns=["exists"])
    else:
        pooled = pooled.drop(columns=["exists"])

    # quick report
    print("Counts by dataset and label:")
    print(pooled.groupby(["dataset","label"]).size())

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pooled.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {len(pooled)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
