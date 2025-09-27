import os
import re
import sys
import argparse
from pathlib import Path
import pandas as pd

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------
# ICBHI (Respiratory Sound Database)
# ---------------------------
def build_icbhi_binary(
    rsd_root: Path,
    out_csv: Path,
    audio_subdir: str = "audio_and_txt_files",
    diag_csv_name: str = "patient_diagnosis.csv",
):
    audio_dir = rsd_root / audio_subdir
    diag_csv  = rsd_root / diag_csv_name

    if not audio_dir.exists():
        raise FileNotFoundError(f"[ICBHI] AUDIO_DIR not found: {audio_dir}")
    if not diag_csv.exists():
        raise FileNotFoundError(f"[ICBHI] DIAG_CSV not found: {diag_csv}")

    # Read patient_diagnosis robustly (with/without header)
    try:
        df = pd.read_csv(diag_csv)
    except Exception:
        df = pd.read_csv(diag_csv, header=None)

    # Normalize columns
    cols = [str(c).strip().lower() for c in df.columns]
    if len(cols) >= 2 and ("patient" in cols[0] or "id" in cols[0] or cols[0].isdigit()):
        pass  # looks ok
    if set(cols) >= {"patient", "diagnosis"}:
        df = df.rename(columns={cols[0]: "patient", cols[1]: "diagnosis"})
    else:
        # force two columns as patient/diagnosis
        if len(df.columns) < 2:
            raise RuntimeError(f"[ICBHI] Could not recognize columns in {diag_csv}. Got: {list(df.columns)}")
        df = df.iloc[:, :2]
        df.columns = ["patient", "diagnosis"]

    # Coerce patient id to string (e.g., "101")
    df["patient"] = df["patient"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()

    # Map to binary: COPD=1, else 0
    df["label"] = (df["diagnosis"].str.lower().str.contains("copd")).astype(int)

    pid_to_label = dict(zip(df["patient"], df["label"]))

    # Scan all wavs and link to patient via filename prefix (e.g., "101_1b1_Al_sc_Meditron.wav" → "101")
    wavs = list(audio_dir.rglob("*.wav"))
    rows = []
    for w in wavs:
        stem = w.name
        # patient id = leading digits before first underscore, or just consecutive digits from start
        m = re.match(r"^(\d+)_", stem)
        if m:
            pid = m.group(1)
        else:
            # fallback: take leading digits
            m2 = re.match(r"^(\d+)", stem)
            pid = m2.group(1) if m2 else None

        if pid is None:
            # skip files without obvious patient id
            continue

        label = pid_to_label.get(pid, None)
        if label is None:
            # patient not found in diagnosis CSV, skip or default to Non-COPD=0
            # here we skip to be safe:
            continue

        rows.append({"path": str(w.resolve()), "label": int(label), "patient_id": pid})

    if not rows:
        raise RuntimeError("[ICBHI] No labeled audio files collected. Check filename pattern and DIAG CSV.")

    out_df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    ensure_dir(out_csv)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[ICBHI] Wrote {len(out_df)} rows → {out_csv}")
    print(out_df["label"].value_counts())

# ---------------------------
# Fraiwan dataset (jwyy9np4gv-3)
# ---------------------------
def build_fraiwan_binary(
    fraiwan_root: Path,
    out_csv: Path,
    audio_subdir: str = "Audio Files",
    diag_xlsx_name: str = "Data annotation.xlsx",
    use_excel: bool = False,
):
    """
    Two strategies:

    1) Default (robust): label from filename substring:
          label = 1 if 'COPD' appears in the filename (case-insensitive), else 0 (Non-COPD).
       This works with your folder where names contain 'COPD', 'Asthma', 'N', etc.

    2) Optional: use Excel if a *filename* column is added. Many public copies
       of this sheet do NOT include a filename column, so we keep it off by default.
    """
    audio_dir = fraiwan_root / audio_subdir
    diag_xlsx = fraiwan_root / diag_xlsx_name

    if not audio_dir.exists():
        raise FileNotFoundError(f"[Fraiwan] AUDIO_DIR not found: {audio_dir}")
    if use_excel and not diag_xlsx.exists():
        raise FileNotFoundError(f"[Fraiwan] Data annotation.xlsx not found: {diag_xlsx}")

    wavs = list(audio_dir.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"[Fraiwan] No .wav files found under {audio_dir}")

    rows = []

    if use_excel:
        # Try to read and find a filename-like column → not present in your sheet,
        # so this will raise unless you add it. Kept for future flexibility.
        df = pd.read_excel(diag_xlsx)
        df_cols = {c.lower(): c for c in df.columns}
        name_col = None
        for k in ["file", "filename", "recording", "recording name", "audio", "wav", "path"]:
            if k in df_cols:
                name_col = df_cols[k]
                break
        if not name_col:
            raise RuntimeError(
                "[Fraiwan] Excel has no filename column. "
                "Please add a column named 'Recording name' (exact file names)."
            )
        label_col = df_cols.get("diagnosis") or df_cols.get("label")
        if not label_col:
            raise RuntimeError("[Fraiwan] Could not find 'Diagnosis' column in Excel.")

        df["__label__"] = (df[label_col].astype(str).str.lower().str.contains("copd")).astype(int)
        name_to_label = {}
        for _, r in df[[name_col, "__label__"]].dropna().iterrows():
            name_to_label[str(r[name_col]).strip()] = int(r["__label__"])

        for w in wavs:
            name = w.name
            label = name_to_label.get(name, None)
            if label is None:
                # try match by stem if extension differs
                label = name_to_label.get(Path(name).stem, None)
            # final fallback → filename substring
            if label is None:
                label = int("copd" in name.lower())
            rows.append({"path": str(w.resolve()), "label": label})
    else:
        # Simple & effective: parse from filename text
        for w in wavs:
            name = w.name
            label = int("copd" in name.lower())  # COPD=1, everything else Non-COPD=0
            rows.append({"path": str(w.resolve()), "label": label})

    out_df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    ensure_dir(out_csv)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[Fraiwan] Wrote {len(out_df)} rows → {out_csv}")
    print(out_df["label"].value_counts())

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Build binary COPD vs Non-COPD label CSVs for ICBHI and Fraiwan datasets.")
    # ICBHI paths
    ap.add_argument("--rsd_root", default=r"F:\COPD Research\Respiratory Sound Database",
                    help="Root of ICBHI/Respiratory Sound Database")
    ap.add_argument("--rsd_audio_subdir", default="audio_and_txt_files")
    ap.add_argument("--rsd_diag_csv_name", default="patient_diagnosis.csv")
    ap.add_argument("--rsd_out_csv", default="artifacts/splits/icbhi_binary.csv")

    # Fraiwan paths
    ap.add_argument("--fraiwan_root", default=r"F:\COPD Research\jwyy9np4gv-3")
    ap.add_argument("--fraiwan_audio_subdir", default="Audio Files")
    ap.add_argument("--fraiwan_diag_xlsx", default="Data annotation.xlsx")
    ap.add_argument("--fraiwan_out_csv", default="artifacts/splits/fraiwan_binary.csv")
    ap.add_argument("--fraiwan_use_excel", action="store_true",
                    help="Use Excel to map labels (requires a filename column).")

    args = ap.parse_args()

    # ICBHI
    build_icbhi_binary(
        rsd_root=Path(args.rsd_root),
        out_csv=Path(args.rsd_out_csv),
        audio_subdir=args.rsd_audio_subdir,
        diag_csv_name=args.rsd_diag_csv_name,
    )

    # Fraiwan
    build_fraiwan_binary(
        fraiwan_root=Path(args.fraiwan_root),
        out_csv=Path(args.fraiwan_out_csv),
        audio_subdir=args.fraiwan_audio_subdir,
        diag_xlsx_name=args.fraiwan_diag_xlsx,
        use_excel=args.fraiwan_use_excel,
    )

if __name__ == "__main__":
    main()
