# scripts/ablation.py
import argparse, itertools, subprocess, json, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model", default="aerocpdnet")
    ap.add_argument("--features", default="mel")
    args = ap.parse_args()

    settings = []
    for use_specaug, mixup, balance in itertools.product([0,1],[0.0,0.2],[0,1]):
        settings.append({
            "label": f"spec{use_specaug}_mix{mixup}_bal{balance}",
            "use_specaug": use_specaug, "mixup": mixup, "no_balance": (1-balance)
        })

    outdir = Path("artifacts/ablation"); outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for s in settings:
        cmd = [
          "python","-m","scripts.cv_train",
          "--csv", args.csv, "--folds", str(args.folds),
          "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
          "--model", args.model, "--features", args.features
        ]
        if s["use_specaug"]: cmd += ["--use_specaug"]
        if s["no_balance"]:  cmd += ["--no_balance"]
        cmd += ["--mixup", str(s["mixup"])]
        print(">>>", " ".join(cmd))
        env = os.environ.copy()
        res = subprocess.run(cmd, capture_output=True, text=True)
        (outdir / f"{s['label']}_stdout.txt").write_text(res.stdout + "\n\n" + res.stderr)

        # read the summary written by cv_train of this run (last created dir)
        # simplest: find most recent cv_old directory and copy summary
        cv_dirs = sorted(Path("artifacts/cv_old").glob("cv_*"), key=os.path.getmtime)
        summ = (cv_dirs[-1] / "summary.csv").read_text()
        (outdir / f"{s['label']}_summary.csv").write_text(summ)
        rows.append({"setting":s["label"], "summary_path":str((outdir / f"{s['label']}_summary.csv").resolve())})

    print("Ablation summaries written to:", outdir)

if __name__ == "__main__":
    main()
