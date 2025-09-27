import argparse
import torch
import pandas as pd
from src.copd.features import FeatureExtractor
from src.copd.old_models import build_model
from src.copd.data import AudioBinaryDataset
from src.copd.utils import device_str

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)  # paths only OK; label ignored
    ap.add_argument("--model_name", default="aerocpdnet")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features", default="mel", choices=["mel","mfcc"])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    args=ap.parse_args()

    dev = device_str()
    ds = AudioBinaryDataset(args.csv, sample_rate=args.sr, duration=args.duration, return_path=True)
    feat = FeatureExtractor(kind=args.features, sr=args.sr)
    model = build_model(args.model_name, 1, 1).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd); model.eval()

    paths=[]; probs=[]
    for i in range(len(ds)):
        wav, lab, path = ds[i]
        x = torch.from_numpy(wav).unsqueeze(0).to(dev)
        X = feat(x).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(model(X)).item()
        paths.append(path); probs.append(p)
    pd.DataFrame({"path":paths, "prob_copd":probs}).to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")

if __name__=="__main__":
    main()
