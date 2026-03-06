# 🔄 Updated Implementation (Post-Review)

A revised and fully reproducible implementation of **AeroCOPDNet** is available in a new repository.  
This rebuild incorporates post-review updates, clarified architecture, ablation studies, and refined evaluation protocols.

👉 **AeroCOPDNetRebuild:**  
https://github.com/emrancub/AeroCOPDNetRebuild

All future updates and finalized materials will be released in the rebuilt repository upon request.

# Previous: 
# AeroCOPDNet (All updates and publications will be made as per request or after acceptance of the paper.)

A reproducible pipeline for **binary COPD detection** from chest auscultation audio using **log-mel spectrograms**.
Includes our compact CNN (**AeroCOPDNet**) and strong baselines (**Basic-CNN, CRNN, LSTM, GRU**) with cross-validation, robust class-imbalance handling, and spectrogram/audio augmentations.

> **Datasets (merged → binary):**
> • ICBHI 2017 Respiratory Sound Database — Kaggle mirror: [https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
> • Fraiwan et al. Lung Sound Dataset — Mendeley: [https://data.mendeley.com/datasets/jwyy9np4gv/3](https://data.mendeley.com/datasets/jwyy9np4gv/3)

---

## ✨ Highlights

* **Input features:** log-mel spectrograms
* **Imbalance:** class-weighted BCE + **SpecAugment** + **Mixup**
* **Splits:** stratified K-fold (no identity leakage)
* **Baselines:** Basic-CNN / CRNN / LSTM / GRU
* **Artifacts:** `artifacts/` (figures, plots, reports) will be uploaded after final journal decisions

---

## 📁 Repository Layout

```text
AeroCOPDNet/
├─ artifacts/                 # (empty for now; will host cv/, figures/, plots/, reports/, splits/)
├─ scripts/
│  ├─ ablation_run.py         # run augmentation/model ablations
│  ├─ augment_gallery.py      # visualize SpecAugment/Mixup
│  ├─ build_binary_labels.py  # make COPD vs non-COPD CSV from merged datasets
│  ├─ cv_train_with_test.py   # K-fold CV with rotated test folds
│  ├─ feature_gallery.py      # visualize log-mel features
│  ├─ infer.py                # batch/single-file inference
│  ├─ merge_to_pooled.py      # merge ICBHI + Fraiwan into one table
│  ├─ split_csv.py            # create patient-wise splits (group=patient_id)
│  └─ train.py                # standard training loop
├─ src/
│  └─ copd/
│     ├─ ast_models.py
│     ├─ augment.py
│     ├─ cv.py
│     ├─ data.py
│     ├─ features.py
│     ├─ metrics.py
│     ├─ models.py            # AeroCOPDNet + baselines (basic_cnn, crnn, lstm, gru)
│     ├─ trainloop.py
│     └─ utils.py
├─ README.md
├─ requirements.txt
```

---

## 🛠️ Installation

```bash
git clone https://github.com/emrancub/AeroCOPDNet.git
cd AeroCOPDNet

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 📦 Data Setup

**Labeling rule:** any recording from a **COPD-diagnosed** patient → **positive**; others → **negative**.
**Important:** all splits are **patient-wise** (enforced by the provided scripts).

---

## ⚡ Quickstart

> **Run the proposed model with 5-fold CV (your exact command):**

```bash
python -m scripts.cv_train_with_test --csv "artifacts\splits\pooled_icbhi_fraiwan_binary.csv" --folds 5 --epochs 100 --batch_size 32 --features mel --model aerocpdnet --use_specaug --mixup 0.2 --lr 3e-4 --weight_decay 1e-4 --dropout 0.2
```

> **Baselines (5-fold CV):**

```bash
# Basic CNN
python -m scripts.cv_train_with_test --csv "artifacts\splits\pooled_icbhi_fraiwan_binary.csv" --folds 5 --epochs 100 --batch_size 32 --features mel --model basiccnn --use_specaug --mixup 0.2

# CRNN
python -m scripts.cv_train_with_test --csv "artifacts\splits\pooled_icbhi_fraiwan_binary.csv" --folds 5 --epochs 100 --batch_size 32 --features mel --model crnn --use_specaug --mixup 0.2

# LSTM
python -m scripts.cv_train_with_test --csv "artifacts\splits\pooled_icbhi_fraiwan_binary.csv" --folds 5 --epochs 100 --batch_size 32 --features mel --model lstm --use_specaug --mixup 0.2

# GRU
python -m scripts.cv_train_with_test --csv "artifacts\splits\pooled_icbhi_fraiwan_binary.csv" --folds 5 --epochs 100 --batch_size 32 --features mel --model gru --use_specaug --mixup 0.2
```

> **Augmentation gallery (save examples):**

```bash
python -m scripts.augment_gallery --csv artifacts\splits\pooled_icbhi_fraiwan_binary.csv --outdir artifacts\figures\pooled_aug --sr 16000 --duration 4.0
```

> **Ablation study:**

```bash
python -m scripts.ablation_run --csv artifacts\splits\pooled_icbhi_fraiwan_binary.csv --folds 5 --epochs 100 --batch_size 32 --sr 16000 --duration 4.0 --model aerocopdnet --dropout 0.3 --lr 5e-4 --wd 1e-4 --outdir artifacts\ablation
```

---

## ⚙️ Common CLI Flags

| Area         | Flag(s) / Values                                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| Model        | `--model {aerocpdnet,basiccnn,crnn,lstm,gru}` *(names match the commands above)*                                     |
| Features     | `--features mel`                                                                                                     |
| Audio/Time   | `--sr 16000 --duration 4.0` *(used in augmentation/ablation utilities)*                                              |
| Augmentation | `--use_specaug --mixup 0.2`                                                                                          |
| Optimization | `--epochs 100 --lr 3e-4 --weight_decay 1e-4 --dropout 0.2` *(ablations may use `--lr 5e-4 --wd 1e-4 --dropout 0.3`)* |
| Batching     | `--batch_size 32`                                                                                                    |
| CV           | `--folds 5`                                                                                                          |
| I/O          | `--csv <path>`; `--outdir <dir>` / `--report_dir <dir>`                                                              |

---

## 🧠 Method (High-Level)

* **Inputs:** log-mel spectrograms (per-bin z-norm using train statistics)
* **Model:** **AeroCOPDNet** — CNN based deep learning algorithom 
* **Imbalance:** class-weighted BCE; **SpecAugment** + **Mixup**
* **Evaluation:** patient-wise K-fold; metrics: **Acc, Sens, Spec, F1, AUROC, AUPR, MCC**

---

## 📊 Baselines Included

* **Basic-CNN**
* **CRNN**
* **LSTM**
* **GRU**

Select via the `--model` flag as shown above.

---

## 🔬 Figures, Ablations, and Galleries (optional)

```bash
# Visualize log-mel features
python scripts/feature_gallery.py

# Visualize SpecAugment / Mixup effects
python -m scripts.augment_gallery --csv artifacts\splits\pooled_icbhi_fraiwan_binary.csv --outdir artifacts\figures\pooled_aug --sr 16000 --duration 4.0

# Ablation grid (models/augs/hparams)
python -m scripts.ablation_run --csv artifacts\splits\pooled_icbhi_fraiwan_binary.csv --folds 5 --epochs 100 --batch_size 32 --sr 16000 --duration 4.0 --model aerocopdnet --dropout 0.3 --lr 5e-4 --wd 1e-4 --outdir artifacts\ablation
```

---

## ✅ Reproducibility Tips

* Keep sampling rate and mel settings **identical** across datasets.
* Start with `--use_specaug --mixup 0.2`; heavy waveform noise is usually unnecessary.
* Tune the decision threshold to your deployment objective (screening vs. precision).
* Always group by **patient\_id** when splitting to avoid leakage.

---

## 📚 Citation

If you use this repository, please cite the datasets and our manuscript:

```bibtex
@article{hasan2026aerocopdnet,
  title={AeroCOPDNet: A deep learning framework for COPD detection from lung sounds},
  author={Hasan, Md Emran and Wu, Yue-Fang and Yu, Dong-Jun},
  journal={Biomedical Signal Processing and Control},
  volume={119},
  pages={109939},
  year={2026},
  publisher={Elsevier}
}
```

```bibtex
@article{Rocha2019ICBHI,
  title   = {An open access database for the evaluation of respiratory sound classification algorithms},
  author  = {Rocha, Bruno M. and Filos, Dorina and Mendes, L. and others},
  journal = {Physiological Measurement},
  year    = {2019},
  doi     = {10.1088/1361-6579/ab03ea}
}
@article{Fraiwan2021Lung,
  title   = {A dataset of lung sounds recorded from the chest wall using an electronic stethoscope},
  author  = {Fraiwan, Mohammad and Fraiwan, Lina and Khassawneh, Bilal and Ibnian, Ayman},
  journal = {Data in Brief},
  year    = {2021},
  doi     = {10.1016/j.dib.2021.106913}
}
```

---

## 📄 License

Add a license file (e.g., **MIT**) in `LICENSE`.
Respect the original dataset licenses and citation requirements.

---

## 📬 Contact

* For research questions, email the corresponding author listed in the paper.
* Or, Please contact **Md Emran Hasan** ([writetoemran@gmail.com](mailto:writetoemran@gmail.com) or [mdemranhasan@njust.edu.cn](mailto:mdemranhasan@njust.edu.cn)).
