# DG-POCFormer

PyTorch implementation of **DG-POCFormer**, a lightweight dual-granularity transformer for products of conception (POC) histopathology image classification.

The code follows the manuscript evaluation protocol:

- use the official `Train` and `Test` directories of the HistoPoC dataset;
- keep the official `Test` directory fixed and isolated;
- create five class-stratified internal train/validation splits only from the official `Train` directory;
- select the best checkpoint in each run using its validation subset;
- evaluate every fold-derived checkpoint on the same fixed independent test set;
- report mean and population standard deviation across the five test-set evaluations.

## Repository layout

```text
DG-POCFormer-GitHub/
├── configs/default.yaml
├── dgpocformer/
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── train_utils.py
├── scripts/
│   ├── make_splits.py
│   ├── train_official_split.py
│   ├── evaluate.py
│   ├── infer_image.py
│   └── profile_model.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset format

Download HistoPoC and arrange it as follows:

```text
/path/to/HistoPoC/
├── Train/
│   ├── Chorionic_villi/
│   ├── Decidual_tissue/
│   ├── Hemorrhage/
│   └── Trophoblastic_tissue/
└── Test/
    ├── Chorionic_villi/
    ├── Decidual_tissue/
    ├── Hemorrhage/
    └── Trophoblastic_tissue/
```

## Installation

```bash
git clone <your-repository-url>
cd DG-POCFormer-GitHub
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training and testing

Edit `configs/default.yaml` and set `data_dir` to the dataset root. Then run:

```bash
python scripts/train_official_split.py --config configs/default.yaml
```

Outputs are written to `output_dir`, including fold checkpoints, training histories, test predictions, pooled ROC/confusion matrix files, and JSON summaries.

## Create split files only

```bash
python scripts/make_splits.py --config configs/default.yaml
```

## Evaluate saved checkpoints

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/dg_pocformer/best_fold1.pth
```

Multiple checkpoints can be evaluated together:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/dg_pocformer/best_fold1.pth outputs/dg_pocformer/best_fold2.pth
```

## Single-image inference

```bash
python scripts/infer_image.py \
  --config configs/default.yaml \
  --checkpoint outputs/dg_pocformer/best_fold1.pth \
  --image /path/to/image.png
```

## Model profiling

```bash
python scripts/profile_model.py --config configs/default.yaml
```

For a quick CPU check without timing loops:

```bash
python scripts/profile_model.py --config configs/default.yaml --device cpu --skip-speed
```

## Notes

This repository contains code only. The dataset and trained weights are not included.
