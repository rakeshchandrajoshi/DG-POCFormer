from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

try:
    import torchstain
    TORCHSTAIN_AVAILABLE = True
except Exception:
    torchstain = None
    TORCHSTAIN_AVAILABLE = False


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class HEStainJitter:
    """Simple optical-density jitter used only for training augmentation."""
    def __init__(self, a: float = 0.15, b: float = 0.08, p: float = 0.5):
        self.a = a
        self.b = b
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        x = np.clip(np.array(img).astype(np.float32) / 255.0, 1e-6, 1.0)
        od = -np.log(x)
        perturbed = (
            np.exp(
                -(
                    od * np.random.uniform(1 - self.a, 1 + self.a, (1, 1, 3))
                    + np.random.uniform(-self.b, self.b, (1, 1, 3))
                )
            ).clip(0, 1) * 255
        ).astype(np.uint8)
        return Image.fromarray(perturbed)


class HistoPoCDatasetFromDF(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, stain_normalizer=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.stain_normalizer = stain_normalizer
        self.paths = self.df["path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self._tt = transforms.ToTensor()
        self._tp = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.stain_normalizer is not None:
            try:
                t = self._tt(img) * 255.0
                t_n, _, _ = self.stain_normalizer.normalize(t, stains=False)
                img = self._tp(t_n.clamp(0, 255) / 255.0)
            except Exception:
                pass
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transforms(img_size: int = 256):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.70, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        HEStainJitter(0.15, 0.08, 0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def scan_histopoc(data_dir: str | Path, classes: Sequence[str]) -> pd.DataFrame:
    data_dir = Path(data_dir).expanduser().resolve()
    rows = []
    for split in ["Train", "Test"]:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for label, cls in enumerate(classes):
            class_dir = split_dir / cls
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                    rows.append({"path": str(path), "label": label, "class": cls, "split": split})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {data_dir}")
    return df


def official_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df["split"] == "Train"].reset_index(drop=True)
    df_test = df[df["split"] == "Test"].reset_index(drop=True)
    if len(df_train) == 0 or len(df_test) == 0:
        raise RuntimeError("Both Train and Test splits must contain images.")
    return df_train, df_test


def make_internal_folds(
    df_train: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = df_train["label"].values
    return list(splitter.split(np.arange(len(df_train)), labels))


def make_weighted_loader(
    dataset: HistoPoCDatasetFromDF,
    batch_size: int,
    classes: Sequence[str],
    decidual_weight_boost: float = 3.0,
    num_workers: int = 0,
) -> DataLoader:
    counts = Counter(dataset.labels)
    n = len(dataset)
    class_weights = []
    for idx, cls in enumerate(classes):
        if counts[idx] == 0:
            raise ValueError(f"Class {cls} has zero samples in the training subset.")
        weight = n / (len(classes) * counts[idx])
        if cls == "Decidual_tissue":
            weight *= decidual_weight_boost
        class_weights.append(weight)
    sample_weights = [class_weights[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


def compute_class_weights(
    labels: Sequence[int],
    classes: Sequence[str],
    decidual_weight_boost: float = 3.0,
    device: Optional[torch.device] = None,
):
    counts = Counter(labels)
    n = len(labels)
    weights = []
    for idx, cls in enumerate(classes):
        if counts[idx] == 0:
            raise ValueError(f"Class {cls} has zero samples in the training subset.")
        weight = n / (len(classes) * counts[idx])
        if cls == "Decidual_tissue":
            weight *= decidual_weight_boost
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit_macenko_normalizer(df_train: pd.DataFrame, stain_ref_img: str = "", use_macenko: bool = True):
    if not use_macenko:
        return None
    if not TORCHSTAIN_AVAILABLE:
        print("torchstain is not installed. Macenko normalization is disabled.")
        return None
    ref_path = stain_ref_img or df_train.iloc[0]["path"]
    ref_path = str(Path(ref_path).expanduser())
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"Stain reference image not found: {ref_path}")
    tensor = transforms.ToTensor()(Image.open(ref_path).convert("RGB")) * 255.0
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(tensor)
    print(f"Macenko normalizer fitted with reference image: {ref_path}")
    return normalizer


def build_datasets_for_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    img_size: int,
    stain_normalizer=None,
):
    train_tf, eval_tf = build_transforms(img_size)
    train_ds = HistoPoCDatasetFromDF(df_train.iloc[train_idx], train_tf, stain_normalizer)
    val_ds = HistoPoCDatasetFromDF(df_train.iloc[val_idx], eval_tf, stain_normalizer)
    test_ds = HistoPoCDatasetFromDF(df_test, eval_tf, stain_normalizer)
    return train_ds, val_ds, test_ds
