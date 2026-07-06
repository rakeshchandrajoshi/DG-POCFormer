from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .losses import mixup_data


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    warmup_epochs = max(0, int(warmup_epochs))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    train: bool = True,
    epoch: int = 0,
    total_epochs: int = 0,
    mixup_alpha: float = 0.3,
    cgcr_lambda: float = 0.05,
):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0.0
    n = 0
    context = torch.enable_grad() if train else torch.no_grad()
    mode = "Train" if train else "Val"
    bar = tqdm(loader, desc=f"Ep {epoch:03d}/{total_epochs} [{mode}]", leave=False, dynamic_ncols=True)

    with context:
        for imgs, labels in bar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if mixup_alpha > 0 and random.random() > 0.5:
                    imgs, ya, yb, lam = mixup_data(imgs, labels, mixup_alpha, device)
                    logits, aux_logits = model(imgs, return_aux=True)
                    main_loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                    preds = logits.argmax(1)
                    correct += (
                        lam * (preds == ya).float() + (1 - lam) * (preds == yb).float()
                    ).sum().item()
                else:
                    logits, aux_logits = model(imgs, return_aux=True)
                    main_loss = criterion(logits, labels)
                    correct += (logits.argmax(1) == labels).sum().item()

                cgcr = F.kl_div(
                    F.log_softmax(aux_logits, -1),
                    F.softmax(logits.detach(), -1),
                    reduction="batchmean",
                )
                loss = main_loss + cgcr_lambda * cgcr
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                logits = model(imgs, return_aux=False)
                loss = criterion(logits, labels)
                correct += (logits.argmax(1) == labels).sum().item()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            bar.set_postfix(loss=f"{total_loss / n:.4f}", acc=f"{correct / n:.4f}")

    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_list, preds_list, labels_list = [], [], []
    for imgs, labels in tqdm(loader, desc="Predict", leave=False, dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        probs = F.softmax(model(imgs, return_aux=False), dim=1)
        probs_list.append(probs.cpu().numpy())
        preds_list.append(probs.argmax(1).cpu().numpy())
        labels_list.append(labels.numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, preds, labels


def save_json(path: str | Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


def load_state_dict_flexible(model, checkpoint_path: str | Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    return model


def time_seconds():
    return time.perf_counter()
