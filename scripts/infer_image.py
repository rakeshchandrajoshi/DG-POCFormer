#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgpocformer.config import load_config
from dgpocformer.data import build_transforms, fit_macenko_normalizer, official_train_test_split, scan_histopoc
from dgpocformer.model import build_model_from_config
from dgpocformer.train_utils import get_device, load_state_dict_flexible, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run DG-POCFormer inference on one image.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable-macenko", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    classes = cfg["classes"]

    stain_normalizer = None
    if not args.disable_macenko and cfg.get("preprocessing", {}).get("use_macenko", True):
        df = scan_histopoc(cfg["data_dir"], classes)
        df_train, _ = official_train_test_split(df)
        stain_normalizer = fit_macenko_normalizer(
            df_train,
            stain_ref_img=cfg.get("preprocessing", {}).get("stain_ref_img", ""),
            use_macenko=True,
        )

    _, eval_tf = build_transforms(int(cfg.get("img_size", 256)))
    img = Image.open(args.image).convert("RGB")
    if stain_normalizer is not None:
        try:
            tt = transforms.ToTensor()
            tp = transforms.ToPILImage()
            t = tt(img) * 255.0
            t_n, _, _ = stain_normalizer.normalize(t, stains=False)
            img = tp(t_n.clamp(0, 255) / 255.0)
        except Exception as exc:
            print(f"Warning: Macenko normalization failed for this image: {exc}")

    x = eval_tf(img).unsqueeze(0).to(device)
    model = build_model_from_config(cfg, num_classes=len(classes)).to(device)
    model = load_state_dict_flexible(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()
    pred_idx = int(torch.argmax(probs).item())

    print(f"Predicted class: {classes[pred_idx]}")
    print("Class probabilities:")
    for cls, prob in zip(classes, probs.tolist()):
        print(f"  {cls:<24} {prob:.6f}")


if __name__ == "__main__":
    main()
