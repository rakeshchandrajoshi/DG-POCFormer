#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgpocformer.config import load_config
from dgpocformer.model import build_model_from_config, count_parameters
from dgpocformer.train_utils import get_device, set_seed


def analytical_macs(cfg, num_classes: int):
    mcfg = cfg.get("model", {})
    S = int(cfg.get("img_size", 256))
    d_f = int(mcfg.get("d_fine", 48))
    d_c = int(mcfg.get("d_coarse", 64))
    depth = int(mcfg.get("depth", 3))
    nh_f = int(mcfg.get("n_heads_fine", 4))
    nh_c = int(mcfg.get("n_heads_coarse", 4))
    nh_x = int(mcfg.get("n_heads_cross", 4))
    ws = int(mcfg.get("window_size", 4))
    head_hidden = int(mcfg.get("head_hidden", 256))

    H1 = S // 2
    H2 = S // 4
    mac_conv1 = 3 * 16 * 3 * 3 * H1 * H1
    mac_conv2 = 16 * 32 * 3 * 3 * H2 * H2

    G = S // 16
    mac_patch = 32 * d_f * 4 * 4 * G * G
    N_f = G * G
    N_c = (G // 4) ** 2 + 1

    mac_fine_qkv = N_f * d_f * 3 * d_f
    ws2 = ws * ws
    n_win = N_f // ws2
    mac_fine_attn = n_win * nh_f * ws2 * ws2 * (d_f // nh_f)
    mac_fine_ctx = n_win * nh_f * ws2 * ws2 * (d_f // nh_f)
    mac_fine_proj = N_f * d_f * d_f

    mac_coarse_qkv = N_c * d_c * 3 * d_c
    mac_coarse_attn = nh_c * N_c * N_c * (d_c // nh_c)
    mac_coarse_ctx = nh_c * N_c * N_c * (d_c // nh_c)
    mac_coarse_proj = N_c * d_c * d_c

    mac_c2f = (
        N_f * d_f * d_c
        + N_c * d_c * d_c * 2
        + nh_x * N_f * N_c * (d_f // nh_x) * 2
        + N_f * d_f * d_f
    )
    mac_f2c = (
        N_c * d_c * d_f
        + N_f * d_f * d_f * 2
        + nh_x * N_c * N_f * (d_c // nh_x) * 2
        + N_c * d_c * d_c
    )

    mac_ffn_fine = N_f * d_f * (d_f * 4) * 3
    mac_ffn_coarse = N_c * d_c * (d_c * 4) * 3

    mac_per_block = (
        mac_fine_qkv + mac_fine_attn + mac_fine_ctx + mac_fine_proj
        + mac_coarse_qkv + mac_coarse_attn + mac_coarse_ctx + mac_coarse_proj
        + mac_c2f + mac_f2c + mac_ffn_fine + mac_ffn_coarse
    )

    mac_coarse_init = N_f * d_f * d_c
    mac_head = (d_f + d_c) * head_hidden + head_hidden * num_classes
    total_macs = mac_conv1 + mac_conv2 + mac_patch + mac_coarse_init + depth * mac_per_block + mac_head
    return total_macs


@torch.no_grad()
def measure_fps(model, device, img_size: int, batch_size: int, n_warmup: int = 30, n_iter: int = 200):
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(n_warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (batch_size * n_iter) / elapsed, (elapsed / (batch_size * n_iter)) * 1e3


def parse_args():
    parser = argparse.ArgumentParser(description="Profile DG-POCFormer parameter count, analytical MACs/FLOPs, and synthetic inference speed.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for synthetic throughput measurement. Default: config batch_size.")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warm-up iterations. Default: 30 on CUDA, 3 on CPU.")
    parser.add_argument("--iters", type=int, default=None, help="Number of timed iterations. Default: 200 on CUDA, 5 on CPU.")
    parser.add_argument("--skip-speed", action="store_true", help="Only report parameters and analytical complexity.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)
    classes = cfg["classes"]
    img_size = int(cfg.get("img_size", 256))
    batch_size = int(args.batch_size or cfg.get("batch_size", 8))

    model = build_model_from_config(cfg, num_classes=len(classes)).to(device).eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, img_size, img_size, device=device))
    if tuple(y.shape) != (1, len(classes)):
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")

    params = count_parameters(model)
    buf = io.BytesIO()
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("aux_head")}, buf)
    model_size_mb = buf.tell() / (1024 ** 2)
    total_macs = analytical_macs(cfg, len(classes))
    gmacs = total_macs / 1e9
    gflops = 2 * gmacs

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        default_warmup, default_iters = 30, 200
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(cuda_index)
    else:
        default_warmup, default_iters = 3, 5
        device_name = "CPU"
    n_warmup = int(args.warmup if args.warmup is not None else default_warmup)
    n_iter = int(args.iters if args.iters is not None else default_iters)

    speed_results = None
    if not args.skip_speed:
        fps1, lat1 = measure_fps(model, device, img_size, batch_size=1, n_warmup=n_warmup, n_iter=n_iter)
        fpsb, latb = measure_fps(model, device, img_size, batch_size=batch_size, n_warmup=n_warmup, n_iter=n_iter)
        speed_results = (fps1, lat1, fpsb, latb)

    print("=" * 72)
    print("DG-POCFormer model profile")
    print(f"Device                  : {device} ({device_name})")
    print(f"Input shape             : (1, 3, {img_size}, {img_size})")
    print(f"Total parameters        : {params['total']:,} ({params['total']/1e6:.3f} M)")
    print(f"Inference parameters    : {params['inference']:,} ({params['inference']/1e6:.3f} M)")
    print(f"Auxiliary parameters    : {params['auxiliary']:,}")
    print(f"Model size              : {model_size_mb:.2f} MB")
    print(f"Analytical cost         : {gmacs:.4f} GMACs / {gflops:.4f} GFLOPs")
    if speed_results is not None:
        fps1, lat1, fpsb, latb = speed_results
        print(f"Speed warmup/iters      : {n_warmup}/{n_iter}")
        print(f"Throughput batch=1      : {fps1:.1f} img/s | {lat1:.2f} ms/img")
        print(f"Throughput batch={batch_size:<6}: {fpsb:.1f} img/s | {latb:.2f} ms/img")
    else:
        print("Synthetic speed test    : skipped")
    print("=" * 72)


if __name__ == "__main__":
    main()
