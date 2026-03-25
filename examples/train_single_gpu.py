"""
Single-GPU training example with wsistream.

Usage:
    python examples/train_single_gpu.py --slides /data/tcga --steps 100
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wsistream.backends import OpenSlideBackend
from wsistream.filters import HSVPatchFilter
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.torch import MonitoredLoader, WsiStreamDataset
from wsistream.transforms import (
    ComposeTransforms,
    HEDColorAugmentation,
    RandomFlipRotate,
    ResizeTransform,
)

logger = logging.getLogger(__name__)


class DummyViT(nn.Module):
    """Placeholder model (replace with your actual model)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x).relu()
        x = self.pool(x).flatten(1)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", required=True, help="Directory or file path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--patches-per-slide", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WsiStreamDataset(
        slide_paths=args.slides,
        backend=OpenSlideBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=1000, target_mpp=0.5),
        patch_filter=HSVPatchFilter(),
        transforms=ComposeTransforms(
            transforms=[
                HEDColorAugmentation(sigma=0.05),
                RandomFlipRotate(),
                ResizeTransform(target_size=224),
            ]
        ),
        pool_size=args.pool_size,
        patches_per_slide=args.patches_per_slide,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",  # faster host-to-GPU transfer
    )

    mon = MonitoredLoader(loader, dataset=dataset, device=device, log_every=args.log_every)

    logger.info(
        "Config: slides=%s, batch_size=%d, num_workers=%d, pool_size=%d, "
        "patches_per_slide=%d, device=%s",
        args.slides,
        args.batch_size,
        args.num_workers,
        args.pool_size,
        args.patches_per_slide,
        device,
    )

    model = DummyViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    mon_iter = iter(mon)
    for step in range(args.steps):
        batch = next(mon_iter)
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        loss = logits.mean()  # placeholder — replace with your actual loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        payload = mon.mark_step(extra={"train/loss": float(loss.detach())})
        if payload is not None:  # returned every log_every steps
            logger.info(
                "Step %d/%d | %s",
                step + 1,
                args.steps,
                {k: f"{v:.1f}" if isinstance(v, float) else v for k, v in payload.items()},
            )

    logger.info("Training complete. %d steps.", args.steps)
    logger.info("Lifetime stats: %s", mon.lifetime_stats())


if __name__ == "__main__":
    main()
