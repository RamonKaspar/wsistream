"""
Multi-GPU DDP training example with wsistream.

Usage:
    torchrun --nproc_per_node=4 examples/train_ddp.py --slides /data/tcga --steps 100
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from wsistream.backends import OpenSlideBackend
from wsistream.filters import HSVPatchFilter
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.torch import MonitoredLoader, WsiStreamDataset, partition_slides_by_rank
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from wsistream.types import resolve_slide_paths

    all_slides = resolve_slide_paths(args.slides)
    my_slides = partition_slides_by_rank(all_slides, rank=rank, world_size=world_size)

    logger.info(
        "Rank %d/%d | slides=%d/%d, batch_size=%d, num_workers=%d, "
        "pool_size=%d, patches_per_slide=%d, seed=%d, device=%s",
        rank,
        world_size,
        len(my_slides),
        len(all_slides),
        args.batch_size,
        args.num_workers,
        args.pool_size,
        args.patches_per_slide,
        args.seed + rank,
        device,
    )

    dataset = WsiStreamDataset(
        slide_paths=my_slides,
        backend=OpenSlideBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, target_mpp=0.5),
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
        seed=args.seed + rank,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,  # avoid re-spawning workers each iteration
    )

    mon = MonitoredLoader(loader, dataset=dataset, device=device, log_every=args.log_every)

    model = DummyViT().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
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
        if payload is not None and rank == 0:  # all ranks call mark_step, only rank 0 logs
            logger.info(
                "Step %d/%d | %s",
                step + 1,
                args.steps,
                {k: f"{v:.1f}" if isinstance(v, float) else v for k, v in payload.items()},
            )

    if rank == 0:
        logger.info("Training complete. %d steps.", args.steps)
        logger.info("Lifetime stats: %s", mon.lifetime_stats())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
