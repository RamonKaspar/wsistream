"""Internal helpers shared across wsistream modules."""

from __future__ import annotations

from typing import Any


def infer_batch_size(batch: Any) -> int:
    """Infer batch size from the first tensor-like field."""
    if isinstance(batch, dict):
        if "image" in batch:
            try:
                return int(batch["image"].shape[0])
            except (AttributeError, IndexError):
                return 1
        for value in batch.values():
            try:
                return int(value.shape[0])
            except (AttributeError, IndexError):
                continue
        for value in batch.values():
            if isinstance(value, (list, tuple)):
                return len(value)
    try:
        return int(batch.shape[0])
    except (AttributeError, IndexError):
        return 1
