"""Training loop monitor for WsiStreamDataset + DataLoader."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class _Accumulator:
    """Running totals for timing metrics. Uses int nanoseconds to avoid float drift."""

    data_wait_ns: int = 0
    compute_ns: int = 0
    step_count: int = 0
    patch_count: int = 0

    def reset(self) -> None:
        self.data_wait_ns = 0
        self.compute_ns = 0
        self.step_count = 0
        self.patch_count = 0

    def to_dict(self) -> dict[str, float]:
        if self.step_count == 0:
            return {}
        total_ns = self.data_wait_ns + self.compute_ns
        total_sec = total_ns / 1e9
        return {
            "loader/data_wait_ms": self.data_wait_ns / 1e6 / self.step_count,
            "loader/compute_ms": self.compute_ns / 1e6 / self.step_count,
            "loader/step_ms": total_ns / 1e6 / self.step_count,
            "loader/data_fraction": self.data_wait_ns / max(total_ns, 1),
            "loader/batches_per_sec": self.step_count / max(total_sec, 1e-9),
            "loader/patches_per_sec": self.patch_count / max(total_sec, 1e-9),
        }


class MonitoredLoader:
    """Wraps a ``DataLoader`` to measure data wait time, compute time, and throughput.

    Parameters
    ----------
    loader : DataLoader
        The PyTorch DataLoader to wrap.
    dataset : object or None
        If provided, must have a ``stats_dict()`` method (e.g., ``WsiStreamDataset``
        or ``PatchPipeline``). Its stats are included in every payload.
    device : torch.device, str, or None
        When set to a CUDA device, ``mark_step()`` calls
        ``torch.cuda.synchronize()`` before reading the clock so that
        GPU compute time is measured accurately.
    log_every : int
        ``mark_step()`` returns a payload dict every ``log_every`` steps
        and ``None`` otherwise.
    """

    def __init__(
        self,
        loader,
        dataset=None,
        device: torch.device | str | None = None,
        log_every: int = 100,
    ) -> None:
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        self._loader = loader
        self._dataset = dataset
        self._device = torch.device(device) if isinstance(device, str) else device
        self._use_cuda_sync = self._device is not None and self._device.type == "cuda"
        self._log_every = log_every

        self._loader_iter = None
        self._batch_yielded_at: int | None = None  # perf_counter_ns timestamp

        self._window = _Accumulator()
        self._lifetime = _Accumulator()
        self._step_count = 0

    def __iter__(self):
        self._loader_iter = iter(self._loader)
        self._batch_yielded_at = None
        return self

    def __next__(self):
        # If mark_step() was not called after the previous batch,
        # count the elapsed time as compute (best effort)
        if self._batch_yielded_at is not None:
            unmeasured = time.perf_counter_ns() - self._batch_yielded_at
            self._window.compute_ns += unmeasured
            self._lifetime.compute_ns += unmeasured
            self._batch_yielded_at = None

        t0 = time.perf_counter_ns()
        batch = next(self._loader_iter)  # propagates StopIteration
        t1 = time.perf_counter_ns()

        wait_ns = t1 - t0
        self._window.data_wait_ns += wait_ns
        self._lifetime.data_wait_ns += wait_ns

        # Infer batch size from the image tensor
        try:
            n = batch["image"].shape[0]
        except (KeyError, AttributeError, IndexError):
            n = 1
        self._window.patch_count += n
        self._lifetime.patch_count += n

        self._batch_yielded_at = t1
        return batch

    def mark_step(self, extra: dict | None = None) -> dict | None:
        """Record the end of a training step.

        If ``device`` is a CUDA device, synchronizes the GPU before
        reading the clock so that compute time includes actual kernel
        execution, not just launch overhead.

        Returns a metrics dict every ``log_every`` steps, ``None``
        otherwise.  The dict includes loader timing metrics,
        ``dataset.stats_dict()`` if a dataset was provided, and
        any ``extra`` entries.
        """
        if self._batch_yielded_at is None:
            raise RuntimeError(
                "mark_step() called before a batch was yielded. "
                "Call next() or iterate the MonitoredLoader first."
            )

        if self._use_cuda_sync:
            torch.cuda.synchronize(self._device)

        t_now = time.perf_counter_ns()
        compute_ns = t_now - self._batch_yielded_at
        self._window.compute_ns += compute_ns
        self._lifetime.compute_ns += compute_ns
        self._batch_yielded_at = None

        self._window.step_count += 1
        self._lifetime.step_count += 1
        self._step_count += 1

        if self._step_count % self._log_every != 0:
            return None

        payload = self._window.to_dict()

        if self._dataset is not None and hasattr(self._dataset, "stats_dict"):
            payload.update(self._dataset.stats_dict())

        if extra:
            payload.update(extra)

        self._window.reset()
        return payload

    def lifetime_stats(self) -> dict[str, float]:
        """Return timing metrics accumulated over the entire training run."""
        return self._lifetime.to_dict()

    def reset(self) -> None:
        """Reset all accumulators and step count."""
        self._window.reset()
        self._lifetime.reset()
        self._step_count = 0
        self._batch_yielded_at = None
