"""Training loop monitor for WsiStreamDataset + DataLoader."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from wsistream._utils import infer_batch_size

logger = logging.getLogger(__name__)


@dataclass
class _Accumulator:
    """Running totals for timing metrics. Uses int nanoseconds to avoid float drift."""

    data_wait_ns: int = 0
    compute_ns: int = 0
    wall_ns: int = 0
    step_count: int = 0
    patch_count: int = 0

    def reset(self) -> None:
        self.data_wait_ns = 0
        self.compute_ns = 0
        self.wall_ns = 0
        self.step_count = 0
        self.patch_count = 0

    def to_dict(self) -> dict[str, float]:
        if self.step_count == 0:
            return {}
        wall_sec = self.wall_ns / 1e9
        return {
            "loader/data_wait_ms": self.data_wait_ns / 1e6 / self.step_count,
            "loader/compute_ms": self.compute_ns / 1e6 / self.step_count,
            "loader/step_ms": self.wall_ns / 1e6 / self.step_count,
            "loader/data_fraction": self.data_wait_ns / max(self.wall_ns, 1),
            "loader/batches_per_sec": self.step_count / max(wall_sec, 1e-9),
            "loader/patches_per_sec": self.patch_count / max(wall_sec, 1e-9),
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
        When set to a CUDA device, compute time is measured with CUDA
        events. Pending timings are resolved after synchronizing the device
        when a metrics payload is produced or :meth:`lifetime_stats` is called.
    log_every : int
        ``mark_step()`` returns a payload dict every ``log_every`` steps
        and ``None`` otherwise.

    Notes
    -----
    ``loader/step_ms`` and the throughput metrics use wall-clock time. On
    CUDA, ``loader/compute_ms`` uses events on the stream that is current
    when the batch is yielded. CUDA work and data loading may overlap, so
    component timings are not expected to add up to ``loader/step_ms``.
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
        self._use_cuda_events = self._device is not None and self._device.type == "cuda"
        self._log_every = log_every

        self._loader_iter = None
        self._step_started_at: int | None = None  # perf_counter_ns timestamp
        self._batch_yielded_at: int | None = None  # perf_counter_ns timestamp
        self._cuda_step: tuple[torch.cuda.Event, torch.cuda.Stream] | None = None
        self._pending_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        self._window = _Accumulator()
        self._lifetime = _Accumulator()
        self._step_count = 0

    def __iter__(self):
        self._loader_iter = iter(self._loader)
        self._step_started_at = None
        self._batch_yielded_at = None
        self._cuda_step = None
        return self

    def __next__(self):
        # If mark_step() was not called after the previous batch,
        # count the elapsed time as compute (best effort)
        if self._batch_yielded_at is not None:
            self._finish_step()

        t0 = time.perf_counter_ns()
        batch = next(self._loader_iter)  # propagates StopIteration
        t1 = time.perf_counter_ns()

        wait_ns = t1 - t0
        self._window.data_wait_ns += wait_ns
        self._lifetime.data_wait_ns += wait_ns

        n = infer_batch_size(batch)
        self._window.patch_count += n
        self._lifetime.patch_count += n

        self._step_started_at = t0
        self._batch_yielded_at = t1
        if self._use_cuda_events:
            stream = torch.cuda.current_stream(self._device)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            self._cuda_step = (start_event, stream)
        return batch

    def mark_step(self, extra: dict | None = None) -> dict | None:
        """Record the end of a training step.

        For CUDA devices, calls that do not produce a payload record an end
        event and return without synchronizing. Pending events are resolved at
        the logging boundary, allowing data loading and GPU work to overlap
        between payloads.

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

        self._finish_step()
        self._step_count += 1

        if self._step_count % self._log_every != 0:
            return None

        self._flush_cuda_events()
        payload = self._window.to_dict()

        if self._dataset is not None and hasattr(self._dataset, "stats_dict"):
            payload.update(self._dataset.stats_dict())

        if extra:
            payload.update(extra)

        self._window.reset()
        return payload

    def lifetime_stats(self) -> dict[str, float]:
        """Return timing metrics accumulated over the entire training run.

        Synchronizes pending CUDA events, if any, before building the result.
        """
        self._flush_cuda_events()
        return self._lifetime.to_dict()

    def reset(self) -> None:
        """Reset all accumulators and step count."""
        self._window.reset()
        self._lifetime.reset()
        self._step_count = 0
        self._step_started_at = None
        self._batch_yielded_at = None
        self._cuda_step = None
        self._pending_cuda_events.clear()

    def _finish_step(self) -> None:
        """Record one yielded batch as a completed training step."""
        if self._batch_yielded_at is None or self._step_started_at is None:
            raise RuntimeError("Cannot finish a step before a batch was yielded")

        if self._use_cuda_events:
            if self._cuda_step is None:
                raise RuntimeError("CUDA timing state is missing for the current step")
            start_event, stream = self._cuda_step
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(stream)
            self._pending_cuda_events.append((start_event, end_event))
        else:
            compute_ns = time.perf_counter_ns() - self._batch_yielded_at
            self._window.compute_ns += compute_ns
            self._lifetime.compute_ns += compute_ns

        step_ended_at = time.perf_counter_ns()
        wall_ns = step_ended_at - self._step_started_at
        self._window.wall_ns += wall_ns
        self._lifetime.wall_ns += wall_ns
        self._window.step_count += 1
        self._lifetime.step_count += 1

        self._step_started_at = None
        self._batch_yielded_at = None
        self._cuda_step = None

    def _flush_cuda_events(self) -> None:
        """Resolve pending CUDA timings and account for the boundary wait."""
        if not self._pending_cuda_events:
            return

        sync_started_at = time.perf_counter_ns()
        torch.cuda.synchronize(self._device)
        sync_ns = time.perf_counter_ns() - sync_started_at

        compute_ns = round(
            sum(start.elapsed_time(end) for start, end in self._pending_cuda_events) * 1e6
        )
        # The reporting wait affects observed wall-clock throughput even though
        # it is not part of any individual CUDA event pair.
        self._window.compute_ns += compute_ns
        self._lifetime.compute_ns += compute_ns
        self._window.wall_ns += sync_ns
        self._lifetime.wall_ns += sync_ns
        self._pending_cuda_events.clear()
