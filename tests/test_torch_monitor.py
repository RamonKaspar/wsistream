"""Tests for MonitoredLoader."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import FakeBackend
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.torch import MonitoredLoader, WsiStreamDataset
from wsistream.torch_monitor import _Accumulator
from wsistream.transforms import ResizeTransform
from wsistream.views import ViewConfig


def _make_dataset(n_slides=4, patches_per_slide=10, cycle=True):
    from tests.conftest import fake_slide_paths

    return WsiStreamDataset(
        slide_paths=fake_slide_paths(n_slides),
        backend=FakeBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
        pool_size=2,
        patches_per_slide=patches_per_slide,
        cycle=cycle,
        seed=123,
    )


LOADER_KEYS = {
    "loader/data_wait_ms",
    "loader/compute_ms",
    "loader/step_ms",
    "loader/data_fraction",
    "loader/batches_per_sec",
    "loader/patches_per_sec",
}


class _FakeCudaEvent:
    def __init__(self, elapsed_ms=2.0):
        self.elapsed_ms = elapsed_ms
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream

    def elapsed_time(self, _end_event):
        return self.elapsed_ms


class TestBasicIteration:
    def test_returns_payload_at_log_interval(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=3)

        results = []
        for step, batch in enumerate(mon):
            result = mon.mark_step()
            results.append(result)
            if step >= 5:
                break

        # Steps 0,1: None. Step 2 (3rd step): payload. Steps 3,4: None. Step 5: payload.
        assert results[0] is None
        assert results[1] is None
        assert results[2] is not None
        assert results[3] is None
        assert results[4] is None
        assert results[5] is not None

    def test_metric_keys_present(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()

        assert payload is not None
        assert LOADER_KEYS.issubset(payload.keys())

    def test_stopiteration_propagates(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=2, cycle=False)
        loader = DataLoader(dataset, batch_size=1, num_workers=0)
        mon = MonitoredLoader(loader, log_every=100)

        count = 0
        for batch in mon:
            mon.mark_step()
            count += 1
        assert count == 2


class TestMetricValues:
    def test_overlapping_times_are_not_added_for_throughput(self):
        accumulator = _Accumulator(
            data_wait_ns=6_000_000,
            compute_ns=8_000_000,
            wall_ns=10_000_000,
            step_count=1,
            patch_count=4,
        )

        metrics = accumulator.to_dict()

        assert metrics["loader/data_wait_ms"] == pytest.approx(6.0)
        assert metrics["loader/compute_ms"] == pytest.approx(8.0)
        assert metrics["loader/step_ms"] == pytest.approx(10.0)
        assert metrics["loader/data_fraction"] == pytest.approx(0.6)
        assert metrics["loader/batches_per_sec"] == pytest.approx(100.0)
        assert metrics["loader/patches_per_sec"] == pytest.approx(400.0)

    def test_data_wait_positive(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()
        assert payload["loader/data_wait_ms"] > 0

    def test_compute_ms_positive(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        next(iter(mon))
        # Simulate compute work
        time.sleep(0.005)
        payload = mon.mark_step()
        assert payload["loader/compute_ms"] > 1  # at least 1ms from the sleep

    def test_patches_per_sec_positive(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()
        assert payload["loader/patches_per_sec"] > 0

    def test_multi_view_batch_size_inferred_from_view_tensor(self):
        from tests.conftest import fake_slide_paths

        dataset = WsiStreamDataset(
            slide_paths=fake_slide_paths(2),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
            views=[ViewConfig(name="view", transforms=ResizeTransform(32))],
            pool_size=1,
            patches_per_slide=4,
            cycle=False,
            seed=123,
        )
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()

        assert payload["loader/patches_per_sec"] > payload["loader/batches_per_sec"]

    def test_data_fraction_between_zero_and_one(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=3)

        for step, batch in enumerate(mon):
            time.sleep(0.001)
            payload = mon.mark_step()
            if payload is not None:
                assert 0 <= payload["loader/data_fraction"] <= 1
                break


class TestDatasetIntegration:
    def test_dataset_stats_merged(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, dataset=dataset, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()

        assert "pipeline/patches_extracted" in payload
        assert "pipeline/slides_processed" in payload

    def test_without_dataset(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, dataset=None, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()

        assert LOADER_KEYS.issubset(payload.keys())
        assert "pipeline/patches_extracted" not in payload

    def test_extra_merged(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        next(iter(mon))
        payload = mon.mark_step(extra={"train/loss": 1.5})

        assert payload["train/loss"] == 1.5
        assert LOADER_KEYS.issubset(payload.keys())


class TestWindowAndLifetime:
    def test_window_resets_between_payloads(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=2)

        payloads = []
        for step, batch in enumerate(mon):
            time.sleep(0.001)
            payload = mon.mark_step()
            if payload is not None:
                payloads.append(payload)
            if len(payloads) >= 2:
                break

        # Each window covers only its own 2 steps, not cumulative
        assert payloads[0]["loader/patches_per_sec"] > 0
        assert payloads[1]["loader/patches_per_sec"] > 0

    def test_lifetime_accumulates(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=3)

        for step, batch in enumerate(mon):
            mon.mark_step()
            if step >= 5:
                break

        lifetime = mon.lifetime_stats()
        assert lifetime["loader/patches_per_sec"] > 0


class TestNoDevice:
    def test_no_cuda_sync_without_device(self, monkeypatch):
        synchronize = Mock()
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, device=None, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()
        assert payload is not None
        synchronize.assert_not_called()


class TestCudaDevice:
    def test_synchronizes_only_at_log_boundary_on_selected_device(self, monkeypatch):
        stream = object()
        events = []

        def make_event(*, enable_timing):
            assert enable_timing is True
            event = _FakeCudaEvent()
            events.append(event)
            return event

        current_stream = Mock(return_value=stream)
        synchronize = Mock()
        monkeypatch.setattr(torch.cuda, "Event", make_event)
        monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

        loader = [{"image": torch.zeros(2, 3, 4, 4)} for _ in range(3)]
        mon = MonitoredLoader(loader, device="cuda:1", log_every=3)
        iterator = iter(mon)

        for _ in range(2):
            next(iterator)
            assert mon.mark_step() is None
            synchronize.assert_not_called()

        next(iterator)
        payload = mon.mark_step()

        assert payload is not None
        synchronize.assert_called_once_with(torch.device("cuda:1"))
        assert [call.args for call in current_stream.call_args_list] == [
            (torch.device("cuda:1"),)
        ] * 3
        assert len(events) == 6
        assert all(event.recorded_stream is stream for event in events)
        assert payload["loader/compute_ms"] == pytest.approx(2.0)

    def test_lifetime_stats_flush_pending_cuda_events(self, monkeypatch):
        stream = object()
        synchronize = Mock()
        monkeypatch.setattr(
            torch.cuda,
            "Event",
            lambda *, enable_timing: _FakeCudaEvent(),
        )
        monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

        mon = MonitoredLoader(
            [{"image": torch.zeros(2, 3, 4, 4)}],
            device="cuda:0",
            log_every=10,
        )
        next(iter(mon))
        assert mon.mark_step() is None
        synchronize.assert_not_called()

        lifetime = mon.lifetime_stats()

        synchronize.assert_called_once_with(torch.device("cuda:0"))
        assert lifetime["loader/compute_ms"] == pytest.approx(2.0)

        assert mon.lifetime_stats() == lifetime
        synchronize.assert_called_once()

    def test_skipped_mark_step_preserves_cuda_timing(self, monkeypatch):
        stream = object()
        events = []

        def make_event(*, enable_timing):
            assert enable_timing is True
            event = _FakeCudaEvent()
            events.append(event)
            return event

        synchronize = Mock()
        monkeypatch.setattr(torch.cuda, "Event", make_event)
        monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

        loader = [{"image": torch.zeros(2, 3, 4, 4)} for _ in range(2)]
        mon = MonitoredLoader(loader, device="cuda:0", log_every=1)
        iterator = iter(mon)

        next(iterator)
        next(iterator)  # Finish the first step without mark_step().
        payload = mon.mark_step()

        assert payload is not None
        synchronize.assert_called_once_with(torch.device("cuda:0"))
        assert len(events) == 4
        assert all(event.recorded_stream is stream for event in events)
        assert payload["loader/compute_ms"] == pytest.approx(2.0)
        inferred_batch_size = payload["loader/patches_per_sec"] / payload["loader/batches_per_sec"]
        assert inferred_batch_size == pytest.approx(2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_real_cuda_events(self):
        device = torch.device("cuda", torch.cuda.current_device())
        loader = [{"image": torch.ones(8, 3, 64, 64)}]
        mon = MonitoredLoader(loader, device=device, log_every=1)

        batch = next(iter(mon))
        image = batch["image"].to(device)
        _ = image.square().sum()
        payload = mon.mark_step()

        assert payload is not None
        assert payload["loader/compute_ms"] >= 0
        assert payload["loader/step_ms"] > 0


class TestWithWorkers:
    def test_num_workers_gt_zero(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=2)
        mon = MonitoredLoader(loader, log_every=3)

        for step, batch in enumerate(mon):
            payload = mon.mark_step()
            if payload is not None:
                assert LOADER_KEYS.issubset(payload.keys())
                break


class TestMarkStepNotCalled:
    def test_unmeasured_compute_counted(self):
        """If mark_step() is skipped, the time is still tracked."""
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, log_every=1)

        it = iter(mon)
        next(it)
        # Don't call mark_step — go straight to next batch.
        # The unmeasured time from batch 1 is counted as compute
        # inside __next__ when fetching batch 2.
        next(it)
        payload = mon.mark_step()

        assert payload is not None
        assert payload["loader/step_ms"] > 0
        inferred_batch_size = payload["loader/patches_per_sec"] / payload["loader/batches_per_sec"]
        assert inferred_batch_size == pytest.approx(4)
