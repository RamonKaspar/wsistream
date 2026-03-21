"""Tests for MonitoredLoader."""

from __future__ import annotations

import time

from torch.utils.data import DataLoader

from tests.conftest import FakeBackend
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.torch import MonitoredLoader, WsiStreamDataset


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
    def test_no_cuda_sync_without_device(self):
        dataset = _make_dataset()
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        mon = MonitoredLoader(loader, device=None, log_every=1)

        next(iter(mon))
        payload = mon.mark_step()
        assert payload is not None


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
