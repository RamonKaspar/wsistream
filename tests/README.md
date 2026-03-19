# Running tests

```bash
# Unit tests (no WSI files needed, uses FakeBackend)
pytest tests/ -m "not e2e and not benchmark"

# End-to-end tests (requires real WSI files)
pytest tests/test_e2e.py --slide-dir /path/to/slides --backend openslide -v

# Benchmarks (requires real WSI files)
pytest tests/test_benchmarks.py --slide-dir /path/to/slides --backend openslide -v -s

# Everything
pytest tests/ --slide-dir /path/to/slides --backend openslide -v
```

Tests marked `e2e` or `benchmark` are skipped automatically when `--slide-dir` is not provided.
