# Tests

Run tests with the project interpreter:

```bash
.venv/bin/python -m pytest
```

The native memory tests use tiny random models or small tensors. They verify state ownership, byte accounting, query-independent writes, active frozen adapters, recurrent gradients, native prompt parity, fixed-readout fitting, and paired analysis. They are not scientific Qwen accuracy results.

The focused baseline command for the memory-update work is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -o addopts='' -q -p no:cacheprovider \
  tests/test_query_pool_slots.py tests/test_mean_pool_slots.py \
  tests/test_native_training.py tests/test_native_recurrent_memory.py \
  tests/test_prefix_reader.py tests/test_fixed_projection_memory_oracle.py \
  tests/test_fingerprint_facts.py tests/test_association_diagnostics.py
```

Full-size model smoke and input-only validation are separate:

```bash
.venv/bin/python -m scripts.opaque.smoke --check-only
```

This command requires staged existing-study artifacts, verifies their identities, and does not load the model or inspect confirmation answers. Use the Della runbook for GPU execution.

Keep frozen historical tests and runner sources unchanged. The unrelated working-tree educational decoder edit can affect the full suite; do not silently restore it to obtain a green run. Report focused native results and clean-export verification separately.
