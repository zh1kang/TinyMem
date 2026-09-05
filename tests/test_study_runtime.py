from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from scripts.opaque.smoke import backward_check
from tinymem.research import study_runtime as runtime
from tinymem.research.memory_prompt import NativeMemoryExample
from tinymem.research.native_training import native_history_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.recurrent_memory import NativeRecurrentMemory


def test_paths_relocate_only_the_known_root(tmp_path):
    assert runtime.repository_path("src/file.py", root=tmp_path) == Path("src/file.py")
    assert runtime.repository_path(tmp_path / "src/file.py", root=tmp_path) == Path("src/file.py")
    assert runtime.repository_path("/Users/caleb/TinyMem/src/file.py", root=tmp_path) == Path("src/file.py")
    for value in ("../outside", "src/../../outside", "/other/project/file.py"):
        with pytest.raises(ValueError):
            runtime.repository_path(value, root=tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runtime.repository_path("escape/file.py", root=tmp_path)


def test_missing_source_and_changed_execution_fail():
    with pytest.raises(FileNotFoundError):
        runtime.sha256("does-not-exist-in-this-repository.py")
    record = runtime.execution_record(torch.device("cpu"))
    runtime.validate_execution(record, deepcopy(record))
    changed = deepcopy(record)
    changed["source_sha256"]["src/tinymem/research/study_runtime.py"] = "0" * 64
    with pytest.raises(ValueError, match="sources changed"):
        runtime.validate_execution(changed)
    changed = deepcopy(record)
    changed["runtime"]["device"] = "cuda"
    changed["runtime"].update(cuda_build="test", gpu_name="test", gpu_capability=[8, 0], cublas_workspace_config=":4096:8")
    with pytest.raises(ValueError, match="do not mix"):
        runtime.validate_execution(changed, record)
    with pytest.raises(ValueError, match="original MPS"):
        runtime.validate_execution({})
    del record["runtime"]
    with pytest.raises(ValueError, match="runtime is incomplete"):
        runtime.validate_execution(record)


def test_portable_sources_are_saved_with_original_sources():
    protocol, sources = {}, [Path("src/tinymem/research/prefix_reader.py")]
    runtime.attach_execution(protocol, sources, torch.device("cpu"))
    assert set(protocol["execution"]["source_sha256"]) == set(runtime.PORTABLE_SOURCES)
    assert all(name in protocol["source_sha256"] for name in runtime.PORTABLE_SOURCES)
    assert len({path.name for path in sources}) == len(sources)


def test_cuda_dispatch_never_calls_mps(monkeypatch):
    observed = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: pytest.fail("MPS used for CUDA"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: observed.append(device))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 11)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 22)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 33)
    device = torch.device("cuda")
    runtime.synchronize(device)
    assert observed == [device]
    assert runtime.allocation_metrics(device) == {
        "cuda_allocated_bytes": 11, "cuda_reserved_bytes": 22, "cuda_peak_allocated_bytes": 33}
    runtime.synchronize(torch.device("cpu"))
    assert runtime.allocation_metrics(torch.device("cpu")) == {}


def test_unavailable_cuda_fails_without_fallback(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        runtime.prepare_device("cuda")


@pytest.mark.parametrize("case", ["wrong_config", "initialized", "old_gpu"])
def test_invalid_cuda_numerics_fail(monkeypatch, case):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: case == "initialized")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    if case == "wrong_config":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS|BF16"):
        runtime.prepare_device("cuda")


@pytest.mark.parametrize("kind", ["query_pool", "mean_pool"])
def test_smoke_loss_and_gradients_match_production_on_cpu(kind):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(19)
    reader = PretrainedReader(transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
        attention_dropout=0.0,
    )).requires_grad_(False), None)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    writer = NativeRecurrentMemory(16, memory_width=8, slots=2, segment_length=2, writer_kind=kind)
    reference = deepcopy(writer)
    frozen_reader = deepcopy(reader.model.state_dict())
    first = NativeMemoryExample("first", (1, 3), (6, 7, 8, 9), (4, 5), (10, 2))
    examples = [first, replace(first, case_id="second", after_ids=(4, 11), answer_ids=(12, 2))]
    chunks = ((6,), (7,), (8,), (9,))
    expected = native_history_answer_loss(reader, reference, examples, history_chunks=chunks)
    expected.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    actual = backward_check(reader, writer, examples, chunks)
    assert actual["answer_ce"] == pytest.approx(float(expected.detach()), abs=1e-6)
    assert actual["state_bytes"] == 66
    assert len(actual["state_gradient_norms"]) == 4
    assert all(value > 0 for value in actual["state_gradient_norms"])
    for left, right in zip(writer.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
    assert all(torch.equal(value, frozen_reader[name]) for name, value in reader.model.state_dict().items())
