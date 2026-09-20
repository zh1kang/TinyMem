from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import torch

from tinymem.research import storage_frontier_fit as fit_module
from tinymem.research.storage_frontier_data import StorageQuestion
from tinymem.research.storage_frontier_training import AnswerTokens


def _question(identity: str, split: str, task: str = "qa1") -> StorageQuestion:
    return StorageQuestion(identity, f"story-{identity}", task, split,
                           ("Alice is kind.", "Bob is there."), "Where is Bob?", "there")


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.tensor(1.0))

    @property
    def device(self) -> torch.device:
        return self.lora_A.device


class _FakeReader:
    def __init__(self) -> None:
        self.model = _FakeModel()


class _SpyStore:
    calls: ClassVar[list[tuple[str, int, tuple[str, ...]]]] = []

    def __init__(self, *, codec: str, budget: int, dictionary: tuple[str, ...] = ()) -> None:
        self.codec = codec
        self.budget = budget
        self.dictionary = dictionary
        _SpyStore.calls.append((codec, budget, dictionary))

    def empty(self) -> bytes:
        return b""

    def update(self, previous: bytes, record: str) -> bytes:
        value = previous + (b"\n" if previous else b"") + record.encode()
        return value[-self.budget:]

    def decode(self, payload: bytes) -> str:
        return payload.decode()


def _patch_fake_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fit_module, "attach_reader_lora", lambda reader, rank, checkpointing: None)
    monkeypatch.setattr(fit_module, "configure_read_adapter", lambda reader, trainable: (reader.model.lora_A,))
    monkeypatch.setattr(
        fit_module,
        "encode_answer",
        lambda reader, question: AnswerTokens((1,), (2,), (3,)),
    )
    monkeypatch.setattr(
        fit_module,
        "text_vectors",
        lambda reader, texts: [torch.ones((1, 1), dtype=torch.float32) for _ in texts],
    )

    def fake_answer_loss(reader, tokens, memories):
        return reader.model.lora_A.square() + memories[0].sum() * 0

    monkeypatch.setattr(fit_module, "answer_loss", fake_answer_loss)

    def fake_optimizer_step(reader, memory, adapters, optimizer, loss):
        loss.backward()
        optimizer.step()
        return {"answer_ce": float(loss.detach()), "gradient_norm": 1.0}

    monkeypatch.setattr(fit_module, "optimizer_step", fake_optimizer_step)


def test_schedule_is_fixed_by_epoch_and_index() -> None:
    assert [fit_module._noise_level(0, i) for i in range(4)] == [0, 0, 0, 0]
    assert [fit_module._noise_level(1, i) for i in range(4)] == [0, 1, 0, 1]
    assert [fit_module._noise_level(2, i) for i in range(6)] == [0, 1, 2, 0, 1, 2]


def test_answer_batch_rolls_all_histories_once(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _FakeReader()
    calls: list[int] = []

    class Memory:
        def memory_vectors(self, state: torch.Tensor) -> torch.Tensor:
            return torch.ones((2, 1, 1))

    monkeypatch.setattr(
        fit_module, "rollout",
        lambda memory, histories, cache: calls.append(len(histories)) or torch.zeros(2, 1, 1),
    )
    monkeypatch.setattr(fit_module, "answer_loss", lambda reader, tokens, memories: torch.tensor(2.0))
    value = fit_module._answer_batch(
        reader, Memory(), (("a",), ("b",)),
        (AnswerTokens((1,), (2,), (3,)), AnswerTokens((1,), (2,), (3,))), {},
    )

    assert value.item() == 2.0
    assert calls == [2]


def test_fit_writes_fixed_baseline_bundle_and_uses_text_only_store(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    _patch_fake_reader(monkeypatch)
    _SpyStore.calls.clear()
    training = (_question("train-0", "train"), _question("train-1", "train"))
    validation = (_question("valid-0", "validation"),)
    reader = _FakeReader()
    output = tmp_path / "run"
    report = fit_module.fit(
        reader, None, {"Alice is kind.": torch.ones((1, 1)), "Bob is there.": torch.ones((1, 1))},
        training, validation, {"train": ("Noise only.",), "validation": ("Validation noise.",)},
        ("Alice is kind.",), {"epochs": 1, "batch_size": 1, "probe_per_task": 1,
                               "schedule_seed": 7}, output, 11, text_store_factory=_SpyStore,
    )

    assert report["mode"] == "strong_text"
    assert report["supervision"] == "answers only"
    assert report["official_test_loaded"] is False
    assert report["steps"] == 2
    assert report["base_hash_before"] == report["base_hash_after"]
    assert report["conditions"]["full_text"] == 1
    assert len(_SpyStore.calls) == 9
    assert {call[0] for call in _SpyStore.calls} == {
        "compressed_recent", "compressed_diverse", "dictionary_recent",
    }
    assert (output / "epoch0" / "checkpoint.safetensors").exists()
    assert (output / "epoch1" / "checkpoint.safetensors").exists()
    assert (output / "final" / "checkpoint.safetensors").exists()
    assert (output / "checkpoint.safetensors").exists()
    assert len((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / "epoch_curves.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_fit_refuses_to_overwrite_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        fit_module.fit(None, None, {}, (), (), {}, (), {}, output, 1)


@pytest.mark.parametrize('learned', [False, True])
def test_complete_fit_with_native_qwen_and_checkpoint(monkeypatch, tmp_path, learned):
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from tinymem.memory.quantized_slots import QuantizedSlotMemory
    from tinymem.research.pretrained import PretrainedReader

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [3 + ord(c) % 60 for c in text]

    torch.set_num_threads(1)
    torch.manual_seed(31)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=1024)
    reader = PretrainedReader(Qwen3ForCausalLM(config), Tokenizer())
    monkeypatch.setattr(fit_module, 'encode_answer', lambda reader, q: AnswerTokens((1,), (2,), (3, 4)))
    memory = QuantizedSlotMemory(32, slots=2) if learned else None
    training = (_question('train-a', 'train'), _question('train-b', 'train'))
    validation = (_question('validation-a', 'validation'),)
    texts = ('Alice is kind.', 'Bob is there.', 'Noise only.', 'Validation noise.')
    features = {s: torch.randn(3, 32) for s in texts}
    output = tmp_path / 'fit'
    result = fit_module.fit(reader, memory, features, training, validation,
                            {'train': ('Noise only.',), 'validation': ('Validation noise.',)},
                            tuple(texts[:3]), {'epochs': 1, 'batch_size': 2, 'probe_per_task': 1,
                                              'schedule_seed': 7, 'noise_seed': 11}, output, 23)
    assert result['steps'] == 1
    assert result['base_hash_before'] == result['base_hash_after']
    saved = load_file(str(output / 'checkpoint.safetensors'))
    for name, parameter in reader.model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(saved['adapter.' + name], parameter)
    if learned:
        restored = QuantizedSlotMemory(32, slots=2)
        restored.load_state_dict({k.removeprefix('writer.'): v for k, v in saved.items() if k.startswith('writer.')})
        state = fit_module.rollout(memory, (training[0].records,), features)
        replayed = fit_module.rollout(restored, (training[0].records,), features)
        assert memory.pack(state) == restored.pack(replayed)
