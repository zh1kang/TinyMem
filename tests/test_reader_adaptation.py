from collections import Counter
from copy import deepcopy

import pytest
import torch

from tinymem.data.reader_adaptation import make_reader_adaptation_data
from tinymem.data.reader_gate import BABI_GATE_FILES, ReaderCase
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.reader_adaptation import (
    ReaderAnswerTokens, attach_reader_lora, encode_reader_answer, reader_answer_loss,
)


def test_adaptation_groups_duplicate_contexts_and_excludes_gate_episodes(tmp_path):
    for task, filename in BABI_GATE_FILES.items():
        episodes = []
        for index in [*range(100), 0]:
            person = f"Person{index}"
            if task == "qa1":
                text = f"1 {person} moved to the kitchen.\n2 Where is {person}?\tkitchen\t1\n"
            else:
                context = f"1 {person} moved to the kitchen.\n2 {person} got the apple there.\n3 {person} moved to the hallway.\n"
                question = "Where is the apple?\thallway\t2 3" if task == "qa2" else "Where was the apple before the hallway?\tkitchen\t2 3 1"
                text = context + f"4 {question}\n"
            episodes.append(text)
        (tmp_path / filename).write_text("".join(episodes))
    gate = [ReaderCase(
        "gate", "babi_qa2", f"{BABI_GATE_FILES['qa2']}:episode-000001",
        "Person0 moved to the kitchen.\nPerson0 got the apple there.\nPerson0 moved to the hallway.",
        "Where is the apple?", "hallway",
    )]
    data = make_reader_adaptation_data(tmp_path, gate, train_per_task=6, development_per_task=3, replacement_examples=20)
    assert data == make_reader_adaptation_data(tmp_path, gate, train_per_task=6, development_per_task=3, replacement_examples=20)
    assert len(data.excluded_episodes) == 4
    assert {row.context for row in data.train}.isdisjoint(row.context for row in data.development)
    assert {row.history_id for row in data.train}.isdisjoint(row.history_id for row in data.development)
    assert all(row.context != gate[0].context for row in (*data.train, *data.development))
    for rows in (data.train, data.development):
        corrections = [row for row in rows if row.category.startswith("correction_")]
        counts = Counter(row.category for row in corrections)
        assert counts["correction_changed"] == counts["correction_unchanged"]
        assert all(value == 2 for value in Counter(row.history_id for row in corrections).values())


@pytest.fixture
def adaptation_reader():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("peft")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4, "kitchen": 5, "bathroom": 6}, unk_token="[UNK]",
    ))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", padding_side="left",
        chat_template="hello world",
    )
    torch.manual_seed(4)
    config = transformers.Qwen3Config(vocab_size=7, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64, eos_token_id=2, pad_token_id=0)
    return PretrainedReader(transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False), tokenizer)


def test_answer_position_loss_matches_standard_shifted_hf_loss(adaptation_reader):
    reader = adaptation_reader
    example = ReaderAnswerTokens("id", (3, 4, 3), (5, 2))
    efficient = reader_answer_loss(reader, example)
    full = torch.tensor([[*example.prompt_ids, *example.answer_ids]])
    labels = full.clone()
    labels[:, :len(example.prompt_ids)] = -100
    standard = reader.model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels, use_cache=False).loss
    torch.testing.assert_close(efficient, standard)
    case = ReaderCase("id", "babi_qa1", "history", "hello", "world", "kitchen")
    encoded = encode_reader_answer(reader, case)
    assert encoded.prompt_ids == (3, 4)
    assert encoded.answer_ids == (5, 2)
    reader.model.config.max_position_embeddings = 2
    with pytest.raises(ValueError, match="truncation"):
        encode_reader_answer(reader, case)


def test_only_adapters_change_and_checkpoint_reloads_exactly(adaptation_reader, tmp_path):
    from peft import PeftModel

    reader = adaptation_reader
    original_base = deepcopy(reader.model)
    attach_reader_lora(reader, rank=2, checkpointing=False)
    frozen = {name: value.detach().clone() for name, value in reader.model.named_parameters() if not value.requires_grad}
    trainable = [(name, value) for name, value in reader.model.named_parameters() if value.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    before = {name: value.detach().clone() for name, value in trainable}
    optimizer = torch.optim.AdamW([value for _, value in trainable], lr=0.01)
    example = ReaderAnswerTokens("id", (3, 4, 3), (5, 2))
    loss = reader_answer_loss(reader, example)
    loss.backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for _, value in trainable)
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in trainable)
    for name, value in reader.model.named_parameters():
        if name in frozen:
            assert value.grad is None
            assert torch.equal(value, frozen[name])
    reader.model.eval().save_pretrained(tmp_path, save_embedding_layers=False)
    reloaded = PretrainedReader(PeftModel.from_pretrained(original_base, tmp_path).eval(), reader.tokenizer)
    torch.testing.assert_close(reader_answer_loss(reader, example), reader_answer_loss(reloaded, example), rtol=0, atol=0)


def test_checkpointing_preserves_lora_gradients(adaptation_reader):
    reader = adaptation_reader
    attach_reader_lora(reader, rank=2, checkpointing=False)
    example = ReaderAnswerTokens("id", (3, 4, 3), (5, 2))
    reader_answer_loss(reader, example).backward()
    expected = {name: parameter.grad.clone() for name, parameter in reader.model.named_parameters() if parameter.requires_grad}
    reader.model.zero_grad(set_to_none=True)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader_answer_loss(reader, example).backward()
    for name, parameter in reader.model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name])


def test_frozen_reader_still_trains_memory_prefix_with_checkpointing(adaptation_reader):
    reader = adaptation_reader
    prefix = torch.randn(1, 2, reader.model.config.hidden_size, requires_grad=True)
    tokens = torch.tensor([[3, 4]])
    embedded = reader.model.get_input_embeddings()(tokens)

    def backward():
        output = reader.model(inputs_embeds=torch.cat((prefix, embedded), dim=1), use_cache=False, logits_to_keep=1)
        torch.nn.functional.cross_entropy(output.logits[:, -1].float(), torch.tensor([5])).backward()

    backward()
    expected = prefix.grad.clone()
    assert expected.abs().sum() > 0
    prefix.grad = None
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    backward()
    torch.testing.assert_close(prefix.grad, expected)
    assert all(parameter.grad is None for parameter in reader.model.parameters())


@pytest.mark.parametrize("profile", [False, True])
def test_training_runner_records_protocol_before_backward_and_saves_adapter(adaptation_reader, tmp_path, monkeypatch, profile):
    import hashlib
    import json
    import sys
    from dataclasses import asdict

    from peft import PeftModel

    from scripts import train_reader_adapter as runner
    from tinymem.data.reader_adaptation import ReaderAdaptationData

    case = ReaderCase("train", "babi_qa1", "train-history", "hello", "world", "kitchen")
    development = ReaderCase("dev", "babi_qa1", "dev-history", "world", "hello", "bathroom")
    data = ReaderAdaptationData((case,), (development,), ("excluded",))
    gate = tmp_path / "gate"
    gate.mkdir()
    for filename in BABI_GATE_FILES.values():
        (tmp_path / filename).write_text("fixture")
    source_hashes = {filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() for filename in BABI_GATE_FILES.values()}
    manifest = json.dumps({"cases": [{"case": asdict(case), "condition": "full_context"}], "babi_source_sha256": source_hashes})
    (gate / "data_manifest.json").write_text(manifest)
    (gate / "protocol.json").write_text(json.dumps({"data_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}))
    output = tmp_path / "output"
    base = deepcopy(adaptation_reader.model)
    original_loss = runner.reader_answer_loss
    calls = []

    def checked_loss(reader, example):
        protocol_path, = output.glob("*/protocol.json")
        protocol = json.loads(protocol_path.read_text())
        manifest_path = protocol_path.parent / "data_manifest.json"
        assert protocol["data_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        calls.append(torch.is_grad_enabled())
        return original_loss(reader, example)

    monkeypatch.setattr(runner, "make_reader_adaptation_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(runner, "verify_qwen_snapshot", lambda _: {"fixture": True})
    monkeypatch.setattr(runner, "load_qwen_reader", lambda *args, **kwargs: adaptation_reader)
    monkeypatch.setattr(runner, "reader_answer_loss", checked_loss)
    arguments = ["train", "--gate-run", str(gate), "--babi-root", str(tmp_path), "--device", "cpu", "--steps", "2",
                 "--gradient-accumulation", "2", "--evaluate-every", "1", "--rank", "2", "--artifact-root", str(output)]
    monkeypatch.setattr(sys, "argv", arguments + (["--profile-only"] if profile else []))
    runner.main()
    result_path, = output.glob("*/results.json")
    result = json.loads(result_path.read_text())
    protocol = json.loads((result_path.parent / "protocol.json").read_text())
    assert result["steps"] == protocol["effective_optimizer_steps"] == (3 if profile else 2)
    assert protocol["effective_gradient_accumulation"] == (1 if profile else 2)
    assert not result["gate_evaluated"]
    assert all(row["gradient_norm"] > 0 for row in result["metrics"])
    assert calls.count(True) == (3 if profile else 4)
    assert calls.count(False) == (0 if profile else 2)
    if profile:
        assert result["best_checkpoint"] is None
        assert not list(output.glob("*/*/adapter_model.safetensors"))
    else:
        assert result["best_internal_macro_ce"] == min(row["development_macro_ce"] for row in result["metrics"])
        from pathlib import Path

        adapter_protocol = json.loads((Path(result["best_checkpoint"]) / "reader_adapter_protocol.json").read_text())
        assert adapter_protocol["excluded_gate_manifest_sha256"] == hashlib.sha256(manifest.encode()).hexdigest()
        assert adapter_protocol["training_protocol_sha256"] == hashlib.sha256((result_path.parent / "protocol.json").read_bytes()).hexdigest()
        reloaded = PretrainedReader(PeftModel.from_pretrained(base, result["best_checkpoint"]).eval(), adaptation_reader.tokenizer)
        selected_loss = reader_answer_loss(reloaded, encode_reader_answer(reloaded, development))
        assert float(selected_loss) == pytest.approx(result["best_internal_macro_ce"])


@pytest.mark.parametrize("change", ["manifest", "source"])
def test_training_runner_rejects_changed_gate_before_loading_model(tmp_path, monkeypatch, change):
    import hashlib
    import json
    import sys

    from scripts import train_reader_adapter as runner

    manifest = json.dumps({"babi_source_sha256": {}})
    (tmp_path / "data_manifest.json").write_text(manifest)
    digest = "incorrect" if change == "manifest" else hashlib.sha256(manifest.encode()).hexdigest()
    (tmp_path / "protocol.json").write_text(json.dumps({"data_manifest_sha256": digest}))
    for filename in BABI_GATE_FILES.values():
        (tmp_path / filename).write_text("changed source")
    monkeypatch.setattr(sys, "argv", ["train", "--gate-run", str(tmp_path), "--babi-root", str(tmp_path)])
    monkeypatch.setattr(runner, "load_qwen_reader", lambda *args, **kwargs: pytest.fail("must not load model"))
    with pytest.raises(ValueError, match="hash" if change == "manifest" else "sources differ"):
        runner.main()


@pytest.mark.parametrize("wrong_gate", [False, True])
def test_gate_runner_loads_and_fingerprints_selected_adapter(adaptation_reader, tmp_path, monkeypatch, wrong_gate):
    import hashlib
    import json
    import sys

    from peft import PeftModel

    from scripts import evaluate_reader_gate as runner
    from tinymem.evaluation.reader_gate import reader_messages
    from dataclasses import asdict

    base = deepcopy(adaptation_reader.model)
    attach_reader_lora(adaptation_reader, rank=2, checkpointing=False)
    adapter = tmp_path / "adapter"
    adaptation_reader.model.save_pretrained(adapter, save_embedding_layers=False)
    for filename in BABI_GATE_FILES.values():
        (tmp_path / filename).write_text("fixture")
    output = tmp_path / "output"
    case = ReaderCase("id", "babi_qa1", "history", "hello", "world", "kitchen")
    prepared = []
    for condition in ("full_context", "question_only"):
        messages = reader_messages(case, condition=condition)
        prompt = adaptation_reader.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prepared.append({"case": asdict(case), "condition": condition, "messages": messages, "prompt": prompt,
                         "prompt_ids": adaptation_reader.tokenizer.encode(prompt, add_special_tokens=False)})
    manifest = json.dumps({
        "split": "development_only", "cases": prepared,
        "babi_source_sha256": {filename: hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() for filename in BABI_GATE_FILES.values()},
        "replacement_protocol": "history_disjoint_v2", "replacement_split": "validation",
    }, sort_keys=True, indent=2) + "\n"
    (adapter / "reader_adapter_protocol.json").write_text(json.dumps({
        "protocol": "visible_reader_adapter_v1",
        "excluded_gate_manifest_sha256": "wrong" if wrong_gate else hashlib.sha256(manifest.encode()).hexdigest(),
    }))
    loaded = PretrainedReader(base, adaptation_reader.tokenizer)
    original_generate = loaded.generate

    def checked_generate(prompts, *, max_new_tokens):
        assert isinstance(loaded.model, PeftModel)
        assert not loaded.model.training
        assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
        protocol_path, = output.glob("*/protocol.json")
        protocol = json.loads(protocol_path.read_text())
        for name, digest in protocol["adapter_sha256"].items():
            assert digest == hashlib.sha256((adapter / name).read_bytes()).hexdigest()
        return original_generate(prompts, max_new_tokens=max_new_tokens)

    monkeypatch.setattr(runner, "make_reader_gate_cases", lambda *args, **kwargs: [case])
    monkeypatch.setattr(runner, "verify_qwen_snapshot", lambda _: {"fixture": True})
    monkeypatch.setattr(runner, "load_qwen_reader", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(loaded, "generate", checked_generate)
    monkeypatch.setattr(sys, "argv", ["gate", "--babi-root", str(tmp_path), "--adapter", str(adapter), "--device", "cpu",
                                     "--artifact-root", str(output), "--max-new-tokens", "2"])
    if wrong_gate:
        with pytest.raises(ValueError, match="exact gate excluded"):
            runner.main()
        assert not list(output.glob("*/predictions.jsonl"))
        return
    runner.main()
    result_path, = output.glob("*/results.json")
    result = json.loads(result_path.read_text())
    assert set(result["by_condition"]) == {"full_context", "question_only"}
