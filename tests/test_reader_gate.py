from collections import Counter
from dataclasses import replace

import pytest

from tinymem.data.reader_gate import BABI_GATE_FILES, ReaderCase, make_reader_gate_cases
from tinymem.evaluation.reader_gate import reader_exact_match, reader_messages, summarize_reader_predictions


@pytest.fixture
def babi_gate_root(tmp_path):
    texts = {
        "qa1": "1 Mary moved to the kitchen.\n2 Sandra moved to the bathroom.\n3 Where is Mary?\tkitchen\t1\n4 Where is Sandra?\tbathroom\t2\n",
        "qa2": "1 Mary moved to the kitchen.\n2 Mary got the apple there.\n3 Sandra moved to the bathroom.\n4 Where is the apple?\tkitchen\t2 1\n",
        "qa3": "1 Mary moved to the kitchen.\n2 Mary got the apple.\n3 Mary moved to the hallway.\n4 Sandra moved to the bathroom.\n5 Where was the apple before the hallway?\tkitchen\t2 3 1\n",
    }
    for task, filename in BABI_GATE_FILES.items():
        (tmp_path / filename).write_text(texts[task] * 3)
    return tmp_path


def test_gate_is_deterministic_grouped_and_preserves_full_context(babi_gate_root):
    cases = make_reader_gate_cases(babi_gate_root, count=2, replacement_count=4, copy_count=2)
    assert cases == make_reader_gate_cases(babi_gate_root, count=2, replacement_count=4, copy_count=2)
    assert Counter(row.category for row in cases) == {
        "babi_qa1": 2, "babi_qa2": 2, "babi_qa3": 2,
        "correction_changed": 2, "correction_unchanged": 2,
        "exact_copy": 2, "randomized_bindings": 2,
    }
    for category in ("babi_qa1", "babi_qa2", "babi_qa3"):
        rows = [row for row in cases if row.category == category]
        assert len({row.history_id for row in rows}) == 2
        assert all("Sandra moved to the bathroom." in row.context for row in rows)
    corrections = [row for row in cases if row.category.startswith("correction_")]
    assert len({row.history_id for row in corrections}) == 2
    assert all("Correction:" in row.context for row in corrections)


def test_gate_does_not_open_test_files_and_validates_source_labels(babi_gate_root):
    for filename in BABI_GATE_FILES.values():
        assert filename.endswith("_train.txt")
        (babi_gate_root / filename.replace("_train", "_test")).write_text("invalid forbidden test data")
    make_reader_gate_cases(babi_gate_root, count=1, replacement_count=2, copy_count=1)
    path = babi_gate_root / BABI_GATE_FILES["qa2"]
    path.write_text(path.read_text().replace("\tkitchen\t", "\tbedroom\t"))
    with pytest.raises(ValueError, match="oracle"):
        make_reader_gate_cases(babi_gate_root, count=1, replacement_count=2, copy_count=1)


def test_insufficient_episodes_and_unbalanced_correction_count_fail(babi_gate_root):
    with pytest.raises(ValueError, match="not enough"):
        make_reader_gate_cases(babi_gate_root, count=4)
    with pytest.raises(ValueError, match="even"):
        make_reader_gate_cases(babi_gate_root, replacement_count=3)


def test_messages_never_use_answers_and_query_only_removes_all_history():
    case = ReaderCase("id", "babi_qa1", "world", "Mary moved to the kitchen.", "Where is Mary?", "kitchen")
    for condition in ("full_context", "question_only"):
        assert reader_messages(case, condition=condition) == reader_messages(replace(case, answer="secret label"), condition=condition)
    full = reader_messages(case, condition="full_context")
    query = reader_messages(case, condition="question_only")
    assert case.context in full[1]["content"]
    assert case.context not in query[1]["content"]
    assert case.question in query[1]["content"]
    with pytest.raises(ValueError, match="condition"):
        reader_messages(case, condition="oracle")


@pytest.mark.parametrize("category", ("exact_copy", "randomized_bindings"))
def test_exact_copy_scoring_does_not_normalize_content(category):
    assert reader_exact_match(" Ab-cD\n", "Ab-cD", category)
    assert not reader_exact_match("ab-cd", "Ab-cD", category)
    assert not reader_exact_match("Ab cD", "Ab-cD", category)
    assert not reader_exact_match("The code is Ab-cD", "Ab-cD", category)


def test_gate_requires_sample_size_and_each_correction_subgroup():
    categories = ("babi_qa1", "correction_changed", "correction_unchanged")
    rows = [{"case_id": f"{category}-{index}", "category": category, "condition": "full_context", "answer": "kitchen", "prediction": "Kitchen."}
            for category in categories for index in range(100)]
    assert summarize_reader_predictions(rows)["gate_passed"]
    assert not summarize_reader_predictions(rows[:-1])["gate_passed"]
    for row in rows[-6:]:
        row["prediction"] = "bathroom"
    result = summarize_reader_predictions(rows)
    assert not result["gate_passed"]
    assert result["by_condition"]["full_context"]["correction_unchanged"]["exact_accuracy"] == 0.94
    assert result["by_condition"]["full_context"]["babi_qa1"]["majority_answer_prior"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        summarize_reader_predictions([rows[0], rows[0]])


def test_runner_freezes_manifest_before_first_generation(babi_gate_root, tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    import torch

    from scripts import evaluate_reader_gate as runner

    output = tmp_path / "results"
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert not tokenize and add_generation_prompt and not enable_thinking
            return messages[1]["content"]

        def encode(self, prompt, *, add_special_tokens):
            assert not add_special_tokens
            return list(prompt.encode())

    class Reader:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.model = torch.nn.Linear(1, 1)
            self.model.config = SimpleNamespace(max_position_embeddings=4096, to_dict=lambda: {"fixture": True})

        def generate(self, prompts, *, max_new_tokens):
            manifest_path, = output.glob("*/data_manifest.json")
            protocol = json.loads((manifest_path.parent / "protocol.json").read_text())
            assert protocol["data_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            manifest = json.loads(manifest_path.read_text())
            assert all(prompt in {row["prompt"] for row in manifest["cases"]} for prompt in prompts)
            return [{"prediction": "kitchen", "prompt_tokens": len(prompt), "generated_ids": [1]} for prompt in prompts]

    monkeypatch.setattr(runner, "load_qwen_reader", lambda *args, **kwargs: Reader())
    monkeypatch.setattr(runner, "verify_qwen_snapshot", lambda _: {"fixture": True})
    monkeypatch.setattr(sys, "argv", ["gate", "--babi-root", str(babi_gate_root), "--examples-per-task", "1", "--replacement-examples", "2", "--copy-examples", "1", "--device", "cpu", "--artifact-root", str(output)])
    runner.main()
    result_path, = output.glob("*/results.json")
    result = json.loads(result_path.read_text())
    assert not result["gate_passed"]
    assert set(result["by_condition"]) == {"full_context", "question_only"}
    assert len((result_path.parent / "predictions.jsonl").read_text().splitlines()) == 14
