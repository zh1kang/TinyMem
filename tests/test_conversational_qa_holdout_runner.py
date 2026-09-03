from pathlib import Path

from scripts.evaluate_conversational_qa import load_babi_test_examples


def test_holdout_loader_uses_only_test_examples() -> None:
    examples = load_babi_test_examples(
        Path("data/raw/tasks_1-20_v1-2/en-valid-10k"),
        tasks=("qa1", "qa2"),
        examples_per_task=3,
        seed=7,
    )

    assert len(examples) == 6
    assert {example.task_id for example in examples} == {"qa1", "qa2"}
    assert {example.split for example in examples} == {"test"}
