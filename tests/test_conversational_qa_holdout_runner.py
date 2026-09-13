import json
from pathlib import Path

from scripts.evaluate_conversational_qa import (
    build_babi_provenance,
    load_babi_test_examples,
)


def test_holdout_loader_uses_only_test_examples(tmp_path: Path) -> None:
    repository_root = tmp_path
    data_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    data_root.mkdir(parents=True)
    (repository_root / "data/manifest.json").write_text(json.dumps({
        "datasets": {"babi": {
            "revision": "synthetic-babi-test-v1",
            "source": "test://synthetic-babi",
        }},
    }))
    for task in ("qa1", "qa2"):
        episodes = []
        for index in range(3):
            episodes.extend([
                "1 Mary went to the hallway.",
                "2 Where is Mary?\thallway\t1",
            ])
        (data_root / f"{task}_test.txt").write_text("\n".join(episodes) + "\n")

    examples = load_babi_test_examples(
        data_root,
        tasks=("qa1", "qa2"),
        examples_per_task=3,
        seed=7,
    )

    assert len(examples) == 6
    assert {example.task_id for example in examples} == {"qa1", "qa2"}
    assert {example.split for example in examples} == {"test"}

    provenance = build_babi_provenance(
        repository_root,
        tasks=("qa1", "qa2"),
        examples=examples,
    )
    assert provenance["revision"] == "synthetic-babi-test-v1"
    assert [source["selected_examples"] for source in provenance["test_files"]] == [
        3,
        3,
    ]
    assert all(
        len(source["sha256"]) == 64 for source in provenance["test_files"]
    )
