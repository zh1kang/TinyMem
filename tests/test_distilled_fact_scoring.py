import json

import pytest
import torch

pytest_plugins = ["test_distilled_fact_protocol"]

from tinymem.research import (
    distilled_fact_fit,
    distilled_fact_protocol,
    distilled_fact_report,
)
from tinymem.research import distilled_fact_scoring as scoring
from tinymem.research.delta_fact_data import build_dataset, replay
from tinymem.research.delta_fact_evaluation import StateRecord
from tinymem.research.delta_fact_profile import file_hash
from tinymem.research.delta_fact_protocol import cell_identity, seal_directory


def _record(episode, *, value=0.5, endpoint=8):
    truth = tuple(int(value) for value in replay(episode.prefix))
    values = torch.zeros((1, 2, 32), dtype=torch.float32)
    values[0, 0, [0, 8, 16, 24]] = value
    valid = torch.ones((1, 2), dtype=torch.bool)
    return StateRecord(episode.id, episode.prefix_id, episode.split, episode.wording,
                       episode.condition, episode.target, endpoint, values, valid, truth)


def test_exact_zero_is_reported_as_undefined_without_state_repair():
    dataset = build_dataset(seed=7, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(e for e in dataset.test if e.condition == "no_write")
    record = _record(episode, value=0.0)

    assert scoring._direct_bits(record) == [None, None, None, None]
    metrics = scoring._direct_bit_metrics(record, record.truth)
    assert metrics["known_fact_correct"] == 0
    assert metrics["known_fact_count"] == 4
    assert torch.equal(record.values[0, 0, [0, 8, 16, 24]], torch.zeros(4))


def test_early_trajectory_truth_keeps_unknown_facts_out_of_the_denominator():
    dataset = build_dataset(seed=13, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(e for e in dataset.test if e.condition == "repeat" and e.wording == "familiar")
    target = scoring._oracle_target_record(episode, 1)
    metrics = scoring._direct_bit_metrics(target, scoring._truth(episode, 1))

    assert metrics["known_fact_count"] == 1
    assert metrics["known_fact_correct"] == 1
    assert metrics["truth"].count(None) == 3


def test_read_metadata_recomputes_literal_correctness():
    read = {"prediction": "bathroom", "generated_ids": [1], "input_positions": 8,
            "memory_positions": 2, "native_envelope_tokens": 6, "correct": False}
    with pytest.raises(ValueError, match="correctness"):
        scoring._read_metadata(read, "bathroom", "update_known")


def test_record_map_rejects_duplicate_and_missing_endpoint_rows():
    dataset = build_dataset(seed=11, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(e for e in dataset.test if e.condition == "no_write")
    record = _record(episode)

    with pytest.raises(ValueError, match="duplicate"):
        scoring._record_map((record, record), (episode,))
    with pytest.raises(ValueError, match="omit or add"):
        scoring._record_map((), (episode,))

    bad_truth = _record(episode)
    bad_truth = StateRecord(bad_truth.episode_id, bad_truth.prefix_id, bad_truth.split,
                            bad_truth.wording, bad_truth.condition, bad_truth.target,
                            bad_truth.after_write, bad_truth.values, bad_truth.valid,
                            tuple(1 - value for value in bad_truth.truth))
    with pytest.raises(ValueError, match="truth"):
        scoring._record_map((bad_truth,), (episode,))


def test_bootstrap_allows_different_wording_predictions_and_averages_them():
    rows = []
    for seed in ("3101", "3102"):
        for prefix in ("p0", "p1"):
            for wording, real in (("familiar", True), ("heldout", False)):
                rows.append({"writer_seed": seed, "prefix_id": prefix, "wording": wording,
                             "after_write": 16, "condition": "repeat", "scope": "unspoken",
                             "reads": {"real": {"correct": real}, "zero": {"correct": False},
                                       "oracle": {"correct": False}, "donor": {"correct": False}}})
    result = scoring._paired_bootstrap(
        rows, "zero", condition="repeat", scope="unspoken",
        settings={"bootstrap_samples": 17, "bootstrap_seed": 3},
    )
    assert result["mean_gap_pp"] == 50.0
    assert result["wordings_averaged_within_prefix"] == ["familiar", "heldout"]


@pytest.mark.parametrize(
    ("fixed_beta", "normalize_hidden"), [(None, False), (0.75, False), (0.75, True)],
)
def test_tiny_study_scores_and_aggregates_all_read_rows(writer_study, fixed_beta, normalize_hidden):
    root, study, protocol, fresh = writer_study
    if fixed_beta is not None:
        parent = study / "parent"
        study = study.with_name("fixed-study")
        protocol = distilled_fact_protocol.prepare_study(
            root, study, protocol["snapshot"],
            distilled_fact_protocol.settings(device="cpu", smoke=True, fixed_beta=fixed_beta,
                                             normalize_hidden=normalize_hidden), parent,
        )
    dataset = distilled_fact_protocol.read_dataset(study / "dataset.json")
    distilled_fact_protocol.prepare_features(fresh(), study, protocol, dataset)
    distilled_fact_fit.train_cell(study, protocol, dataset, 0)
    distilled_fact_protocol.seal_training(study, protocol)

    cell_report = scoring.score_cell(fresh(), study, protocol, dataset, 0)
    aggregate = distilled_fact_report.aggregate_study(study, protocol)

    assert cell_report["rows"] == 224
    assert aggregate["all_rows"] == 224
    assert aggregate["state_metrics"]["state_bytes"] == 258
    assert aggregate["trajectory"]
    assert aggregate["interpretation"].get("fixed_beta") == fixed_beta
    assert aggregate["interpretation"].get("normalize_hidden", False) is normalize_hidden
    if normalize_hidden:
        assert aggregate["interpretation"]["hidden_normalization"] == {
            "kind": "layer_norm", "width": 64, "eps": 1e-5,
            "elementwise_affine": False, "placement": "preGELU",
        }

    diagnostics_path = study / "evaluation/0/diagnostics.json"
    original = json.loads(diagnostics_path.read_text())
    mutated = [dict(row) for row in original]
    mutated[0]["state_sse"] += 1.0
    diagnostics_path.write_text(json.dumps(mutated))
    (study / "report.json").unlink()
    completion = study / "evaluation/0/complete.json"
    completion.unlink()
    seal_directory(
        study / "evaluation/0",
        {**cell_identity(study, protocol, 0, "evaluation"),
         "training_seal_sha256": file_hash(study / "training_sealed.json")},
    )
    with pytest.raises(ValueError, match="numeric metric"):
        distilled_fact_report.aggregate_study(study, protocol)

    diagnostics_path.write_text(json.dumps(original))
    mutated = [dict(row) for row in original]
    first_bit = mutated[0]["direct_bits"][0]
    mutated[0]["direct_bits"] = [1 - first_bit if first_bit in (0, 1) else 1,
                                   *mutated[0]["direct_bits"][1:]]
    diagnostics_path.write_text(json.dumps(mutated))
    (study / "evaluation/0/complete.json").unlink()
    seal_directory(
        study / "evaluation/0",
        {**cell_identity(study, protocol, 0, "evaluation"),
         "training_seal_sha256": file_hash(study / "training_sealed.json")},
    )
    with pytest.raises(ValueError, match="direct bits"):
        distilled_fact_report.aggregate_study(study, protocol)
