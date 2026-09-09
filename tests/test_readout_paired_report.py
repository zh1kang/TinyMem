"""Paired reports reject incomplete arm sets before aggregating results."""

from copy import deepcopy

import pytest

from tinymem.research.readout_paired_report import validate_arm_pairs, validate_paired_protocols


def paired_protocols():
    common = dict.fromkeys((
        "kind", "steps", "splits", "checkpoint_selection", "optimizer",
        "max_new_tokens", "persistent_bytes", "input_identity",
        "input_identity_verification", "reader_parameters_sha256", "reader_config",
        "reader_width", "device", "torch_version", "source_sha256", "shared_parameters",
        "cuda_version", "device_name", "deterministic_algorithms", "reader_dtype",
    ), "matched")
    return [dict(common, arm=arm, seed=seed, schedule=[seed])
            for seed in (23, 17) for arm in ("affine", "gelu")]


def test_paired_protocols_allow_different_schedules_between_seeds():
    assert validate_paired_protocols(paired_protocols()) == [17, 23]


@pytest.mark.parametrize("field", ["reader_parameters_sha256", "input_identity",
                                  "source_sha256", "optimizer", "device", "splits",
                                  "cuda_version", "device_name", "reader_dtype"])
def test_paired_protocols_reject_changed_contract(field):
    protocols = deepcopy(paired_protocols())
    protocols[-1][field] = "different"
    with pytest.raises(ValueError, match="unmatched protocol field"):
        validate_paired_protocols(protocols)


def test_paired_protocols_reject_missing_contract_field():
    protocols = paired_protocols()
    del protocols[0]["optimizer"]
    with pytest.raises(ValueError, match="missing protocol field"):
        validate_paired_protocols(protocols)


@pytest.mark.parametrize("missing", [False, True])
def test_paired_protocols_reject_unpaired_schedule(missing):
    protocols = paired_protocols()
    if missing:
        del protocols[-1]["schedule"]
    else:
        protocols[-1]["schedule"] = [999]
    with pytest.raises(ValueError, match="training schedule"):
        validate_paired_protocols(protocols)



def test_requires_both_arms_for_every_seed():
    with pytest.raises(ValueError, match="both arms"):
        validate_arm_pairs([{"arm": "affine", "seed": 17}])


def test_rejects_duplicate_arm_seed():
    with pytest.raises(ValueError, match="duplicate"):
        validate_arm_pairs([
            {"arm": "affine", "seed": 17},
            {"arm": "affine", "seed": 17},
        ])


def test_returns_sorted_paired_seeds():
    assert validate_arm_pairs([
        {"arm": arm, "seed": seed}
        for seed in (23, 17) for arm in ("gelu", "affine")
    ]) == [17, 23]
