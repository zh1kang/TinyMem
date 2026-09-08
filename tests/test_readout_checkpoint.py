import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from test_readout_interface import reader
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research.readout_checkpoint import save_checkpoint, load_checkpoint
from tinymem.research.readout_controls import controlled_state
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.readout_read import read_state_answer


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_checkpoint_round_trip_and_hash_guard(tmp_path, kind):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind)
    path = tmp_path / "weights.safetensors"
    digest = save_checkpoint(path, encoder, bridge)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_checkpoint(path, expected_sha256=digest, reader_width=16, kind=kind)
    for actual, expected in zip(loaded, (encoder, bridge), strict=True):
        assert not actual.training
        for key, value in actual.state_dict().items():
            assert torch.equal(value, expected.state_dict()[key])
    with pytest.raises(FileExistsError):
        save_checkpoint(path, encoder, bridge)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, expected_sha256=digest, reader_width=32, kind=kind)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, expected_sha256=digest, reader_width=16, kind="gelu" if kind == "affine" else "affine")
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(path, expected_sha256=digest, reader_width=16, kind=kind)


def test_checkpoint_rejects_nonfinite_parameters(tmp_path):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, "affine")
    with torch.no_grad():
        bridge.input_projection.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_checkpoint(tmp_path / "bad.safetensors", encoder, bridge)
    assert not (tmp_path / "bad.safetensors").exists()


@pytest.mark.parametrize("defect", ["missing", "extra", "shape", "dtype", "nonfinite", "metadata"])
def test_correctly_hashed_malformed_checkpoint_is_rejected(tmp_path, defect):
    from safetensors.torch import load, save

    path = tmp_path / "weights.safetensors"
    save_checkpoint(path, OneShotEncoder(16), ReadoutBridge(16, "affine"))
    weights = load(path.read_bytes())
    metadata = {"format": "tinymem-readout-v1", "reader_width": "16", "kind": "affine"}
    key = "encoder.queries"
    if defect == "missing":
        del weights[key]
    elif defect == "extra":
        weights["unexpected"] = torch.zeros(1)
    elif defect == "shape":
        weights[key] = torch.zeros(1, 16)
    elif defect == "dtype":
        weights[key] = weights[key].double()
    elif defect == "nonfinite":
        weights[key][0, 0] = float("inf")
    else:
        metadata["kind"] = "gelu"
    payload = save(weights, metadata=metadata)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(path, expected_sha256=hashlib.sha256(payload).hexdigest(), reader_width=16, kind="affine")


def test_checkpoint_operations_preserve_cpu_random_state(tmp_path):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, "gelu")
    rng = torch.random.get_rng_state().clone()
    path = tmp_path / "weights.safetensors"
    digest = save_checkpoint(path, encoder, bridge)
    assert torch.equal(torch.random.get_rng_state(), rng)
    load_checkpoint(path, expected_sha256=digest, reader_width=16, kind="gelu")
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_separate_process_reads_without_history_or_encoder(reader, tmp_path, kind):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind).eval()
    state = controlled_state(encode_readout_history(reader, encoder, torch.tensor([3, 4, 3, 4])), "normal")
    before, questions = torch.tensor([3]), [torch.tensor([4, 5]), torch.tensor([3, 5])]
    expected = [read_state_answer(reader, bridge, state, before, q, max_new_tokens=3) for q in questions]
    reader.model.save_pretrained(tmp_path / "reader")
    reader.tokenizer.save_pretrained(tmp_path / "reader")
    save_file({"values": state.values, "valid": state.valid}, tmp_path / "state.safetensors")
    save_file(bridge.state_dict(), tmp_path / "bridge.safetensors")
    # The child receives no serialized encoder, history tokens, or gold answers.
    script = '''
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.memory.readout_interface import ReadoutBridge, check_readout_state
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_read import read_state_answer
root = Path(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(root / "reader", local_files_only=True).eval().requires_grad_(False)
reader = PretrainedReader(model, AutoTokenizer.from_pretrained(root / "reader", local_files_only=True))
payload = load_file(root / "state.safetensors")
state = LatentSlotState(payload["values"], payload["valid"])
check_readout_state(state)
bridge = ReadoutBridge(16, sys.argv[2]).eval()
bridge.load_state_dict(load_file(root / "bridge.safetensors"), strict=True)
results = [read_state_answer(reader, bridge, state, torch.tensor([3]), torch.tensor(q), max_new_tokens=3)
           for q in ([4, 5], [3, 5])]
(root / "result.json").write_text(json.dumps({"values": state.values.tolist(), "bytes": state.nbytes, "results": results}))
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path), kind], check=True, timeout=60,
                   capture_output=True, text=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["bytes"] == 66
    assert result["values"] == state.values.tolist()
    assert result["results"] == expected
