import hashlib
import json

import pytest
import torch

from tinymem.research import pretrained


def test_snapshot_verifies_git_and_lfs_hashes(tmp_path, monkeypatch):
    files = ("config.json", "weights.safetensors")
    monkeypatch.setattr(pretrained, "QWEN_FILES", files)
    payload = b"test snapshot"
    siblings = []
    for filename in files:
        (tmp_path / filename).write_bytes(payload)
        item = {"rfilename": filename, "size": len(payload), "blobId": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
        if filename.endswith("safetensors"):
            item["lfs"] = {"sha256": hashlib.sha256(payload).hexdigest()}
        siblings.append(item)
    metadata = {"id": pretrained.QWEN_MODEL_ID, "sha": pretrained.QWEN_REVISION, "siblings": siblings}
    (tmp_path / "upstream.json").write_text(json.dumps(metadata))
    result = pretrained.verify_qwen_snapshot(tmp_path)
    assert result["files"][files[0]]["sha256"] == hashlib.sha256(payload).hexdigest()
    for filename in files:
        (tmp_path / filename).write_bytes(b"x" * len(payload))
        with pytest.raises(ValueError, match="hash mismatch"):
            pretrained.verify_qwen_snapshot(tmp_path)
        (tmp_path / filename).write_bytes(payload)
    metadata["sha"] = "main"
    (tmp_path / "upstream.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="pinned"):
        pretrained.verify_qwen_snapshot(tmp_path)


@pytest.fixture
def tiny_qwen_reader():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4, "answer": 5}, unk_token="[UNK]",
    ))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", padding_side="left")
    torch.manual_seed(4)
    config = transformers.Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=32, eos_token_id=2, pad_token_id=0)
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    return pretrained.PretrainedReader(model, tokenizer)


def test_left_padded_native_generation_matches_independent_calls(tiny_qwen_reader):
    reader = tiny_qwen_reader
    prompts = ["hello world answer", "hello"]
    batch = reader.generate(prompts, max_new_tokens=3)
    individual = [reader.generate([prompt], max_new_tokens=3)[0] for prompt in prompts]
    assert [row["prediction"] for row in batch] == [row["prediction"] for row in individual]
    assert [row["prompt_tokens"] for row in batch] == [3, 1]
    inputs = reader.tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        explicit = reader.model.generate(**inputs, do_sample=False, max_new_tokens=3, eos_token_id=2, pad_token_id=0)
    assert [row["generated_ids"] for row in batch] == explicit[:, 3:].tolist()
    assert not hasattr(reader, "past_key_values")


def test_reader_rejects_overlength_instead_of_truncating(tiny_qwen_reader):
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tiny_qwen_reader.generate(["hello " * 31], max_new_tokens=2)
    with pytest.raises(ValueError, match="positive"):
        tiny_qwen_reader.generate(["hello"], max_new_tokens=0)
