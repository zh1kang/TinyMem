"""Pinned, locally verified Qwen reader and native-token generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


QWEN_MODEL_ID = "Qwen/Qwen3-1.7B"
QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QWEN_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "model.safetensors.index.json", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors", "LICENSE",
)


def verify_qwen_snapshot(directory: Path) -> dict[str, object]:
    """Check downloaded files against the pinned Hub Git/LFS object hashes."""
    upstream = json.loads((directory / "upstream.json").read_text(encoding="utf-8"))
    if upstream.get("id") != QWEN_MODEL_ID or upstream.get("sha") != QWEN_REVISION:
        raise ValueError("upstream metadata does not match the pinned Qwen snapshot")
    siblings = {item["rfilename"]: item for item in upstream["siblings"]}
    files = {}
    for filename in QWEN_FILES:
        path = directory / filename
        metadata = siblings[filename]
        if path.stat().st_size != metadata["size"]:
            raise ValueError(f"snapshot file size mismatch: {filename}")
        with path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if "lfs" in metadata:
            valid = sha256 == metadata["lfs"]["sha256"]
        else:
            payload = path.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            valid = git_hash == metadata["blobId"]
        if not valid:
            raise ValueError(f"snapshot content hash mismatch: {filename}")
        files[filename] = {"bytes": path.stat().st_size, "sha256": sha256}
    return {"model_id": QWEN_MODEL_ID, "revision": QWEN_REVISION, "files": files}


@dataclass
class PretrainedReader:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @torch.inference_mode()
    def generate(self, prompts: list[str], *, max_new_tokens: int = 16) -> list[dict[str, object]]:
        """Generate independently per row; no cache survives this call."""
        from transformers import GenerationConfig

        if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("prompts must contain nonempty strings")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        inputs = self.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
        if inputs.input_ids.shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt and generation exceed reader context; truncation is forbidden")
        inputs = inputs.to(self.model.device)
        generation = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            temperature=1.0, top_p=1.0, top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
        )
        sequences = self.model.generate(**inputs, generation_config=generation)
        suffixes = sequences[:, inputs.input_ids.shape[1]:].tolist()
        return [
            {"prediction": self.tokenizer.decode(ids, skip_special_tokens=True),
             "generated_ids": ids, "prompt_tokens": int(valid.sum())}
            for ids, valid in zip(suffixes, inputs.attention_mask, strict=True)
        ]


def load_qwen_reader(directory: Path, *, device: torch.device, dtype: torch.dtype) -> PretrainedReader:
    """Load safetensors without remote code, downloads, or implicit CPU fallback."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    if model.config.model_type != "qwen3":
        raise ValueError("expected the pinned Qwen3 architecture")
    model.eval().requires_grad_(False)
    return PretrainedReader(model, tokenizer)
