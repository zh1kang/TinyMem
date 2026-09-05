# Data

Dataset contents are downloaded into `data/raw/` and are not committed.

`manifest.json` defines canonical sources and selected subsets. After installation, `installed.lock.json` records the exact byte count and SHA-256 digest of every downloaded file.

## Active study and preserved sources

The active study uses controlled bAbI-derived fact updates. Split by underlying source/history group before producing paired additions, repetitions, corrections, or queries. Validate answers independently, and keep confirmation outside development. Fresh entity IDs do not make a reused source history independent.

The implemented [paired-update data design](../docs/memory_update_study.md) has 256 training, 32 development, and 64 new confirmation worlds. Run `python -m scripts.prepare_memory_updates --output <fresh-directory>` after staging the existing source/exclusion artifacts. This builds data only; it does not train a model or inspect the old confirmation dataset.

The existing opaque-study manifests and consumed-data exclusions remain unchanged. No reserved bAbI test or external answers may be used to develop the update study. Generated datasets and predictions stay outside Git.

The table below records the historical installed data layers, not additional active research requirements.

## Original installation

| Layer | Dataset | Initial selection | Role |
|---|---|---|---|
| Controlled | bAbI | Official tasks 1–20 archive | Training, smoke tests, symbolic validation |
| Controlled | BABILong | `qa1`–`qa5`, lengths 1k–8k | Long-delay controlled evaluation |
| Ordinary LM | WikiText-2 raw | Full pinned train/validation/test Parquet files | Language-model loss and perplexity |
| External | LongMemEval cleaned | Oracle and cleaned-S | Harness validation and final held-out evaluation |

The full BABILong corpus and LongMemEval-M were not part of this installation.

## Licenses

- bAbI: BSD; consult the upstream bAbI license.
- BABILong: Apache-2.0 project code; its data combines BSD-licensed bAbI with Apache-2.0 PG-19.
- WikiText: CC BY-SA 3.0 according to the current Salesforce dataset card.
- LongMemEval: MIT.

Dataset licenses apply to the downloaded data independently of TinyMem's source code.
