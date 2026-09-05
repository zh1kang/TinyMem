# Literature relevant to the memory-update study

Primary-source review started 2026-09-04; scope narrowed 2026-09-05. This is a targeted reading record, not an implementation backlog or independent reproduction.

## The distinction that matters

Shorter model inputs, less KV storage per token, fixed total persistent state, and writing before a future query is known are different contracts. TinyMem studies the last two. Published token/vector compression ratios do not establish superiority over packed raw IDs with recomputation.

| Source | Relevant observation | Limit for TinyMem |
|---|---|---|
| [ICAE](https://arxiv.org/html/2307.06945v4) | Trains an encoder for a frozen decoder using reconstruction and continuation before instruction training | Full-width memory slots and a different training regime, not a constant 66-byte recurrent state |
| [No Mean Feat](https://arxiv.org/html/2510.20797v2) | Mean pooling is strong after a trained bidirectional encoder; freezing/removing the encoder substantially degrades its ablation | TinyMem pools frozen causal features; the two mean-pooling systems are not equivalent |
| [Chimera](https://arxiv.org/html/2606.21562v1) | A query-independent teacher bottleneck directly supervises recurrent memory at matching prefixes | Visual pose estimation, 20 slots of width 3,072, and much larger training resources |
| [LatentPress v2](https://arxiv.org/html/2609.01507v2) | Reader-matched contextual and literal features can train a frozen-reader soft-token interface | Full-width growing state; oracle evidence and retained user turns in its conversational evaluation |
| [Cognitive Chunking](https://arxiv.org/html/2602.13980v1) | Local receptive-field constraints can reduce compressor routing difficulty | Different trained encoder, memory layout, and computation budget |
| [Attention Matching](https://arxiv.org/html/2602.16284v1) | Matching attention outputs and mass provides a direct compaction target | Per-layer KV state and per-context fitting are not the present storage/interface contract |

## Implications for the current design

A capable full-text reader does not prove that it can interpret the chosen compressed vectors. TinyMem's two width-eight states each expand through the same eight-dimensional linear subspace. Two unrestricted width-2,048 BF16 vectors would instead require 8,192 value bytes before metadata.

The current writer averages contextual features before its trainable nonlinear compression head. That ordering may weaken entity–value bindings, but contextual features can already contain interactions. It is a hypothesis to test with paired binding changes, not proof of a bag-of-words defect.

Full recurrent gradients are already implemented. Nonzero gradients establish connectivity, not useful preservation. Immediate retrieval, incorporation of a correction, and retention of unrelated facts therefore need separate measurements.

Direct state supervision is a possible diagnostic only if a useful query-independent teacher bottleneck first exists. Targets must use exactly the student's observed prefix, without future events. It is not part of the initial implementation.

## What to do now

Keep the existing architecture. Finish training-fit and fixed-readout tooling, then implement the controlled before/after event measurements. Use literature to interpret a demonstrated failure rather than adding components in anticipation of one.

The contribution sought is an evidence-backed accuracy–storage trade-off for reliable updates, with strong explicit-memory references and honest limits. A negative result can be informative without becoming a claim that bounded learned memory is impossible.
