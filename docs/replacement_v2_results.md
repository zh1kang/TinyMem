# replacement v2 development results

updated: 2026-09-04.
this report supersedes the legacy split for new replacement development claims, not for historical replay.

## protocol

both conditions use the same parent WikiText checkpoint, 1,000 training examples, 100 validation examples, data seed 0, validation seed 10,000, and optimizer seeds 1337, 2027, and 4099.
the exact manifest SHA-256 is `b202ed38eb2716a31e0e2f68e6b817480da8dcd186f321287974f4189e2e5a52` in all six runs.
v2 keeps complete histories, alternate questions, and reordered independent facts within one split.
validation contains 50 histories and two queries per history.

each run uses 2,000 steps, batch 64, learning rate 0.001, weight decay 0.01, 64-byte segments, two full-width float32 slots, and the existing answer-byte weighting.
the supervised reference uses 250 controller-only warmup steps and slot loss weight 1.
the answer-loss-only condition uses zero warmup steps and slot loss weight 0.
this compares those complete training recipes, not an isolated infinitesimal change in one coefficient.

## results

all values are generated exact accuracy percentages.
the ± term is sample standard deviation across three optimizer seeds, not a confidence interval over fresh histories.

| training condition | seed 1337 | seed 2027 | seed 4099 | mean ± sd |
| --- | --- | --- | --- | --- |
| slot supervision | 100 | 100 | 98 | 99.3 ± 1.2 |
| answer loss only | 62 | 67 | 67 | 65.3 ± 2.9 |

| condition | normal | dropped memory | different-history memory | corrected query | unchanged query |
| --- | --- | --- | --- | --- | --- |
| slot supervision | 99.3 | 19.0 | 18.3 | 99.3 | 99.3 |
| answer loss only | 65.3 | 18.0 | 18.7 | 94.0 | 36.7 |

the supervised result survives removal of train/validation overlap.
the answer-loss-only model also uses memory, but does not reliably preserve the unchanged fact when learning the replacement rule.
this is consistent with prioritizing the new correction while losing another live binding.
it does not show that answer supervision can never learn replacement; it describes this architecture, data distribution, objective, and fixed budget.

the grouped-query shuffle bug was found during the supervised sweep.
all three supervised checkpoints were re-evaluated with a different-history permutation, without retraining.
every non-shuffle result was checked for exact equality with the original record.
original artifacts remain unchanged.

## implications and next steps

keep the supervised controller as a diagnostic, not the main learned-compression result.
test the new query-blind recurrent writer, which updates its whole narrow state from fixed-byte history chunks and receives no correct-slot labels.
then screen a capable pretrained reader before the complete matched-byte baseline comparison.
do not infer storage superiority from these full-width two-slot results: their retained state still costs more than the short raw history.
do not use these development questions as a fresh final holdout later.

## artifacts

- supervised runs: `artifacts/predictions/replacement_v2_supervised/`.
- answer-loss-only runs: `artifacts/predictions/replacement_v2_answer_only/`.
- repaired controls: `reevaluation_history_shuffle_v2.json` beside each supervised result.
- source-linked summary and figure: `artifacts/figures/replacement_v2_development/`.
- split repair: `0b0acd9`; shuffle repair: `817bd1e`.

![development comparison](../artifacts/figures/replacement_v2_development/comparison.png)
