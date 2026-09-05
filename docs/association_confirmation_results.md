# Association confirmation and diagnostic results

Status: **user-reported aggregate results**, received while the implementation session was recovering from context-overflow errors. The matching final report/checkpoint-evaluation artifacts have not yet been located and independently verified in this workspace. These are not new update-study results. Preserve the frozen protocol and do not tune using confirmation examples.

## Confirmation

| Method/control | Known-answer accuracy | Correct absent answer |
|---|---:|---:|
| Query pooling, first reported seed | 132/1,024 | 38/128 |
| Query pooling, second reported seed | 180/1,024 | 1/128 |
| Query pooling, third reported seed | 175/1,024 | 10/128 |
| Query pooling, three-seed mean | 15.85% | 12.76% |
| Trained contextual mean/FIFO, three-seed mean | 14.42% | Not supplied |
| Latest templates | 74.41% | Not supplied |
| Full history | 97.36% | Not supplied |

The user supplied counts in sequence without individual seed labels. Confirm the mapping to seeds 1337, 2027, and 4099 from artifacts; do not infer it from presentation order.

Query pooling minus mean/FIFO was **+1.43 percentage points**, with a reported **99% interval of −0.29 to +3.13 points**: inconclusive. Query pooling was inferior to recent/latest vocabulary retention by **9.15 points** and latest templates by **58.56 points**. The absent-answer threshold failed; **noninferiority was not claimed**. Rounded means should not be used to reconstruct exact baseline counts or a new confidence interval.

## Training fit and fixed-projection code oracle

| Diagnostic | Known | Absent |
|---|---:|---:|
| Query-pool training fit | 245/2,048 | 80/256 |
| Mean/FIFO training fit | 298/2,048 | 44/256 |
| Privileged-code oracle: initial normal | 5/32 | 1/4 |
| Privileged-code oracle: final | 5/32 | 0/4 |
| Oracle full-context control | 32/32 | 4/4 |

Training fit is poor; failure is not explained solely by generalization to held-out histories. The tested fixed-projection privileged-code oracle also failed despite perfect full-context answers. This does **not** prove that 66 bytes cannot encode the task, nor that every continuous-memory interface is unusable. It is conditional on the tested projection, code constraints, initialization, objective, optimizer, and update budget.

## Consequence for the lean update study

Keep the existing architecture as a controlled measurement subject, not an assumed solution. Do not add writer mechanisms or more training runs solely because the confirmation result is negative. First qualify the shared text reader on the new development states. Then report initial learned-state competence alongside update accuracy and paired answer transitions. If initial competence fails, label the results **competence-limited**; low observed forgetting is not successful memory preservation.

The new tooling can produce meaningful quantitative results about update/preservation trade-offs even when learned memory loses. It cannot establish a useful learned compressor without adequate initial binding competence. An architectural change or additional budget needs a separate development-only diagnostic and declaration, not post-hoc confirmation tuning.
