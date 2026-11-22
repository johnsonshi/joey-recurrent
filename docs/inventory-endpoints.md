# Inventory decision endpoints

The transfer protocol is a secondary development comparison. It does not select methods or calibrate probabilities. The unit of replication is an inventory group, not its two renderings. Measurement for each physical payload starts with a new one-pass parity check on its assigned device.

The ordinary legal-slot metrics and three native answer metrics are complementary:

| Native answer | Main endpoint | Other evidence |
| --- | --- | --- |
| Five-way `choice` | Accuracy of the actual decoded depot key or tie key | First-argmax accuracy, tie support, and agreement after translating both menus to the same depot names |
| `noul` | Accuracy at native `p_yes >= 0.5` | Binary Brier and binary log loss; count six-decimal values at exactly 0.5 and wrong answers assigned zero probability |
| Four-level `score` | MAE of the native expected level, on 0–3 | RMSE and mean squared cumulative-probability error across the three ordinal cutpoints (normalized ranked probability score) |

Full-vector Brier sums over classes. For two classes, this is twice the usual single-event Brier on the same unrounded probabilities. Do not compare it directly to a Brier calculated on the rounded native binary response. The expected level is continuous, not a class choice. A prediction `[0.5, 0.1, 0.1, 0.3]` has level 0 as its first maximum and an expected level of 1.2.

`joey_research.inventory_metrics.decisions` checks each label, probability row and independently captured native answer against the shard's exact ID order and canonical input digest. Supply each `Menu` from the request and generator's public semantics, **not** from predicted keys. `summarize` reports task and cell supports and counts agreement on complete groups. `paired` reports candidate-minus-baseline point estimates and pointwise 95% whole-group bootstrap intervals within task, seed and ledger length. Smaller losses and larger accuracies are better. A genuine zero probability on the truth is not smoothed: NLL is reported as nonfinite, with a support count and no invented interval.
