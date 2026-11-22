# Paired development screening

`joey_research.search` proposes a **development shortlist**, not a workload or a test admission. It reads no files, labels, weights or calibration data. It only consumes scored dev items and explicit measured-cost and paired-interval evidence. The upstream validator must verify artifact contents, split isolation, scorer semantics and measurement provenance before constructing the input records.

## Declare the study before looking at results

Build one `Plan` per comparable model, benchmark and measurement protocol. Persist `plan.sha256` before the first rung. It binds:

* A candidate catalog with exactly two branches, `loop` and `compact`. Every candidate names its stock arm at the intended matched cost. Each compact `k>1` candidate also names a **separate `k=1` arm on the identical compact payload**. Controls are not competing configurations.
* Nested `Rung(ids, keep_per_branch)` dev membership sets and nonincreasing survivor caps. The final cap is one or two **per branch**, excluding controls. Set caps lower on later rungs to spend the larger paired sample on fewer configurations.
* The study digest, paired interval method digest, high-confidence cutoff, measured matching axis (`latency_us` or `layer_millicalls`), and relative stock cost tolerance. A zero tolerance means an exact match. The study digest must bind the model, dev partition, seed, scorer and its version, device/class, batch, token budget and cost normalization. The interval method must specify the confidence level, paired resampling unit and how repeated looks are handled.

`Search(plan).initial_slots()` gives the first two candidate IDs: loop first, compact second. `branch_wave(completed)` proposes at most one remaining candidate per logical branch; it does not borrow an empty lane. `required_controls()` gives the separate required measurements. The coordinator owns all physical leases and must honor these waves when dispatching work.

## Supply a complete paired rung

Use `DevRun.from_registry(...)` for a queue result already accepted as `split="dev", state="succeeded"`. It checks the trial ID and that its configuration digest equals `arm.id`. Bind the validated scored rows, validation proof and measured cost to that result. An `Item` contains only the member digest, correctness, predicted confidence and **summed multiclass Brier**. Numerical inputs use integers: confidence, scores, probabilities and interval endpoints are in millionths of their native unit; latency is in microseconds, average layer calls in thousandths, and peak/payload sizes in bytes. Negative, zero or missing *costs*, and NaN/infinite or unordered numeric intervals, are not substitutes for measurements.

For every candidate/reference edge, supply `Paired` evidence in **candidate minus reference** orientation. It binds both result digests, the exact membership digest, study, method and proof. Include paired intervals for accuracy, Brier, joint high-confidence error, conditional high-confidence error and high-confidence coverage. Outward-round interval ends when converting units. A missing conditional interval is explicitly review-only. The selector does not manufacture uncertainty. Reject a validation artifact assembled from unpaired runs or independent confidence intervals.

`search.advance(runs, paired)` rejects test/calibration inputs, duplicate or missing members, mismatched cohorts, missing references, substituted pairs and bounds inconsistent with the supplied item estimates. Nested rungs may add members but cannot replace earlier scores. A rejected call does not advance the search. To resume, replay the plan and validated rung artifacts; catalog order, row order and pair order do not affect nominations.

## Decision rules

| Check | Rule |
| --- | --- |
| Stock at matched cost | Compare actual measurements on the declared axis, relative to stock. Reject a cost mismatch, regardless of the nominal schedule. |
| Limited compression loss | The compact payload must be measurably smaller than stock. For the proposed compact `k>1` system versus matched stock, the paired lower accuracy bound must be **above -0.02** and the upper summed-Brier bound **below +0.01**. Touching or crossing a margin is not proof of noninferiority. This comparison does not establish noninferiority of `k=1` alone. |
| Recovery | Compare compact `k>1` with its own `k=1` control. The module marks a quality recovery signal only if the paired accuracy and Brier bounds exclude deterioration on both and establish strict improvement on at least one. This signal does not override a safety flag. |
| Confident errors | Report `high_wrong / all`, `high_wrong / high`, and `high / all` separately for candidate and controls. `high=0` makes conditional risk undefined, not zero. A paired interval supporting an increase in either error rate eliminates the candidate. An observed increase or an interval that cannot exclude an increase is flagged for review. Neither permits an automatic finalist, even when accuracy improves. |
| Pareto screen | Within each branch, prefer clear candidates over review-only ones. An arm is dominated if another in the same tier is no worse on accuracy, Brier, joint and conditional high-confidence error, latency, calls, peak bytes and payload bytes, and better on at least one. This frontier describes measured tradeoffs, not statistical superiority. |

Within a frontier the deterministic order is higher stock-relative accuracy lower bound, lower Brier upper bound, lower joint-error upper bound, lower measured latency, then arm digest. Review-only configurations may occupy spare capacity for another dev rung; they can never become final nominees. An empty shortlist is a valid outcome.

`Decision.report()` is JSON-ready and includes exact risk fractions, coverage, cost and paired evidence, all assessment flags, selected dev IDs and their required control dev IDs. Preserve `decision_sha` and the report with its upstream proofs. `nominee_dev_ids` is empty before the last rung. **Do not translate a shortlist into extra held-out runs.** The freeze coordinator applies its preregistered held-out roles and existing once-only admission policy separately.

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -p test_search.py -v
```
