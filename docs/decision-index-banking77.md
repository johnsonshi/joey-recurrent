# BANKING77: native Choice versus research metrics

`benchmarks.decision_index_banking77` is a standard-library adapter for **one** public task: Decision Index edition 0.3, catalog 4, BANKING77. It does not compute the public index, the Full score, or a Jev model score. It contains no questions or labels.

**Source and scope.** The adapter follows `apolinario/decision-index` revision `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec` (MIT). The kit is independent of TypeSafe AI; its edition manifest names `jev-1.13.0` as the reference model. The source data are PolyAI's BANKING77, described in Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), and licensed under CC BY 4.0. This adapter distributes neither the vocabulary nor the test corpus. Preserve attribution, license notice and disclosure of modifications if separately redistributing licensed content. The kit's license does not cover its upstream datasets.

**Blind projection.** Obtain a de-labeled local export from an independent custodian; do not open the combined test CSV in the inference process. Preserve the original zero-based CSV index, UTF-8 utterance and order of `categories.json`. The native request puts the utterance in `q1.instructions`, leaves `state={}`, and gives `q1` all 77 criteria keyed `option_0` through `option_76`. Do not rearrange columns or silently keep the closest four. An engine unable to return all options is unsupported.

```python
from benchmarks import decision_index_banking77 as b

categories = b.load_categories(local_categories_json)   # validates raw bytes against the kit pin
case = b.project_blind(blind_record, categories)         # exactly {source_index, text}
native_request = case.native_payload()                  # only state and questions
joey_input = case.research_input()                      # five-field input for Joey evaluation
```

The native case/group ID is `BANKING77:test:<source index>`; the run ID prefixes this with `4:BANKING77:`. Each source row is one request with one decision, not 77 separate cases. Native payload hashes include the reference model and preserve JSON key insertion order. The Joey input hash instead uses its own canonical schema, and its research group ID unifies duplicate normalized utterances for split/bootstrap purposes. Neither the ID nor the group is model input.

**Outcome custody.** Freeze finalists, protocol, blinded states and method identities outside this module. `load_released_test(test_csv, categories_json, finalists=..., finalists_sha256=..., manifest=..., manifest_sha256=..., arms=..., baseline=..., passes=...)` is for the label custodian only. Before opening the combined file it requires a pinned finalists receipt and an independently pinned input manifest assigning **all 3,080 original IDs** to `final`. It then checks the kit's two source byte hashes and reconciles every released input hash. It returns `(cases, label_rows)` separately. For an authorized subset, use only separately released label rows; a subset receipt cannot authorize opening the entire labeled test CSV. A digest is not a substitute for access controls or a trusted timestamp.

**Native single-task metric.** `score_native(cases, label_rows, native_results)` computes **macro-F1 over the union of gold and predicted semantic classes in completed, supported cases**. This is not multiclass accuracy and does not average 77 zero-filled per-class scores on a small subset. `native_results` carries `run_id`, `payload_sha256`, `status` and, for `ok`, `response.answers.q1` with `type`, `choice` and a complete map of probabilities. The native scorer honors the *declared* `choice`; it does not replace it with the probability argmax. Missing requests are pending. Unsupported, abstained, error and pending requests remain in the coverage denominator, not the complete-case F1 numerator. Report selected count, total source count, accepted count, each failure count, F1 support, and coverage together. No supported cases gives `null`, not zero. This adapter rejects stale/duplicate result records: resolve resumed-run duplicates to one final status upstream before scoring.

**Categorical research subset.** For NLL/Brier/ECE, vectorize the `probabilities` map in `option_0,...,option_76` order; never by lexicographic key sort. Joey's probability-mass tolerance is `1e-6`, versus `0.01` in the kit. Its deterministic argmax can differ from the native declared choice (reported as `declared_choice_not_first_argmax`). Label these metrics as Joey research metrics, keep the paired state set fixed, and report invalid distributions. A smaller menu, alternate wording, sampled rows, re-split test, or new method readout is not a published-index result.

The kit's public index also requires edition-specific eligibility/exclusions, coverage-adjusted and chance-corrected task values, and aggregation over the complete area panel. Its Full score additionally uses private evaluations. Neither is inferred here. No real-corpus score or validation of live service behavior is asserted.
