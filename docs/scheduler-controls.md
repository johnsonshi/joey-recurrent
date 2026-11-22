# Held-out control nominations

`Queue.nominate(picks, *, decision_sha, witnesses=None, payloads=None) -> None`
still replaces the whole proposal **before freeze**. Stock/selected-only calls need
neither new argument; their limits, proposal bytes and existing seals are unchanged.
There is no new database version, admission route or final-phase release.

With either `Role.ZERO_EXTRA` or `Role.ORDER_ENSEMBLE`, supply both:

- `witnesses: Mapping[Pick, MethodWitness]`, keyed by **every exact** `Pick(dev_id, test_id, role)`.
  `MethodWitness(settings_json: bytes, parent: Pick | None = None)` takes worker settings
  only, not a packet or a claimed hash. Encode with
  `json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")`
  (no newline; at most 8192 bytes). `worker.method_sha256(settings)` must equal **both**
  registered `TrialKey.config` digests, with no default expansion. Opaque legacy hashes
  are not method witnesses.
- `payloads: Mapping[str, str | None]`, keyed by **exactly** the registered model
  digests in the proposal. Supply `None` for full or the independently verified
  `compact_manifest_sha256` for compact. Obtain these pins from staging, **not**
  from the settings being checked. A hash match cannot verify staging provenance.

For example, given registered pairs and their canonical method bytes:

```python
from joey_research.scheduler import MethodWitness, Pick, Role

stock = Pick(stock_dev, stock_test, Role.STOCK)
winner = Pick(selected_dev, selected_test, Role.SELECTED)
split = Pick(split_dev, split_test, Role.ZERO_EXTRA)
order = Pick(order_dev, order_test, Role.ORDER_ENSEMBLE)
q.nominate([stock, winner, split, order], decision_sha=decision_sha,
           witnesses={stock: MethodWitness(stock_bytes), winner: MethodWitness(selected_bytes),
                      split: MethodWitness(split_bytes, winner), order: MethodWitness(order_bytes, winner)},
           payloads={full_model_digest: None})
```

| Witnessed role | Worker method |
| --- | --- |
| `STOCK` | Forward, ordinary one-pass stock (`k=1`, naive), full or same-payload compact. |
| `SELECTED` | Fixed recurrence (`k>1`, not `zero_extra`), not a diagnostic or live policy. |
| `ZERO_EXTRA` | `method="zero_extra"`, `k>1`, **one** split traversal; no ensemble. |
| `ORDER_ENSEMBLE` | Forward **full** `k=2` zero-extra; the worker repeats on reversed option order for a frozen fraction of whole groups. |

Each additional control names the actual selected pick in this proposal and matches
both its dataset views, model, code, seed, scorer, precision, window, option order
and compact receipt/recipe. All witnessed picks use final readout, no policy, and
at most one pick per `(model, role)` across views. For the ensemble,
`numerator / denominator = (selected_k − 1) × (hi − lo) / 24`, checked exactly.
For instance, `[8,12)` at `k=3` needs `1/3` (or `2/6`); it matches planned layer
visits, not latency. The preregistered group-ranking seed is `91473`, separate
from the trial seed.

The freeze payload retains `payloads` and each pick's `settings` and `method_sha256`;
controls also retain `selected_parent` with both IDs, role and method hash.
Every pick still needs successful dev evidence and an unchanged key except for
dataset. Neither a witness nor a seal substitutes for either gate, available
budget, or the coordinator's independent final release.
