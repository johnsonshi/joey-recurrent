# Native transfer check: warehouse inventory

This synthetic check asks whether the same model can reconcile stock using Joey's native verification, ordered score, and choice interfaces. It is not an official benchmark, a model-selection stage, or a reason to reselect finalists. The independent unit is **one ledger group**, not one rendering. Keep both renderings together for splitting and uncertainty estimates.

## Coordinator interface

`benchmarks/inventory_transfer.py` is a standalone Python 3.10+ module; it uses only the standard library and performs no I/O. To import it without installing or merging anything, place this surface's `benchmarks/` directory on the import path and `import inventory_transfer`.

| Call | Return value |
| --- | --- |
| `iter_inputs(seed: int, groups_per_cell: int)` | Iterator of fresh dictionaries with **only** `id`, `group_id`, `task`, `prompt` (strings), and `choices` (ordered list of strings). Two consecutive rows share one group and task. |
| `group_manifest(seed: int, groups_per_cell: int)` | `dict[group_id, metadata]`. Each metadata dict has `seed` (int), `task` (str), `difficulty` (int: **2, 6, or 12 total receipts**), and `renderings` (`{first_input_id: 0, second_input_id: 1}`). No answers or stocks; this call does not generate ledgers. |
| `adjudicate(row: dict)` | `{"id": row["id"], "input_sha256": hex_digest, "gold_index": i}`. `i` is zero-based in **this row's** menu. Reads just the row, never the manifest, seed, or companion view. Raises `ValueError` for invalid data. |

Use the same integer seed and positive integer `groups_per_cell` for both generation calls (booleans are not integers here). Size is part of the run identity: a smaller run is not a prefix of a larger one. IDs are opaque 128-bit draws on a separate deterministic stream, not names for classes, positions, or seeds. `input_sha256` is SHA-256 over the UTF-8 bytes of `json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`. It binds IDs, exact text, task, and menu order, not just normalized evidence.

For inference, pass **`row["prompt"]` unchanged as the state**, `row["choices"]` unchanged as the menu, and the corresponding instruction separately. The module exports `NATIVE_TYPES` and `NATIVE_INSTRUCTIONS` keyed by task. Do not prepend the instruction to the state or pass the manifest or adjudication output to the model.

| `task` | Native type and exact instruction | Legal menu and decision |
| --- | --- | --- |
| `stock-eligibility` | `noul`: After posted entries only, does the focus depot have at least five units? | `["Yes","No"]`, in that order. Five qualifies. |
| `stock-level` | `score`: After posted entries only, rate the final stock of the focus depot using the ordered levels. | `["Empty (0 units)","Low (1 to 3 units)","Medium (4 to 6 units)","High (7 or more units)"]`, always low to high. |
| `stock-choice` | `choice`: After posted entries only, which depot has the most units? If there is no unique maximum, choose No unique depot. | The four declared names and `No unique depot`, in the row's order. A tie **at the maximum** selects the latter; a tie lower down does not. Focus is not a tiebreaker. |

## Reading a ledger

Here is a hand-written example, not a generated study row:

```text
Warehouse inventory ledger (units; records unordered)
Focus depot: Iris
Opening Moss: 2
Opening Iris: 4
Opening Ash: 6
Opening Pine: 1
R-0013ABF2 | VOID | remove 8 from Iris
R-001274FC | POSTED | transfer 1 from Ash to Iris
```

Iris receives one unit; the void removal has no effect. Iris and Ash both close at five: the decisions are `Yes`, `Medium (4 to 6 units)`, and `No unique depot`.

After the header, the reader accepts facts in any order and ignores blank lines and redundant whitespace. There must be one focus and four distinct openings. Names are single ASCII words matching `[A-Z][a-z]*`; the reader does not assume the generator's names. Receipt IDs match `R-[0-9A-F]{8}` and are unique within a ledger. Each receipt specifies `POSTED` or `VOID` and exactly one action: `add Q to D`, `remove Q from D`, or `transfer Q from S to D` with distinct endpoints. Openings are unsigned decimal integers; quantities are positive. Neither allows leading zeroes except the opening `0`.

Every receipt, including a void, must refer to declared depots. A void is an unposted record, not a reversal of a previous ID. Both statuses must occur. Only posted movements affect closing stock, and none of the four closing stocks may be negative. Record order is not a chronology: a displayed prefix need not be funded. Extra fields, facts, trailing prose, repeated IDs, and illegal or reordered native binary/ordinal menus are errors.

## Sampling and validation

Each seed has nine cells: three tasks × three receipt counts. The allocator shuffles semantic outcomes with counts differing by at most one **group** per cell. Choice balances the four named winners and a real maximum-tie class, not the five menu positions. At 96 groups per cell this gives **864 independent groups and 1,728 decisions per seed**.

The generator samples a closing stock in the allocated class, then works backwards through feasible postings to reconstruct the openings. Focal quantities are changes between varying balances, not printed closing targets. There are respectively 1, 2–3, and 4–7 posted movements involving the focus. All groups include void distractors; the longer ledgers also include posted movements elsewhere. For choice, a posting always touches an opening or closing leader; sometimes the opening winner survives, so neither copying nor always discarding it is a rule. The two views preserve the same names, openings, ticket IDs, amounts, statuses and focus. Separate streams shuffle openings, receipts and (choice only) options, redrawing the second order if identical. They do not condition on the position of the answer.

Run the bounded CPU-only validation **from this isolated surface** with a Python 3.10+ interpreter:

```sh
python3 -B -m unittest discover -s tests -p 'test_inventory_transfer.py' -v
```

The tests stream both seeds `81403` and `94217` through a separate forward accountant, checking conservation, support, balance, paired views, disjoint IDs and normalized prompts. They also check hand-edited rows, menu-sensitive hashes, threshold edges, malformed evidence, and opening/position shortcuts. No campaign inputs, label files, study receipts, or model scores are produced. Staging and inference remain with the coordinator.
