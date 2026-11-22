# Decision evidence SVGs

`scripts/plot_decision_evidence.py` needs only Python 3.9+ and its standard library.
`render_evidence(data) -> dict[str, bytes]` accepts a decoded JSON object, does no I/O,
and raises `ValueError` on an invalid group. Each value is a standalone UTF-8 SVG.
No model, GPU, Git, network, font download, or per-example file is needed.

```sh
python3 -B scripts/plot_decision_evidence.py released-summary.json new-figure-directory
python3 -B -m unittest discover -s tests -v
```

The CLI validates before creating the output directory, which must not exist.
`load_summary(stream)` rejects duplicate JSON keys and nonfinite literals. For a
complete **invented** in-memory example covering every view, see `fixture()` in
`tests/test_plot_decision_evidence.py`. A minimal valid input is:

```json
{"version":1,"split":"Toy holdout","projection":"Binary choice","n":8,"depth_unit":"layers",
 "series":[{"model":"Cedar","family":"Grove","passes":[{"pass":1,"depth":4,"correct":5}]}]}
```

## v1 contract

One input = **one released stage, fixed matched split, and decision projection**.
All series and passes refer to the *same N decisions*, not merely equal sample sizes.
The producer must verify matching and measurement provenance before stripping identifiers.
Run other splits, projections, stages, timing protocols, or storage conventions separately;
never pool them. Only supply measurements intended for display. An omitted field means
unreleased, not zero. No extra keys, `null` placeholders, numeric strings or booleans.
Counts are JSON integers, nonnegative and at most `2^53−1`; all other numbers must be finite.

| Object | Required fields | Optional fields |
|---|---|---|
| Group | `version`: integer `1`; `split`, `projection`: labels; `n`: positive count; `depth_unit`: `"passes"`, `"layers"` or `"blocks"`; `series`: array of 1–6 series | `interval_level`: nominal interval coverage strictly between 0 and 1; `timing`: object below |
| Series | `model`, `family`: labels; `passes`: array of 1–32 pass records (may be empty if `adaptive` is supplied) | `path`: `"ordinary"` or `"stock_split"`; `adaptive`: actual policy `{ "accepted": A, "wrong": E }` |
| Pass | `pass`, `depth`: strictly increasing integers in `[1,1000000]`; `correct`: integer `0..N` for this fixed exit | `brier`, `bins`, `thresholds`, `from_previous`, `cuda_ms`, `stored_bytes`, `ties` as below |
| Timing | `statistic`: `"mean"`, `"median"` or `"p95"`; `per`: `"batch"` or `"decision"`; `batch_size`: integer `1..1000000` | None |

`(model, path)` must be unique, and the same model must retain the same family.
An omitted path is *unspecified*, not ordinary. `depth` counts completed work, not a
requested maximum; it must equal `pass` when the depth unit is passes.
Labels are public display names, 1–24 ASCII characters: letters, digits, spaces and
`. , _ + ' " & < > ( ) -`; at least one alphanumeric and no outer spaces. They are
XML-escaped in captions/keys. Figure titles are fixed. Never pass prompts, row IDs,
user names, infrastructure strings, paths, or secret identifiers as display names.

## Pass fields and views

| Output | Input and interpretation |
|---|---|
| `reliability.svg` | `bins`: up to 15 distinct `{index,n,correct,mean_confidence}` objects; optional `accuracy_interval:[lo,hi]` in **accuracy fractions** (`0 ≤ lo ≤ hi ≤ 1`). For `n=0`, supply **only** `{index,n}`. `index=i` is zero-based: `[i/15,(i+1)/15)`, with 1 included in bin 14. The mean of the top-label confidences must lie in that bin. The plot uses the actual mean and `correct/n`, not the bin midpoint. The table numbers bins 1–15: `0` means observed empty, `—` means not supplied; neither gets a dot or whisker. An interval requires group `interval_level` and must already come from the producer's paired-group bootstrap; the renderer does not estimate intervals. |
| `depth.svg` | Separate panels for `correct/N` (%) and optional `brier`: **mean over decisions of the sum over classes** of `(predicted probability − one-hot outcome)²`, unhalved, range `[0,2]`, not a percentage or a mean over classes. Missing scores get no point. |
| `confident_wrong.svg` | `thresholds`: object with **exactly** `"0.80"`, `"0.90"`, `"0.95"`, `"0.99"`. Each value is `{ "accepted": A, "wrong": E }`, where `A` counts confidence **≥** that threshold, `0 ≤ E ≤ A ≤ N`. Both errors and correct accepts must nest as the threshold rises. Curves distinguish `E/N` from `E/A`; the table shows `E`, `A`, and coverage `A/N`. At `A=0`, `E/N=0`, but `E/A` is undefined: a gap, never a zero point. |
| `transitions.svg` | `from_previous`: `{matched,corrections,corruptions}` with optional `overthought`, attached to ending pass `p` when `p−1` is present. Supply **paired** counts on `matched=N`: wrong→right and right→wrong. They must reconcile both accuracy marginals; the renderer never infers flips from the net change. `overthought` counts distinct decisions correct at **any** earlier pass but wrong at `p`. It includes current corruptions, is not additive with them, and requires matched history. Omit it if only adjacent flips were released. |
| `cuda_time.svg` | `cuda_ms`: measured CUDA milliseconds in `[1e−9,1e12]` at this **same operating point**, summarized as declared in group `timing` (required). Hold device, precision, inputs, batching, timing boundaries and protocol comparable upstream. No estimates from pass count; no conversion between per-batch and per-decision time. Color = family; shape distinguishes ordinary, stock split, and unspecified paths. Exact measurements are tabulated. |
| `storage.svg` | `stored_bytes`: **positive integer physical bytes** for stored model artifacts at this setting, under a common inclusion rule. Not a parameter count, logical layer/pass count, activation size or peak VRAM. Do not multiply shared storage by passes. Decimal prefixes on ticks; exact bytes in the table. |
| `selective_risk.svg` | `ties`: nonempty array of `{confidence,n,wrong}`, in **strictly descending** confidence order (`0..1`), one aggregate per distinct, unrounded score, `n>0`, `0 ≤ wrong ≤ n`. Counts must sum to `N`, errors to `N−correct`. Plot one attainable `(cumulative n/N, cumulative wrong/cumulative n)` dot after each **whole tie**. No interpolation, within-tie ordering, or point at zero coverage. Series-level `adaptive` supplies actual accepted decisions and errors on this split, **not** a best threshold or a simulated policy. It has a separate panel/table; zero accepts give undefined risk. |

Complete bin tables must cover N; partial bin totals cannot exceed the marginals.
When ties accompany bin or threshold summaries, counts must agree and bin means must
agree within `1e−9` absolute. Only supported views are returned: no background benchmarks,
synthetic fills, inferred costs, or blank files for missing sections. For instance, the
minimal input returns just `depth.svg`. To keep figures legible, include diagnostics only
at settings of interest; a large release can be rendered in several comparable groups.
