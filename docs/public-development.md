# BANKING77 train-derived routing diagnostic

This is a **development** sample of public complaints from the original training partition. It measures routing across all 77 original intents. It is neither the official BANKING77 test score nor a Decision Index or Jev leaderboard result. Public training data can overlap model pretraining; performance here is not independent evidence of generalization.

## Host staging

From the package root, on the **coordinator's host, never in a pod**, set `PYTHON` to an installed Python 3.10+ executable (the macOS system `python3` may be 3.9). Set `PRIVATE_BUNDLE` to a new absolute path outside any repository. Its parent directory must already exist.

```sh
umask 077
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" -B -m benchmarks.banking77_development \
  stage --fetch --out "$PRIVATE_BUNDLE"
```

Record the printed manifest SHA-256 independently. To stage already authorized local files without network access, replace `--fetch` with `--source-dir "$LOCAL_SOURCE_DIR"`. The importer selects exactly two files by name, path, byte length, and SHA-256. It does not list the source directory, load an HF builder, follow HTTP redirects, or request the official test file. A failed stage may leave a private incomplete directory; it is not usable without a matching completed manifest. Do not stage into a repository, mount the custody directory for inference, or transfer any source CSV to a worker.

| Only permitted download URLs | Bytes | SHA-256 of file bytes |
| --- | ---: | --- |
| [`train.csv`](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv) | 839073 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| [`categories.json`](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/categories.json) | 2036 | `53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63` |

These are the original PolyAI GitHub files, not converted Parquet. [PolyAI/banking77 builder metadata](https://huggingface.co/datasets/PolyAI/banking77/blob/90d4e2ee5521c04fc1488f065b8b083658768c57/dataset_infos.json) records the original train hash, byte count and 10,003 records. The [upstream tree](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) fixes the revision and both file identities. The [Decision Index acquisition manifest](https://github.com/apolinario/decision-index/blob/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec/decision_index/suite/build/acquire.py) independently pins the category file. Source attribution: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), PolyAI; [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE). State the train selection, text grouping and format changes in any derived report. This package includes no source rows.

## Frozen selection and projections

Salt `joey:banking77:train-development:v1`. Complaints are grouped by Unicode NFKC, case folding, then whitespace collapse. Group hashes depend only on normalized text. The first hex digit assigns one of 16 strata; a separately domain-tagged hash ranks whole groups within each. Select 32 groups per stratum, or fail if any stratum is undersized. This yields exactly **512 distinct normalized complaint groups**; all original duplicate records remain with their group. Labels and predictions never select records. Category balance and semantic independence are not guaranteed. Rows have opaque IDs and are emitted in hash-ID order, not source-CSV order. The unused groups are not a second benchmark or a campaign holdout.

| Output | Consumer and contents |
| --- | --- |
| `inference/inputs.jsonl` | Generic Joey runner. `{id,group_id,task,prompt,choices}`; give the model only the prompt and all 77 ordered descriptions. |
| `inference/requests.jsonl` | Native-shaped runner. Each wrapper is `{id,input_sha256,payload}`. Send **only** `payload`: empty `state`, one `questions.q1` of type `choice`, instructions containing the original utterance, and criteria `option_0` ... `option_76` in original vocabulary order. These are not official run IDs. |
| `custody/labels.jsonl` | Custodian only. `{id,input_sha256,gold_index}` for the selected train-derived states only. No gold or per-state source metadata appears in either inference projection. |
| `custody/groups.jsonl`, `coverage.json`, `manifest.json` | Full label-free group assignment and opaque membership; source/selected/duplicate counts and per-stratum supports; source and output byte hashes. The original files are not copied into the bundle. |

## Score one frozen arm

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" -B -m benchmarks.banking77_development \
  score --bundle "$PRIVATE_BUNDLE" --manifest-sha256 "$MANIFEST_PIN" \
  --projection native --predictions "$PREDICTION_JSONL"
```

For `native`, submit `{id,input_sha256,status,response}` when `status="ok"`; `response.answers.q1` must have `type:"choice"`, the declared `choice` key, and a probability map covering all 77 keys. For `unsupported`, `abstained`, `error` or `pending`, omit `response`. For `--projection research`, use `{id,input_sha256,probs}` in **numeric option order** or `{id,input_sha256,error}`. Never shrink the menu to fit a runner. Missing IDs are counted as pending; duplicate, unknown, or stale IDs are errors.

`observed_class_macro_f1` averages F1 over the union of gold and predicted classes on completed choices. For native-shaped responses it uses the declared choice, not necessarily the probability argmax. For the generic projection it uses the first argmax. Always report `states_with_choice`, `choice_coverage`, `scored_gold_or_predicted_classes`, `groups_with_any_choice`, and the failure counts alongside F1. This is row-weighted within class; duplicate records are not independent bootstrap units. `probability_metrics` contains full-population NLL, sum-over-class Brier, 15-bin ECE and argmax accuracy. Proper scores are withheld for incomplete or invalid distributions. A native vector may meet the kit's `0.01` mass tolerance but not Joey's `1e-6`; the report retains the different supports. A zero probability on gold gives infinite NLL, represented by `null` and `nll_status="infinite"`.

Python API: `stage(Path(out), fetch=True)` or `stage(Path(out), source_dir=Path(local_dir))`; `split_groups(Iterable[BlindRow])`; `score_development(inputs, labels, predictions, projection="native"|"research")`. The CLI verifies the bundle pin; the in-memory score API assumes the caller has already verified it.
