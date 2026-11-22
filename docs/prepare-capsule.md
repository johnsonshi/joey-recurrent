# Prepare the discovery capsule

`scripts/prepare_capsule.py` runs as an ordinary user on the **authorized ingestion host**, not in a worker. It prepares local files. It does not install packages, import downloaded Python, open a registry connection, build an image or run inference.

Use a private, owner-controlled copy of this surface, including `.gitignore`, `scripts/`, `locks/` and the two pinned `src/joey_research/` files. Use Python 3.11+ with its stdlib and a system CA bundle. Examples start at the surface root. On the development Mac, use `python3.14`, not the system `python3`.

### 1. Check the plan (no network)

```sh
python3.14 -IBS scripts/prepare_capsule.py plan
python3.14 -IBS -m unittest discover -s tests
```

The plan checks the five bundled public files against their SHA-256 pins and reports 7 model files, 8 qev files and 40 wheels. Tests create synthetic files only under this surface and block socket connections. They exercise the public `verify_payload` implementation with fixture hashes and a fixture row validator, not production data.

### 2. Stage public inputs (ingestion host only)

```sh
python3.14 -IBS scripts/prepare_capsule.py review-refs discovery-01
# Optional: stage sidecars and reuse the independently materialized tensor now.
python3.14 -IBS scripts/prepare_capsule.py model discovery-01 --resume --model-file "$STOCK_TENSOR"
```

`review-refs` materializes just `serve.patch`, model `LICENSE` and `NOTICE`, and the OneJev `LICENSE` under `.capsules/discovery-01/review/`. Those four SHA-256 pins are not supplied by the model lock. The utility compares each file's length and blob identity to metadata for the exact public revision, then emits `reference-pins.json`. Review the four files and retain the approved hashes on the coordinator. Do not treat a file downloaded from a moving branch as a reference. Neither `review-refs` nor `model` publishes a capsule.

**Reusing the coordinator's tensor:** Set `STOCK_TENSOR` to the absolute path of the authorized `downloads/stock-payload/model.safetensors` on the ingestion host. Both `model` and `prepare` accept `--model-file "$STOCK_TENSOR"` or `--model-cache DIR` (exactly `DIR/model.safetensors`, no directory scan). Relative paths are relative to the invocation directory; `..` and symlinks are refused. The file must be owner-controlled, regular, single-link and beneath an owner-private staging directory. The utility verifies its SHA-256 before copying, hashes the copy again before atomic publication, and never links, changes or logs the original. With either option the weight URL is **removed** from the network allowlist: a bad or missing cache stops the run, with no download fallback. The canonical receipt describes the same public upstream bytes, not the cache location or the independent registry transfer. Omit both options only if an HTTPS weight transfer is authorized or this run already has the verified materialized tensor.

The fetch policy is code, not a caller-provided URL list:

| Input | What binds the bytes |
|---|---|
| Stock checkpoint | `baselabs/joey-1` revision `db5c2a631f2da93c4173378438d8246966e25159`, per-file SHA-256 from the bundled `native.py`. No snapshot, setup script, calibration override or remote model code. |
| Native runtime | Eight raw `qev/*.py` files at `OmniJev/OneJev` revision `81ce62f1597c91e46767d4d02d6ac2e18534fe94`. Check the approved patch SHA **before** applying exact unified-diff hunks in memory. Require all eight postimage hashes from `native.py`. No checkout, external patch command, server or setup hook. |
| Linux wheelhouse | Independent [40-wheel lock](../locks/linux-inference-candidates.json). Resolve 39 exact filenames from version-specific PyPI release metadata and one compiled kernel from its exact GitHub release. Require matching release, filename, size and SHA. Fetch only the content-addressed distribution path or verified release asset ID; rehash the binary. No resolution against the host ABI, sdist or install step. |
| Notices | Retain the three pinned upstream notices, intact wheel archives, copies of embedded notices and the separately pinned FLA license. An empty embedded-notice list or a reported license string is not a distribution grant; review it. |

GETs use direct HTTPS, normal system CA/hostname checks and identity encoding. No ambient proxies, credentials, cookies, user trust stores, key logging, telemetry or URL overrides. Only the originating service's explicitly approved delivery redirects are followed; a signed CDN query stays in memory, never in logs or receipts. On a new redirect host or metadata mismatch, stop and review the policy rather than disabling checks.

### 3. Bind the current study's discovery export

The ingestion utility **does not select a dataset, manufacture replacement rows, or strip labels from a mixed split**. The coordinator supplies the actual, already label-free discovery JSONL, its independently approved byte hash, and the reviewed public worker source closure. Keep the test split, answer keys, configs, credentials, results and other work surfaces out of this exchange.

In the trusted coordinator process, with this distribution's `scripts/` importable, the optional `capsule_build.export_handoff` API writes the fixed ignored `.capsule-input/` directory. It takes **bytes**, not input paths:

```python
from capsule_build import export_handoff

handoff_sha256 = export_handoff(
    discovery_bytes=approved_discovery_jsonl,        # exact bytes, including line endings
    discovery_sha256=approved_discovery_sha256,      # from the study coordinator
    validate_input=input_sha256,                    # trusted joey_research.evaluation function
    public_sources=reviewed_worker_sources,          # {relative_path: (approved_sha256, bytes)}
    entrypoint=approved_worker_entrypoint,          # one exact path in reviewed_worker_sources
    references=approved_reference_pins,             # reviewed reference-pins.json object
)
```

This calls the study validator on each row, preserves byte and option ordering, and refuses to merge into an existing inbox. Include the public `src/joey_research/__init__.py`, `evaluation.py`, actual worker entry point and its local imports. Only individually named, SHA-pinned `src/joey_research/*.py` and `scripts/*.py` files are admitted. Do **not** supply `native.py` or `staging.py`; their reviewed bytes come from this distribution. Nothing scans the checkout for source or imports code from the handoff.

Alternatively, transfer this same reviewed inbox privately to the ingestion host. The inbox and all its directories must be owner-only (0700); files must be real, single-link, owner-only files (0600). `handoff.json` has exactly these fields:

| Field | Value |
|---|---|
| `schema` | `joey-ingress-handoff/v1` |
| `discovery_sha256`, `discovery_fields` | Approved `discovery.jsonl` file SHA-256 and the exact list of top-level fields in each validated row |
| `serve_patch_sha256`, `notices` | Approved patch SHA-256 and `{model-LICENSE, model-NOTICE, OneJev-LICENSE}` → SHA-256 mapping from step 2 |
| `sources` | Explicit relative source paths → approved SHA-256; each file is at the same path under `.capsule-input/` |
| `entrypoint` | The exact worker entry-point path in `sources` (recorded, never executed by ingress) |

Keep the hash of the exact handoff JSON bytes **independently of the inbox**. A matching field list or absence of a field called `label` cannot establish split provenance. For a manual handoff, the coordinator must perform the same `input_sha256` review before approving the file hash. Missing pins are a stop condition.

### 4. Assemble and retain the pins

Set `HANDOFF_SHA256` to the coordinator-held digest; it is not a credential.

```sh
python3.14 -IBS scripts/prepare_capsule.py check-input --handoff-sha256 "$HANDOFF_SHA256"
python3.14 -IBS scripts/prepare_capsule.py prepare discovery-01 --resume --handoff-sha256 "$HANDOFF_SHA256" --model-file "$STOCK_TENSOR"
```

`check-input` is offline. Omit `--resume` when using a **new** run name. A run is private (0700), single-writer and bound to one handoff. An interrupted transfer leaves no final file: resume rehashes the partial, validates the HTTP range, or restarts the full object if the server ignores the range. Completed files must still match their pins. Checksum failures never publish; do not repair them by changing a pin. Use a new name for a different handoff. Use a local filesystem supporting `flock`, `O_NOFOLLOW`, `fsync` and atomic same-filesystem renames.

Only **`.capsules/discovery-01/capsule/`** is a completed result. `work/`, `review/` and `partials/` are not build contexts. Publication checks the exact inventory, rehashes the materialized files, makes directories 0500 and files 0400, then renames the whole tree. Within it:

- `payload/` contains exactly `model/`, `qev/`, `notices/`, `wheelhouse/`, `inputs/discovery.jsonl` (59 files). `receipt.json` is adjacent, **not** in `payload/`.
- `runtime.tar` is a deterministic regular-file-only source package; `runtime-files.json` binds its members. It includes the approved study sources, patched qev and upstream notices, but no wheel, model, discovery row, checkout history or private host metadata.
- `requirements-linux.txt`, `locks/`, `wheel-notices.json` and `notices/` are build/review inputs. `pins.json` records their relevant digests, patch and handoff identities, and payload pins. Retain the success output, including `pins_sha256`, in the private coordinator record **before** transporting the capsule.

Next: [ACR readback and offline local build](offline-build.md). A sealed capsule is not a mirrored image, a successful import, a passing parity gate or a GPU result.
