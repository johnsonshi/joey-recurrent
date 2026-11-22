# Comparing frozen SwiGLU channels

`scripts/mlp_channel_alignment.py` answers one question: **does a channel permutation chosen on one half of the hidden coordinates reduce weight differences on the other half?** This is a geometric comparison, not a test of model quality. It takes arrays in memory; it does not load checkpoints, update weights, use tokens or labels, or run a model.

## Host contract

```python
P, d = analyze_pair(earlier, later, seed=seed, sketch=64, shortlist=16, null_reps=32)
```

Both mappings supply finite float32 NumPy arrays or memmaps: `gate` and `up` are `(M,H)`, `down` is `(H,M)`. Shapes must agree across layers; `M >= 1` and `H >= 2` must permit two exhaustive, equal halves (`H` even). Read-only and noncontiguous inputs are supported. `seed` is a nonnegative integer; `sketch` and `shortlist` are positive integers, clipped to the available coordinates and destinations, respectively. `null_reps` is a nonnegative integer.

`P[j]` is the **later** channel assigned to **earlier** channel `j`. Thus the aligned later matrices are logically `gate[P, :]`, `up[P, :]`, and `down[:, P]`. There is no sign change, rescaling, or hidden-coordinate permutation of the weights. `P` is an index array checked to contain `0..M-1` exactly once. `d` is a plain dictionary (`schema_version = 1`). Its `permutation_sha256` is the lowercase hex digest of `np.asarray(P, dtype="<i8").tobytes(order="C")`: signed little-endian int64 elements in source order, no shape, length, or other header. `permutation_encoding` records this convention.

## Which coordinates choose P?

`coordinate_split(H, *, seed)` is public. It makes exactly one draw, `np.random.Generator(np.random.PCG64(seed)).permutation(H)`. The first `H//2` indices are fit, the rest audit. `d["coordinates"]` contains `seed`, `generator`, and the actual `fit` and `audit` lists **in draw order**. The `sketch` list is the first `min(sketch,H//2)` fit indices sorted into coordinate order for arithmetic. All three projections use those same indices. Neither the rest of fit nor any audit value is used to construct or normalize the sketches.

For each channel, take its gate row, up row and down **column** on the sketch. Unit-normalize each segment separately in FP64, leaving zero segments zero. Concatenate them; the dot product divided by three is the signed mean cosine. Zero segments still count in that denominator. This normalization is only for choosing P, never for scoring weight differences.

1. In row tiles, compute scores against all destinations. Retain each row's top `shortlist`; when the cutoff is tied, take smaller destination IDs first.
2. Greedily accept nonconflicting edges in order of decreasing score, increasing source, then increasing destination. A heap merges the ranked row lists in that order; it need only carry the head of each unassigned row. Wide lists are paged, not materialized as a dense cost matrix.
3. Visit unmatched sources in increasing order, recompute their sketch scores, and take the best still-free destination, breaking ties by smaller destination ID. `d["matching"]` records the effective `shortlist`, `shortlist_matches`, and the number assigned by `fallback`.

This is not a globally optimal assignment. Fit, audit, random controls and action checks are reported for the *same* P. None is used to switch to identity, choose a seed, or retry. Exact numeric ties have deterministic ordering; nearly tied scores need not be bit-identical across BLAS implementations. For repeatable runs the host can fix its CPU BLAS build and thread count before importing NumPy.

## Exact metrics and pooling across pairs

`d["fit"]` scores **all fit coordinates**, not just the sketch. `d["audit"]` independently scores **all audit coordinates**. Each split has the following fields:

| Path within split | Value |
|---|---|
| `projections[name]["count"]` | Number of raw entries, `M * H//2` for each of `gate`, `up`, `down`. |
| `projections[name]["earlier_ss"]`, `["later_ss"]` | Sums of squared weights; reordering leaves the later sum unchanged. |
| `projections[name]["identity_dot"]`, `["aligned_dot"]` | Frobenius inner products of earlier with later, in natural order or gathered by P. |
| `projections[name]["identity_cosine"]`, `["aligned_cosine"]` | Corresponding dot divided by `sqrt(earlier_ss) * sqrt(later_ss)`; `None` if either norm is zero. |
| `projections[name]["identity_sse"]`, `["aligned_sse"]` | Sums of squared **entrywise differences** in FP64. Not differences of large norm/dot terms. |
| `pooled["count"]`, `["identity_sse"]`, `["aligned_sse"]` | Sums of the respective raw values across the three projections. |
| `pooled["fraction_sse_removed"]` | `1 - aligned_sse / identity_sse`. Negative means worse than identity; `None` when identity SSE is zero. |

**Across pairs**, sum `d["audit"]["pooled"]["aligned_sse"]` and `d["audit"]["pooled"]["identity_sse"]` separately, then take `1 - sum_aligned / sum_identity` if the latter sum is nonzero. Do not average layer percentages or drop zero-identity pairs from the sums. Use `fit` instead for a separate fit result. For an MSE, divide the corresponding SSE sum by the sum of `pooled["count"]`.

For an independent audit check, let `C = d["coordinates"]["audit"]`. For each channel `j` and `c` in `C`, accumulate `(float(earlier[name][j,c]) - float(later[name][P[j],c]))**2` for `gate` and `up`; for `down` use indices `[c,j]` and `[c,P[j]]`. Substitute `j` for `P[j]` for identity. The core visits increasing source tiles and sorted coordinates, reduces in FP64, and combines tile totals with `math.fsum`. “Exact” means exhaustive entries, not exact real arithmetic.

## Shuffled control: a probe, not the exact audit metric

`d["null_probe"]` uses the first `min(8,H//2)` indices of the original audit draw, sorted. Its `coordinates`, `count`, `identity_sse`, `aligned_sse`, and list `shuffled_sse` are **raw, unscaled, pooled probe** values. Compare the shuffled values only to P and identity **on this probe**. A probe position has the same inclusion probability as any other audit coordinate; the probe is not selected by observed performance.

Replicates use successive `rng.permutation(M)` draws with `rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(1,))))`. Each map is shared across all three projections. There are `null_reps` draws (default 32; zero yields an empty list); no all-pairs products or full-audit rescoring per draw. The null stream is independent of the split stream. No probe result is an acceptance criterion.

## Action check

For **later versus itself reordered by P**, evaluate `down @ (SiLU(gate @ x) * (up @ x))` in FP64 on two fixed unit-length sign vectors (PCG64 seed 0), the first scaled by `2**-24`, and an exact-zero vector. All `M` channels and all `H` outputs participate. Reordering both input projection rows and the corresponding output columns preserves the sum in exact arithmetic for **any** bijection, good or bad.

`d["action_check"]["vectors"]` has entries `unit_0`, `unit_1`, `near_zero`, `zero`. `reference_l2` is the norm of the full original output; `absolute_l2` is the norm of the **output difference**, not the difference of two norms; `relative_l2` divides it by `reference_l2`, or is `None` for exact zero. The stable sigmoid avoids exponential overflow. BLAS sums within channel tiles and Kahan compensation combines the tile outputs. Reordering can change rounding: `passed` tests `absolute_l2 <= atol + rtol * reference_l2`, with `atol=1e-12`, `rtol=1e-10`. The check also reports whether all four vectors passed. It does not establish equivalence between the two different layers.

## Scratch space and local tests

`d["workspace"]` reports the row counts and byte sizes of the reusable similarity tile and maximum full-width FP64 weight tile (`pair_tile_*`, `weight_tile_*`). Each is capped at 4 MiB. Exact scoring keeps only a few weight tiles at a time; transposing `down` is a view. The two feature matrices and the cached candidate IDs/scores are guarded before allocation (128 MiB combined state limit; the two raw null-probe matrices use a separate phase under the same limit). There is also linear bookkeeping, Python/NumPy overhead and native BLAS workspace; these byte counts are **not** a process RSS limit.

At `(M,H)=(6144,2048)` with defaults and 64-bit indices, the similarity tile is 3 MiB, a full-width weight tile 2 MiB, the sketch pair 18 MiB, the candidate cache 1.5 MiB and the probe pair 2.25 MiB. The quadratic computation is tiled dot products on `3 * sketch_width` features, not an `M × M × H` broadcast. Fallback revisits tiles with unmatched rows. At most 64 candidates per source are cached; very wide shortlists can require extra tile passes.

From the repository root, run `python3 -B -m unittest discover -s tests -p test_mlp_channel_alignment.py`. The suite uses only invented arrays and skips the numerical tests if NumPy is absent. The numerical module itself needs Python 3.9+ and NumPy. It has no storage, network or model-runtime dependency.

For a local measurement, `scripts/measure_channel_alignment.py` accepts a materialized, read-only copy of the pinned full safetensors checkpoint, the frozen protocol SHA-256, and a fresh absolute output file in an owner-only directory. It checks the whole source digest, independently recomputes each projection's held-out SSE, and reports the fixed geometry gate. Its output contains numeric measurements and permutation indices, not a model or inference result.

## Separate test of normalization scales

`scripts/norm_projection_geometry.py` does not permute channels or fit any parameters. It asks whether the original normalization vectors from a removed layer make the retained layer's direct input projections resemble the removed layer's projections. For retained matrix `A`, removed matrix `B`, and their stored normalization vectors `a` and `b`, the three effective maps are `A diag(1+a)`, `A diag(1+b)`, and the reference `B diag(1+b)`. The `1+` comes from the verified decoder's RMSNorm implementation. Scale **input columns**, not output rows.

The host verifies the full model, physical compact file, sidecar, decoder wheel and protocol pins. It includes only projections that immediately consume the normalized hidden state, and accumulates the complete squared errors in row tiles. It independently checks them from column norms and dot products. Pool by summing the squared errors before dividing. A negative fraction of the gap removed means the donor scales made this geometric comparison worse. This is not activation-weighted, does not simulate BF16 activation rounding, and says nothing by itself about a model decision or latency.
