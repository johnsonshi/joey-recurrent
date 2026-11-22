# Joey recurrence experiments

This independent study asks two different questions about the structured decision model [Joey-1](https://huggingface.co/baselabs/joey-1). Can another traversal of existing layers improve its decisions? Can a smaller checkpoint recover some quality by reusing the layers it retains?

The **Loop** comparison holds the checkpoint fixed. The **Compact-Loop** comparison holds the compact weights fixed and varies their execution depth. An improvement in accuracy is not necessarily an improvement in the probabilities: a model can get more questions right and also become more confident about its mistakes.

Use the original decision interface, not a text-generation surrogate. Treat one-pass equivalence, independent dataset splits and measured readout costs as prerequisites for interpreting results. No training or distillation is part of this study.

## Evidence boundaries

- A CPU fixture checks the implementation; it is not a measurement on Joey.
- Layer visits are not parameter counts, model bytes or measured latency.
- Use frozen, staged inputs at runtime. Keep source weights, third-party data, credentials, raw predictions and deployment-specific files outside Git.
- Do not interpret an example configuration as a model release, a measured result, or an official benchmark score.

## Running the implementation

The portable fixtures require Python 3.10+ and can run without model weights:

```sh
PYTHONPATH=src:. python3 -B -m unittest discover -s tests -q
```

For model execution, start with the [native runtime and stock parity checks](docs/native-runtime.md). The [Loop interface](docs/loop.md) changes the execution order without changing source tensors. The [Compact exporter](docs/compact.md) builds a smaller file, and the [Compact runtime](docs/native-compact.md) checks the reloaded model against an independent construction. The [discovery worker](docs/worker.md) measures one staged shard without receiving labels.

The [evaluator](docs/evaluation.md) keeps gold labels separate and scores paired predictions at fixed menu order. The [two-slot scheduler](docs/scheduler.md) enforces leases, frozen selection and time cutoffs; it is not a cloud provisioner. The [study configuration](configs/study.json) defines the initial candidates and calibration thresholds before model results are known.

For development analyses, see the [original-training-only public benchmark](docs/public-development.md), [native inventory endpoints](docs/inventory-endpoints.md), [one-probe replay](docs/single-probe-replay.md), [measured trace cost accounting](docs/trace-cost.md) and [evidence figure format](docs/decision-figures.md). A replay can compare decisions from observed passes, but it cannot establish the latency of a different execution policy.
