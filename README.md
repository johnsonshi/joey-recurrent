# Joey recurrence experiments

This independent study asks two different questions about the structured decision model [Joey-1](https://huggingface.co/baselabs/joey-1). Can another traversal of existing layers improve its decisions? Can a smaller checkpoint recover some quality by reusing the layers it retains?

The **Loop** comparison holds the checkpoint fixed. The **Compact-Loop** comparison holds the compact weights fixed and varies their execution depth. An improvement in accuracy is not necessarily an improvement in the probabilities: a model can get more questions right and also become more confident about its mistakes.

Use the original decision interface, not a text-generation surrogate. Treat one-pass equivalence, independent dataset splits and measured readout costs as prerequisites for interpreting results. No training or distillation is part of this study.

## Evidence boundaries

- A CPU fixture checks the implementation; it is not a measurement on Joey.
- Layer visits are not parameter counts, model bytes or measured latency.
- Use frozen, staged inputs at runtime. Keep source weights, third-party data, credentials, raw predictions and deployment-specific files outside Git.
- Do not interpret an example configuration as a model release, a measured result, or an official benchmark score.
