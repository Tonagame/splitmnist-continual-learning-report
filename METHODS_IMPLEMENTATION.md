# Methods Implementation Status

## Important Update

The assignment rule is that we may not submit code taken from GitHub.
Therefore, the classic methods must be implemented by us.

The earlier experiment path used `GMvandeVen/continual-learning` as a runnable
reference implementation. That was useful for setup, learning the protocol, and
getting reference numbers, but it is not enough for the final project under the
new rule.

The clean implementation for submission is now in:

`code/from_scratch/`

## What Is Implemented From Scratch

Main file:

`code/from_scratch/splitmnist_cl.py`

Implemented by us:

| Method | From-scratch status | Notes |
|---|---:|---|
| None | Yes | Sequential fine-tuning over Split MNIST contexts |
| Joint Training | Yes | Trains on the union of all train contexts |
| EWC | Yes | Diagonal Fisher estimate and quadratic parameter penalty |
| LwF | Yes | Teacher snapshot and temperature-scaled logit distillation |
| A-GEM | Yes | Replay buffer and gradient projection against memory gradient |
| Separate Networks | Yes | One MLP per task, Task-CL only |
| Generative Classifier | Yes | Class-conditional diagonal Gaussian statistics |
| LSR-lite | Yes | Real replay, labels, teacher logits, feature anchoring |
| LSR-lite + Fourier | Yes | Adds auxiliary Fourier feature-spectrum loss |
| LSR-lite + ASW | Yes | Adaptive Stability Weighting for KD and feature losses |
| LSR-lite + Fourier + ASW | Yes | Combined ablation |

Also implemented by us:

- Split MNIST construction with five digit-pair contexts
- Class-CL evaluation with no task identity
- Domain-CL binary-domain label mapping
- Task-CL evaluation with task identity / allowed classes
- replay buffers from train data only
- summary.csv writing
- learning_curve.csv writing
- per-run JSON metrics
- a PowerShell runner for long experiments
- a plotting script for from-scratch summaries

## What The Original GitHub Repository Is Used For Now

The original repository should be treated only as a reference target:

https://github.com/GMvandeVen/continual-learning

We can compare our results against its reported/reference behavior, but the
submitted implementation should be the code in `code/from_scratch/`.

## Current Verification

Smoke tests passed on the RTX 3070 for:

- None
- EWC
- LwF
- A-GEM
- Generative Classifier
- LSR-lite + Fourier + ASW
- Separate Networks on Task-CL

The smoke output is local and intentionally ignored by Git:

`results_from_scratch_smoke/`

## What Still Needs To Be Run

To reproduce the serious Split MNIST results with the new independent code, run
the long experiments from:

`code/from_scratch/run_splitmnist_from_scratch.ps1`

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\from_scratch\run_splitmnist_from_scratch.ps1 -Scenario class -Iters 2000 -EvalEvery 100
```

Then repeat for:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\from_scratch\run_splitmnist_from_scratch.ps1 -Scenario domain -Iters 2000 -EvalEvery 100
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\from_scratch\run_splitmnist_from_scratch.ps1 -Scenario task -Iters 2000 -EvalEvery 100
```

After each run, create a graph:

```powershell
E:\conda-envs\continual\python.exe .\code\from_scratch\plot_from_scratch_summary.py --results-dir .\results_from_scratch\splitmnist_class_2000
```

## Bottom Line

The old GitHub-runner path is historical.
The new from-scratch path is the one that matches the assignment rule.
