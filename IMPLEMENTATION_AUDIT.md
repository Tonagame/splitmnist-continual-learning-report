# Implementation Audit And Reproduction Status

## Purpose

This document checks whether the project is "up to code" after the assignment
requirement changed.

The important rule is:

> We should not submit code copied from GitHub. We must implement the continual
> learning methods ourselves and use the original repository / paper only as
> reference targets.

## Short Answer

The project now has a clean-room implementation under:

`code/from_scratch/`

This is the code path that should be presented as the assignment implementation.

The older GMvandeVen-based runs are still useful as reference results, but they
should not be presented as the final implementation.

## What Is Implemented By Us

Main file:

`code/from_scratch/splitmnist_cl.py`

Implemented from scratch:

| Component | Status | Notes |
|---|---:|---|
| Split MNIST construction | Done | Five contexts: `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, `(8,9)` |
| Class-CL protocol | Done | No task identity during evaluation |
| Domain-CL protocol | Done | Shared binary label space per digit pair |
| Task-CL protocol | Fixed | Uses allowed classes during training and evaluation |
| MLP classifier | Done | Simple PyTorch network |
| None | Done | Sequential fine-tuning |
| Joint | Done | Trains on all training contexts together |
| EWC | Partial reproduction | Implemented, but still below GMvandeVen results in Task-CL |
| LwF | Done | Teacher snapshot + logit distillation |
| A-GEM | Done | Replay memory + gradient projection |
| Separate Networks | Done | Task-CL only |
| Generative Classifier | Done, but simple | Diagonal Gaussian classifier; not equivalent to the paper's stronger method |
| LSR-lite variants | Done | Replay + labels + logits + feature anchoring + optional Fourier / ASW |
| CSV logging | Done | `summary.csv`, `learning_curve.csv`, metrics JSON |
| Graph generation | Done | Aggregation and comparison scripts |

## What We Fixed

### Task-CL Training Protocol

Initial issue:

The first from-scratch Task-CL run used full 10-class cross entropy during
training for `None` and `EWC`.

Why this was wrong:

Task-CL assumes task identity is available. That means the model should train
and evaluate against the active task's allowed classes.

Fix:

`splitmnist_cl.py` now uses `supervised_context_loss(...)`.

For Task-CL, this function:

1. Selects only the two active class logits.
2. Remaps labels to `0` or `1`.
3. Computes cross entropy on that task-specific two-class problem.

This same corrected loss is also used when estimating Fisher information for
EWC.

## Result Of The Fix

Task-CL results before and after the protocol fix:

| Method | Before fix | After fix | GMvandeVen reference |
|---|---:|---:|---:|
| None | 67.14% | 84.91% | 87.65% |
| EWC | 64.75% | 88.06% | 99.52% |

Interpretation:

- `None` is now close to the reference and the paper.
- `EWC` improved a lot, but is still not a faithful reproduction.

## Why EWC Is Still Not Fully Reproduced

Our EWC is a clean and valid simplified implementation, but it likely does not
match the exact details in the GMvandeVen repository.

Likely reasons:

- Fisher estimation is simplified.
- Fisher scaling / normalization may differ.
- The original repository may use better tuned EWC defaults.
- The original implementation may handle active classes and consolidation in a
  more specialized way.
- Increasing `ewc-lambda` from `5000` to `50000` did not close the gap, so the
  issue is not only one scalar hyperparameter.

Current honest status:

> EWC is implemented, but only partially reproduces the GMvandeVen Task-CL
> result.

This should be stated clearly in the report.

## Generative Classifier Note

Our clean-room Generative Classifier is a simple diagonal Gaussian classifier.
It stores class statistics:

```text
count, sum, sum of squares
```

This is a valid simple generative classifier, but it is not necessarily the same
as the stronger generative method reported in the paper.

In Task-CL, our implementation can run because task identity limits evaluation
to the active classes. This should be treated as an extra clean-room variant,
not as a direct reproduction of the original repository's supported setup.

## What Results Should Be Used In The Report

Use three categories:

| Category | Use in report? | Meaning |
|---|---:|---|
| Paper Table 2 | Yes | Published reference |
| GMvandeVen code run | Yes, as reference only | Shows expected behavior of the original repository |
| Our from-scratch code | Yes, as main implementation | This is the assignment-compliant result |

Do not present the old GMvandeVen-code run as our implementation.

## Current Important Files

Code:

- `code/from_scratch/splitmnist_cl.py`
- `code/from_scratch/run_all_classic_from_scratch.ps1`
- `code/from_scratch/aggregate_from_scratch.py`
- `code/from_scratch/compare_reproduction_sources.py`

Documentation:

- `CODE_EXPLANATION.md`
- `METHODS_IMPLEMENTATION.md`
- `PAPER_COMPARISON.md`
- `README.md`

Graphs and CSVs:

- `assets/paper_vs_gmvandeven_vs_from_scratch.png`
- `assets/paper_vs_gmvandeven_vs_from_scratch.csv`
- `assets/from_scratch_classic_no_lsr_2000_summary.csv`
- `assets/from_scratch_classic_no_lsr_2000_final_accuracy.png`
- `assets/from_scratch_classic_no_lsr_2000_learning_curves.png`

## Final Assessment

The project is now in a much stronger and more honest state:

- The core code is implemented by us.
- The Task-CL protocol bug was found and fixed.
- The comparison graph clearly separates paper, original-code reference, and our
  from-scratch implementation.
- Known limitations are documented instead of hidden.

Remaining weakness:

- EWC still needs deeper reproduction work if the goal is to match the original
  repository's Task-CL result near 99%.

Recommended phrasing for the final report:

> We implemented all methods independently. Most reproduced the expected trends.
> Joint, A-GEM, LwF, and Task-CL None are close to reference behavior. EWC is a
> partial reproduction: the method is implemented, but our simplified Fisher /
> consolidation details do not fully match the original repository's Task-CL
> result.
