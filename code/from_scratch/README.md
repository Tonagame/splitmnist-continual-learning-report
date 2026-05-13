# Clean-Room Split MNIST Continual Learning Implementation

This folder contains a from-scratch PyTorch implementation for the project.
It does not import or copy code from `GMvandeVen/continual-learning`.

The original repository can still be used as a reference target for expected
results, but the code in this folder is the implementation that should be used
when the assignment requires us to implement the methods ourselves.

## Main File

`splitmnist_cl.py`

It implements:

- Split MNIST dataset construction
- Class-CL, Domain-CL, and Task-CL evaluation protocols
- MLP classifier
- None / sequential fine-tuning
- Joint training
- Elastic Weight Consolidation (EWC)
- Learning without Forgetting (LwF)
- Average Gradient Episodic Memory (A-GEM)
- Separate Networks for Task-CL
- Generative Classifier using class-conditional diagonal Gaussian statistics
- LSR-lite
- LSR-lite + Fourier
- LSR-lite + ASW
- LSR-lite + Fourier + ASW
- CSV learning-curve logging
- summary.csv and per-run metrics JSON files
- combined summary and graph aggregation

## Smoke Test

Example:

```powershell
E:\conda-envs\continual\python.exe .\code\from_scratch\splitmnist_cl.py `
  --method none `
  --scenario class `
  --iters 1 `
  --batch 16 `
  --acc-n 64 `
  --eval-every 1 `
  --hidden 64 `
  --results-dir .\results_from_scratch_smoke
```

## Serious Split MNIST Runs

Run the classic methods for all scenarios, without LSR variants:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\from_scratch\run_all_classic_from_scratch.ps1 `
  -Iters 2000 `
  -EvalEvery 100
```

This creates:

- `results_from_scratch/classic_no_lsr_2000/combined_summary.csv`
- `results_from_scratch/classic_no_lsr_2000/combined_learning_curve.csv`
- `results_from_scratch/classic_no_lsr_2000/from_scratch_classic_final_accuracy.png`
- `results_from_scratch/classic_no_lsr_2000/from_scratch_classic_learning_curves.png`
- `results_from_scratch/classic_no_lsr_2000/FROM_SCRATCH_CLASSIC_REPORT.md`

To create the reproduction comparison against the paper and the earlier
GMvandeVen-code run:

```powershell
E:\conda-envs\continual\python.exe .\code\from_scratch\compare_reproduction_sources.py
```

This creates:

- `assets/paper_vs_gmvandeven_vs_from_scratch.csv`
- `assets/paper_vs_gmvandeven_vs_from_scratch.png`

Class-CL example:

```powershell
E:\conda-envs\continual\python.exe .\code\from_scratch\splitmnist_cl.py `
  --method ewc `
  --scenario class `
  --contexts 5 `
  --iters 2000 `
  --batch 128 `
  --acc-n 1024 `
  --eval-every 100 `
  --memory-per-class 100 `
  --results-dir .\results_from_scratch\splitmnist_class_2000
```

Use `--method` with:

```text
none
joint
ewc
lwf
agem
gen-classifier
lsr-lite
lsr-lite-fourier
lsr-lite-asw
lsr-lite-fourier-asw
separate
```

`separate` is only valid for `--scenario task`.

## Important Protocol Notes

- Test data is used only for evaluation.
- Replay buffers are built only from train data.
- A-GEM and LSR variants use the same default budget: 100 samples per original digit class.
- Class-CL evaluates over all 10 classes with no task identity.
- Task-CL uses task identity by masking to the two allowed classes for the current task.
- Task-CL training also uses only the active task's two allowed classes for the supervised loss.
- Domain-CL maps each digit pair to a shared binary label space.

## Current Status

Smoke tests passed for:

- None
- EWC
- LwF
- A-GEM
- Generative Classifier
- LSR-lite + Fourier + ASW
- Separate Networks on Task-CL

The long 2000-iteration from-scratch reproduction still needs to be run and
compared against the previous GMvandeVen reference results.
